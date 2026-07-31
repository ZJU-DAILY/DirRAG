"""
DirRAG v3.0 — Two-Stage Hierarchical Retrieval-Augmented Generation
====================================================================
Stage 1: Section routing — coarse-grained directory-level matching
Stage 2: Chunk retrieval — fine-grained text-level search within routed sections

Per the design doc, each query gets its OWN freshly-built index
(for Qasper which has no persistent document collection).
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import math
import uuid
import heapq
import shutil
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm

# ── bring in the project-level singletons ──────────────────────────────────
import sys
sys.path.insert(0, "/root/dir-sem-rag/gte-Qwen2-1.5B-instruct")

from embedding import EmbeddingModel, EMBEDDING_DIM
from llm import LocalLLMClient, llm_judge
from data_utils import (
    load_completed_sample_ids,
    save_single_result,
    calculate_qa_metrics,
)

# ═══════════════════════════════════════════════════════════════════════════
# ──────────────────────────  CONFIG  ───────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

CHUNK_MIN_TOKENS   = 100
CHUNK_MAX_TOKENS   = 500

# Section embedding blend weights
TITLE_WEIGHT   = 0.7
SUMMARY_WEIGHT = 0.3
SUMMARY_CHARS  = 200

# Scoring hyper-params (Section stage)
SEC_ALPHA = 0.5   # semantic similarity weight
SEC_BETA  = 0.2   # title BM25 weight
SEC_GAMMA = 0.2   # entity match weight
SEC_DELTA = 0.1   # depth reward weight

# Scoring hyper-params (Chunk stage)
CHK_ALPHA = 0.5   # semantic similarity weight
CHK_BETA  = 0.2   # BM25 weight
CHK_GAMMA = 0.2   # entity match weight
CHK_DELTA = 0.1   # position prior weight

SECTION_TOP_K   = 10
CHUNK_ANCHOR_N  = 20
NEIGHBOR_WINDOW = 2

USED_SECTION_PENALTY = 0.1
MAX_ITER             = 5

BM25_K1 = 1.5
BM25_B  = 0.75

ENTITY_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')

STOP_WORDS = {
    "a","an","the","is","are","was","were","and","or","but",
    "in","on","at","to","for","of","with","by","from","that",
    "this","it","its","be","as","do","did","has","have","had",
}

# Milvus entity index config
ENTITY_MILVUS_DIR      = "/tmp/dirrag_entity_milvus"
ENTITY_COLLECTION_NAME = "entities"
ENTITY_FUZZY_THRESHOLD = 0.85

OUTPUT_PATH = "./dir_rag_qasper_results.jsonl"
INPUT_PATH  = "/root/dir-sem-rag/longbench/test/clean_qasper/qasper_cleaned.jsonl"

# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────  DATA STRUCTURES  ───────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    chunk_id:     str
    section_id:   str
    text:         str
    embedding:    Optional[np.ndarray] = None
    entities:     List[str]            = field(default_factory=list)
    index_in_doc: int                  = 0


@dataclass
class Section:
    section_id:   str
    title:        str
    path:         List[str]            = field(default_factory=list)
    depth:        int                  = 1
    parent_id:    Optional[str]        = None
    embedding:    Optional[np.ndarray] = None
    entities:     List[str]            = field(default_factory=list)
    children_ids: List[str]            = field(default_factory=list)
    chunk_ids:    List[str]            = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# ──────────────────────────  DATA LOADING  ─────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def load_qasper_detailed(input_path: str) -> List[Dict]:
    """
    Load Qasper data from a JSONL file.

    Each line is expected to be a JSON object with the following fields:
        {
          "queryId":   <str>,       # unique sample identifier
          "question":  <str>,
          "answer":    <str>,       # gold answer
          "max_depth": <int>,
          "content": [
            {"path": <str>, "text": <str>, "depth": <int>},
            ...
          ]
        }

    Returns a list of dicts with normalised keys used by the main loop:
        {
          "_id":      <str>,        # mapped from queryId
          "question": <str>,
          "gold_answer": <str>,     # mapped from answer
          "content":  [...]         # passed through as-is for passage building
        }
    """
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            samples.append({
                "_id":         raw["queryId"],
                "question":    raw["question"],
                "gold_answer": raw["answer"],
                "content":     raw["content"],   # list of {path, text, depth}
            })
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────  HELPERS  ───────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def extract_entities(text: str, llm: LocalLLMClient = None) -> List[str]:
    """
    Use LLM to extract clean, meaningful entities (persons, locations,
    organizations, events, laws, cases, works, proper nouns) from text.
    Returns deduplicated entities in original order.
    """
    if not text or len(text.strip()) < 10:
        return []

    prompt = f"""
You are a precise entity extractor.
Extract ALL important named entities from the following text.
Entities include: people, places, organizations, cases, laws, events, titles, proper nouns.

Output ONLY a valid JSON list of strings. No extra words, no explanations, no markdown.

Text: {text}
""".strip()

    try:
        raw_response = llm.chat(prompt)
        cleaned = re.sub(r"```(?:json)?|```", "", raw_response).strip()
        entities = json.loads(cleaned)
        if not isinstance(entities, list):
            return []
        entities = [str(e).strip() for e in entities if e and str(e).strip()]
        return list(dict.fromkeys(entities))
    except Exception:
        return []


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def extract_keywords(text: str) -> List[str]:
    return list(dict.fromkeys(tokenize(text)))


def split_into_chunks(text: str, section_id: str, doc_offset: int = 0,
                      llm: LocalLLMClient = None) -> List[Chunk]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: List[Chunk] = []
    buffer: List[str]   = []
    buf_tokens = 0

    def flush(buf: List[str], idx: int, llm: LocalLLMClient = None) -> Chunk:
        txt = " ".join(buf)
        return Chunk(
            chunk_id     = str(uuid.uuid4()),
            section_id   = section_id,
            text         = txt,
            entities     = extract_entities(txt, llm),
            index_in_doc = idx,
        )

    for sent in sentences:
        n_tok = len(sent.split())
        if buf_tokens + n_tok > CHUNK_MAX_TOKENS and buffer:
            chunks.append(flush(buffer, doc_offset + len(chunks), llm))
            buffer, buf_tokens = [], 0
        buffer.append(sent)
        buf_tokens += n_tok

    if buffer:
        chunks.append(flush(buffer, doc_offset + len(chunks), llm))

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# ───────────────────────  BM25 (in-memory)  ────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class BM25Index:

    def __init__(self, corpus: List[str]):
        self.corpus    = corpus
        self.n_docs    = len(corpus)
        self.tokenized = [tokenize(doc) for doc in corpus]
        self.avgdl     = (sum(len(t) for t in self.tokenized) / max(self.n_docs, 1))
        self._build_idf()

    def _build_idf(self):
        df: Dict[str, int] = defaultdict(int)
        for toks in self.tokenized:
            for tok in set(toks):
                df[tok] += 1
        self.idf: Dict[str, float] = {}
        for tok, freq in df.items():
            self.idf[tok] = math.log(
                (self.n_docs - freq + 0.5) / (freq + 0.5) + 1
            )

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        toks   = self.tokenized[doc_idx]
        dl     = len(toks)
        tf_map = defaultdict(int)
        for t in toks:
            tf_map[t] += 1
        score = 0.0
        for qt in query_tokens:
            if qt not in tf_map:
                continue
            tf  = tf_map[qt]
            idf = self.idf.get(qt, 0.0)
            score += idf * (
                tf * (BM25_K1 + 1)
                / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / max(self.avgdl, 1)))
            )
        return score

    def top_k(self, query_tokens: List[str], k: int) -> List[Tuple[int, float]]:
        scores = [(i, self.score(query_tokens, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ═══════════════════════════════════════════════════════════════════════════
# ──────────────────────────  ENTITY INDEX  ─────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class EntityInvertedIndex:

    def __init__(self, embed_model: EmbeddingModel, milvus_path: str):
        from pymilvus import MilvusClient

        self._embed_model = embed_model
        self._milvus_path = milvus_path
        self._client = MilvusClient(milvus_path)
        self._chunk_to_section: Dict[str, str] = {}

        self._client.create_collection(
            collection_name = ENTITY_COLLECTION_NAME,
            dimension       = EMBEDDING_DIM,
            metric_type     = "COSINE",
        )

        self._pending: List[Dict] = []

    def add_chunk(self, chunk: Chunk, llm: LocalLLMClient = None):
        self._chunk_to_section[chunk.chunk_id] = chunk.section_id

        first_sent_entities = set(
            extract_entities(chunk.text.split('.')[0], llm)
        )
        tf_map: Dict[str, int] = defaultdict(int)
        for ent in chunk.entities:
            tf_map[ent] += 1

        for ent, tf in tf_map.items():
            bonus  = 0.3 if ent in first_sent_entities else 0.0
            weight = tf + bonus
            self._pending.append({
                "entity_text": ent,
                "chunk_id":    chunk.chunk_id,
                "section_id":  chunk.section_id,
                "weight":      weight,
            })

    def flush(self):
        if not self._pending:
            return

        texts = [r["entity_text"] for r in self._pending]
        embs  = self._embed_model.encode(texts)

        rows = [
            {
                "id":          i,
                "vector":      embs[i].tolist(),
                "entity_text": record["entity_text"],
                "chunk_id":    record["chunk_id"],
                "section_id":  record["section_id"],
                "weight":      record["weight"],
            }
            for i, record in enumerate(self._pending)
        ]

        self._client.insert(
            collection_name = ENTITY_COLLECTION_NAME,
            data            = rows,
        )
        self._pending.clear()

    def lookup_entities(
        self,
        query_entities: List[str],
    ) -> Tuple[Set[str], Set[str]]:
        if not query_entities:
            return set(), set()

        q_embs = self._embed_model.encode(query_entities)

        matched_chunks:   Set[str] = set()
        matched_sections: Set[str] = set()

        for emb in q_embs:
            results = self._client.search(
                collection_name = ENTITY_COLLECTION_NAME,
                data            = [emb.tolist()],
                limit           = 50,
                output_fields   = ["chunk_id", "section_id"],
            )
            for hit in results[0]:
                if hit["distance"] >= ENTITY_FUZZY_THRESHOLD:
                    matched_chunks.add(hit["entity"]["chunk_id"])
                    matched_sections.add(hit["entity"]["section_id"])

        return matched_chunks, matched_sections

    def score_chunk(self, chunk_id: str, query_entities: List[str]) -> float:
        if not query_entities:
            return 0.0

        q_embs = self._embed_model.encode(query_entities)

        total = 0.0
        for emb in q_embs:
            results = self._client.search(
                collection_name = ENTITY_COLLECTION_NAME,
                data            = [emb.tolist()],
                limit           = 50,
                output_fields   = ["chunk_id", "weight"],
            )
            for hit in results[0]:
                if (hit["distance"] >= ENTITY_FUZZY_THRESHOLD
                        and hit["entity"]["chunk_id"] == chunk_id):
                    total += hit["entity"]["weight"]
        return total

    def destroy(self):
        try:
            self._client.drop_collection(ENTITY_COLLECTION_NAME)
        except Exception:
            pass
        try:
            self._client.close()
        except Exception:
            pass
        if os.path.exists(self._milvus_path):
            if os.path.isdir(self._milvus_path):
                shutil.rmtree(self._milvus_path)
            else:
                os.remove(self._milvus_path)


# ═══════════════════════════════════════════════════════════════════════════
# ──────────────────────────────  INDEX  ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class DirRAGIndex:

    def __init__(self, embed_model: EmbeddingModel):
        self.embed_model = embed_model
        self.sections: Dict[str, Section] = {}
        self.chunks:   Dict[str, Chunk]   = {}

        short_id = uuid.uuid4().hex[:8]
        milvus_path = f"{ENTITY_MILVUS_DIR}_{short_id}.db"
        self.entity_index = EntityInvertedIndex(embed_model, milvus_path)

        self._section_ids_ordered: List[str] = []
        self._chunk_ids_ordered:   List[str] = []

        self._section_bm25: Optional[BM25Index] = None
        self._chunk_bm25:   Optional[BM25Index] = None

    # ── offline build ──────────────────────────────────────────────────────

    def add_passage(
        self,
        title:   str,
        content: str,
        depth:   int = 1,
        path:    Optional[List[str]] = None,
        llm:     LocalLLMClient = None,
    ):
        """
        Add a single section (directory node) to the index.

        Parameters
        ----------
        title   : display name of the section (for BM25 + embedding blend)
        content : raw text belonging to this section
        depth   : hierarchy depth (1 = root-level)
        path    : full path as list of components, e.g. ["/Introduction", "Background"]
                  defaults to [title] when not provided
        llm     : LLM client used by entity extraction
        """
        sec_id  = str(uuid.uuid4())
        section = Section(
            section_id = sec_id,
            title      = title,
            path       = path if path is not None else [title],
            depth      = depth,
        )

        chunks = split_into_chunks(content, sec_id, llm=llm)
        for chunk in chunks:
            section.chunk_ids.append(chunk.chunk_id)
            self.chunks[chunk.chunk_id] = chunk
            self.entity_index.add_chunk(chunk, llm)
            self._chunk_ids_ordered.append(chunk.chunk_id)

        all_ents: List[str] = []
        for cid in section.chunk_ids:
            all_ents.extend(self.chunks[cid].entities)
        section.entities = list(dict.fromkeys(all_ents))

        self.sections[sec_id] = section
        self._section_ids_ordered.append(sec_id)

        # ── Wire parent → child relationship ─────────────────────────────
        # The parent of a node with path ["A", "B", "C"] has path ["A", "B"].
        # Scan already-registered sections for a matching parent path and
        # append this section's id to its children_ids list.
        current_path = section.path   # e.g. ["Introduction", "Background"]
        if len(current_path) > 1:
            parent_path = current_path[:-1]   # e.g. ["Introduction"]
            for existing_sid in self._section_ids_ordered[:-1]:  # exclude self
                existing_sec = self.sections[existing_sid]
                if existing_sec.path == parent_path:
                    existing_sec.children_ids.append(sec_id)
                    section.parent_id = existing_sid
                    break

    def build(self):
        """Compute all embeddings and BM25 indexes. Call once after all passages are added."""
        # ── section embeddings ──────────────────────────────────────────
        title_texts   = [self.sections[s].title for s in self._section_ids_ordered]
        summary_texts = []
        for sid in self._section_ids_ordered:
            sec = self.sections[sid]
            if sec.chunk_ids:
                raw = self.chunks[sec.chunk_ids[0]].text[:SUMMARY_CHARS]
            else:
                raw = sec.title
            summary_texts.append(raw)

        title_embs   = self.embed_model.encode(title_texts)
        summary_embs = self.embed_model.encode(summary_texts)
        section_embs = TITLE_WEIGHT * title_embs + SUMMARY_WEIGHT * summary_embs

        for i, sid in enumerate(self._section_ids_ordered):
            self.sections[sid].embedding = section_embs[i]

        # ── chunk embeddings ─────────────────────────────────────────────
        chunk_texts = [self.chunks[c].text for c in self._chunk_ids_ordered]
        if chunk_texts:
            chunk_embs = self.embed_model.encode(chunk_texts)
            for i, cid in enumerate(self._chunk_ids_ordered):
                self.chunks[cid].embedding = chunk_embs[i]

        # ── BM25 indexes ─────────────────────────────────────────────────
        self._section_bm25 = BM25Index(title_texts)
        self._chunk_bm25   = BM25Index(chunk_texts if chunk_texts else [""])

        # ── flush entity vectors into Milvus ─────────────────────────────
        self.entity_index.flush()

    def destroy(self):
        self.entity_index.destroy()

    # ── helpers for retrieval ──────────────────────────────────────────────

    def _section_idx(self, sid: str) -> int:
        return self._section_ids_ordered.index(sid)

    def _chunk_idx(self, cid: str) -> int:
        return self._chunk_ids_ordered.index(cid)

    def _doc_total_chunks(self, section_id: str) -> int:
        return len(self.sections[section_id].chunk_ids)


# ═══════════════════════════════════════════════════════════════════════════
# ───────────────────────────  RETRIEVER  ───────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class DirRAGRetriever:

    def __init__(self, index: DirRAGIndex):
        self.index = index
        self.max_depth = max(
            (s.depth for s in index.sections.values()), default=1
        )

    def retrieve_sections(
        self,
        q_emb:            np.ndarray,
        q_keywords:       List[str],
        q_entities:       List[str],
        used_section_ids: Set[str],
        top_k:            int = SECTION_TOP_K,
    ) -> List[Tuple[Section, float]]:
        idx   = self.index
        sids  = idx._section_ids_ordered

        sem_scores: Dict[str, float] = {}
        for sid in sids:
            sec = idx.sections[sid]
            if sec.embedding is not None:
                sem_scores[sid] = float(np.dot(q_emb, sec.embedding))

        bm25_raw = idx._section_bm25.top_k(q_keywords, k=20)
        bm25_max = bm25_raw[0][1] if bm25_raw and bm25_raw[0][1] > 0 else 1.0
        bm25_scores: Dict[str, float] = {}
        for doc_i, score in bm25_raw:
            sid = sids[doc_i]
            bm25_scores[sid] = score / bm25_max

        _, ent_section_ids = idx.entity_index.lookup_entities(q_entities)

        candidates = set(sids)

        scored: List[Tuple[str, float]] = []
        for sid in candidates:
            sec    = idx.sections[sid]
            s_sem  = sem_scores.get(sid, 0.0)
            s_bm25 = bm25_scores.get(sid, 0.0)

            if q_entities:
                s_ent = 1.0 if sid in ent_section_ids else 0.0
            else:
                s_ent = 0.0

            s_depth = min(sec.depth / max(self.max_depth, 1), 1.0)

            total = (
                SEC_ALPHA * s_sem
                + SEC_BETA  * s_bm25
                + SEC_GAMMA * s_ent
                + SEC_DELTA * s_depth
            )

            if sid in used_section_ids:
                total *= USED_SECTION_PENALTY

            scored.append((sid, total))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [(idx.sections[sid], scored_val)
                for sid, scored_val in scored[:top_k]]

    def retrieve_chunks(
        self,
        q_emb:        np.ndarray,
        q_keywords:   List[str],
        q_entities:   List[str],
        top_sections: List[Section],
        n_anchors:    int = CHUNK_ANCHOR_N,
    ) -> List[Chunk]:
        idx = self.index

        candidate_cids: Set[str] = set()
        for sec in top_sections:
            for cid in sec.chunk_ids:
                candidate_cids.add(cid)
            for child_sid in sec.children_ids:
                if child_sid in idx.sections:
                    for cid in idx.sections[child_sid].chunk_ids:
                        candidate_cids.add(cid)

        if not candidate_cids:
            return []

        cid_list   = list(candidate_cids)
        pool_texts = [idx.chunks[c].text for c in cid_list]
        pool_bm25  = BM25Index(pool_texts)

        bm25_raw = pool_bm25.top_k(q_keywords, k=min(20, len(cid_list)))
        bm25_max = bm25_raw[0][1] if bm25_raw and bm25_raw[0][1] > 0 else 1.0
        bm25_map: Dict[str, float] = {
            cid_list[i]: s / bm25_max for i, s in bm25_raw
        }

        ent_chunk_ids, _ = idx.entity_index.lookup_entities(q_entities)
        ent_chunk_ids &= candidate_cids

        scored: List[Tuple[str, float]] = []
        for cid in candidate_cids:
            chunk = idx.chunks[cid]
            sec   = idx.sections[chunk.section_id]
            doc_total = idx._doc_total_chunks(chunk.section_id)

            s_sem = float(np.dot(q_emb, chunk.embedding)) \
                    if chunk.embedding is not None else 0.0
            s_bm25 = bm25_map.get(cid, 0.0)

            if q_entities:
                ent_hits = sum(
                    1 for qe in q_entities
                    if any(qe.lower() in ce.lower() for ce in chunk.entities)
                )
                s_ent = ent_hits / len(q_entities)
            else:
                s_ent = 0.0

            s_pos = 1.0 - chunk.index_in_doc / max(doc_total, 1)

            total = (
                CHK_ALPHA * s_sem
                + CHK_BETA  * s_bm25
                + CHK_GAMMA * s_ent
                + CHK_DELTA * s_pos
            )
            scored.append((cid, total))

        scored.sort(key=lambda x: x[1], reverse=True)
        anchor_cids = [cid for cid, _ in scored[:n_anchors]]

        expanded: List[str] = []
        seen: Set[str] = set()

        for anchor_cid in anchor_cids:
            anchor = idx.chunks[anchor_cid]
            sec    = idx.sections[anchor.section_id]
            sec_cids = sec.chunk_ids
            try:
                pos = sec_cids.index(anchor_cid)
            except ValueError:
                pos = 0

            lo = max(0, pos - NEIGHBOR_WINDOW)
            hi = min(len(sec_cids) - 1, pos + NEIGHBOR_WINDOW)

            for i in range(lo, hi + 1):
                cid = sec_cids[i]
                if cid not in seen:
                    seen.add(cid)
                    expanded.append(cid)

        expanded.sort(
            key=lambda c: (
                idx.chunks[c].section_id,
                idx.chunks[c].index_in_doc,
            )
        )
        return [idx.chunks[c] for c in expanded]


# ═══════════════════════════════════════════════════════════════════════════
# ──────────────────────────────  PIPELINE  ─────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def build_context_string(chunks: List[Chunk], index: DirRAGIndex) -> str:
    blocks: List[str] = []
    for chunk in chunks:
        sec  = index.sections[chunk.section_id]
        path = " > ".join(sec.path)
        blocks.append(f"[Source: {path}]\n{chunk.text}")
    return "\n\n".join(blocks)


def print_retrieved_context(
    iteration: int,
    sections:  List[Tuple[Section, float]],
    chunks:    List[Chunk],
    index:     DirRAGIndex,
):
    print("\n" + "═" * 70)
    print(f"  RETRIEVAL — iteration {iteration}")
    print("═" * 70)
    print(f"  Top-{len(sections)} sections:")
    for sec, score in sections:
        print(f"    [{score:.3f}] {' > '.join(sec.path)}")
    print(f"\n  Retrieved {len(chunks)} chunk(s) after expansion:")
    for i, chunk in enumerate(chunks, 1):
        sec     = index.sections[chunk.section_id]
        path    = " > ".join(sec.path)
        preview = chunk.text[:120].replace('\n', ' ')
        print(f"    {i}. [{path}] {preview}…")
    print("═" * 70 + "\n")


def run_dirrag_query(
    question:    str,
    content:     List[Dict],   # [{"path": str, "text": str, "depth": int}, ...]
    embed_model: EmbeddingModel,
    llm:         LocalLLMClient,
) -> str:
    """
    Full DirRAG pipeline for a single Qasper query.

    Each content element becomes one Section in the index.
    `path` is used as the section title and path list;
    `depth` is preserved for the depth-reward scoring signal.
    """

    # ── Build per-query index ─────────────────────────────────────────────
    index = DirRAGIndex(embed_model)
    for node in content:
        node_path  = node["path"]          # e.g. "/Introduction/Background"
        node_text  = node["text"]
        node_depth = node.get("depth", 1)

        # Split the path string into components for display and BM25
        # e.g. "/Introduction/Background" → ["Introduction", "Background"]
        path_components = [p for p in node_path.split("/") if p]
        title = path_components[-1] if path_components else node_path

        index.add_passage(
            title   = title,
            content = node_text,
            depth   = node_depth,
            path    = path_components if path_components else [node_path],
            llm     = llm,
        )
    index.build()

    retriever = DirRAGRetriever(index)

    # ── Query parsing ─────────────────────────────────────────────────────
    q_emb      = embed_model.encode([question])[0]
    q_entities = extract_entities(question, llm)
    q_keywords = extract_keywords(question)

    # ── Iterative retrieval ───────────────────────────────────────────────
    all_chunks:       List[Chunk] = []
    used_section_ids: Set[str]    = set()
    accumulated_cids: Set[str]    = set()

    extra_keywords = list(q_keywords)
    extra_entities = list(q_entities)

    try:
        for iteration in range(1, MAX_ITER + 1):

            # Stage 1
            top_sections = retriever.retrieve_sections(
                q_emb            = q_emb,
                q_keywords       = extra_keywords,
                q_entities       = extra_entities,
                used_section_ids = used_section_ids,
            )
            for sec, _ in top_sections:
                used_section_ids.add(sec.section_id)

            # Stage 2
            new_chunks = retriever.retrieve_chunks(
                q_emb        = q_emb,
                q_keywords   = extra_keywords,
                q_entities   = extra_entities,
                top_sections = [s for s, _ in top_sections],
            )

            truly_new = [c for c in new_chunks if c.chunk_id not in accumulated_cids]
            for c in truly_new:
                accumulated_cids.add(c.chunk_id)
                all_chunks.append(c)

            print_retrieved_context(iteration, top_sections, new_chunks, index)

            context_str  = build_context_string(all_chunks, index)
            judge_result = llm_judge(llm, question, context_str)

            if judge_result.get("is_sufficient"):
                print(f"  ✅  LLM judged context sufficient at iteration {iteration}.")
                return judge_result.get("answer", "")

            missing = judge_result.get("missing_keywords", [])
            if missing:
                print(f"  ⚙️   Missing keywords for next iter: {missing}")
                extra_keywords = list(dict.fromkeys(
                    extra_keywords + [kw.lower() for kw in missing]
                ))
                extra_entities = list(dict.fromkeys(
                    extra_entities + [e for e in missing if e[0].isupper()]
                ))

        # ── Fallback: force answer with all accumulated context ───────────
        print(f"  ⚠️   max_iter={MAX_ITER} reached. Forcing final answer.")
        context_str  = build_context_string(all_chunks, index)
        final_prompt = (
            f"You are a precise reading-comprehension assistant.\n"
            f"Answer the question using ONLY the provided context. "
            f"Do NOT use any knowledge not present in the context.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Instructions:\n"
            f"1. Key context: Quote the 1-3 sentences from the context that are "
            f"most directly relevant to the question.\n"
            f"2. Reasoning: Explain step by step how those sentences lead to the "
            f"answer. Be explicit about every inference you make.\n"
            f"3. Confidence: Rate your confidence as High / Medium / Low and "
            f"briefly state why (e.g. direct evidence vs. multi-step inference).\n"
            f"4. Final answer: State the answer in as few words as possible "
            f"(a name, date, yes/no, or short phrase).\n\n"
            f"Output format (keep labels exactly as shown):\n"
            f"Key context: <quote(s) from context>\n"
            f"Reasoning: <step-by-step explanation>\n"
            f"Confidence: <High|Medium|Low> — <reason>\n"
            f"Final answer: <concise answer>\n"
        )
        return llm.chat(final_prompt).strip()

    finally:
        index.destroy()


# ═══════════════════════════════════════════════════════════════════════════
# ────────────────────────────  MAIN LOOP  ──────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  DirRAG v3.0  —  Qasper Evaluation")
    print("=" * 70)

    embed_model = EmbeddingModel()
    llm         = LocalLLMClient()

    dataset  = load_qasper_detailed(INPUT_PATH)
    done_ids = load_completed_sample_ids(OUTPUT_PATH)
    print(f"  Resuming: {len(done_ids)} samples already done.\n")

    total_em  = 0
    total_f1  = 0.0
    processed = 0

    for sample in tqdm(dataset, desc="DirRAG"):
        sid = sample["_id"]
        if sid in done_ids:
            continue

        question = sample["question"]
        gold     = sample["gold_answer"]
        content  = sample["content"]   # list of {path, text, depth}

        print(f"\n{'─'*70}")
        print(f"  Q: {question}")
        print(f"  Gold: {gold}")

        try:
            pred = run_dirrag_query(question, content, embed_model, llm)
        except Exception as e:
            print(f"  ❌ Error on sample {sid}: {e}")
            pred = ""

        print(f"  Pred: {pred}")

        em, f1 = calculate_qa_metrics(pred, gold)
        print(f"  EM={em}  F1={f1:.4f}")

        save_single_result(sid, question, gold, pred, em, f1, OUTPUT_PATH)

        total_em  += em
        total_f1  += f1
        processed += 1

    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    if processed > 0:
        print(f"  Samples processed : {processed}")
        print(f"  Average EM        : {total_em / processed:.4f}")
        print(f"  Average F1        : {total_f1 / processed:.4f}")
    else:
        print("  No new samples processed.")
    print("=" * 70)


if __name__ == "__main__":
    main()