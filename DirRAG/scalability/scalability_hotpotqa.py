import os
import re
import json
import math
import uuid
import time
import random
import argparse
import numpy as np
import spacy
import hnswlib

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm


# ── spaCy ─────────────────────────────────────────────────────────────────
try:
    NLP = spacy.load("en_core_web_sm", disable=["parser", "textcat"])
except OSError:
    raise RuntimeError("请运行：python -m spacy download en_core_web_sm")

NER_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "EVENT",
    "WORK_OF_ART", "LAW", "FAC", "PRODUCT", "NORP",
}
STOP_WORDS = {
    "a","an","the","is","are","was","were","and","or","but",
    "in","on","at","to","for","of","with","by","from","that",
    "this","it","its","be","as","do","did","has","have","had",
}


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
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
    depth:        int                  = 1
    embedding:    Optional[np.ndarray] = None
    entities:     List[str]            = field(default_factory=list)
    children_ids: List[str]            = field(default_factory=list)
    chunk_ids:    List[str]            = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# NER / TOKENIZE
# ═══════════════════════════════════════════════════════════════════════════

def extract_entities_ner(text: str) -> List[str]:
    if not text or len(text.strip()) < 5:
        return []
    doc = NLP(text[:900_000])
    return list(dict.fromkeys(
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in NER_TYPES and len(ent.text.strip()) > 1
    ))


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def extract_keywords(text: str) -> List[str]:
    return list(dict.fromkeys(tokenize(text)))


# ═══════════════════════════════════════════════════════════════════════════
# CHUNK SPLITTING
# ═══════════════════════════════════════════════════════════════════════════

CHUNK_MAX_TOKENS = 500


def split_into_chunks(text: str, section_id: str, doc_offset: int) -> List[Chunk]:
    """切分文本为 chunks，但不在此处做 NER（延迟到 build() 阶段统一计时）"""
    sentences  = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: List[Chunk] = []
    buffer: List[str]   = []
    buf_tokens = 0

    def flush(buf: List[str], idx: int) -> Chunk:
        txt = " ".join(buf)
        return Chunk(
            chunk_id     = str(uuid.uuid4()),
            section_id   = section_id,
            text         = txt,
            # entities 留空，build() 阶段统一填充
            index_in_doc = idx,
        )

    for sent in sentences:
        n_tok = len(sent.split())
        if buf_tokens + n_tok > CHUNK_MAX_TOKENS and buffer:
            chunks.append(flush(buffer, doc_offset + len(chunks)))
            buffer, buf_tokens = [], 0
        buffer.append(sent)
        buf_tokens += n_tok

    if buffer:
        chunks.append(flush(buffer, doc_offset + len(chunks)))
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# BM25
# ═══════════════════════════════════════════════════════════════════════════

class BM25Index:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.n_docs    = len(corpus)
        self.k1, self.b = k1, b
        self.tokenized = [tokenize(doc) for doc in corpus]
        self.avgdl     = sum(len(t) for t in self.tokenized) / max(self.n_docs, 1)
        df: Dict[str, int] = defaultdict(int)
        for toks in self.tokenized:
            for tok in set(toks):
                df[tok] += 1
        self.idf: Dict[str, float] = {
            tok: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for tok, freq in df.items()
        }

    def scores(self, query_tokens: List[str]) -> np.ndarray:
        result = np.zeros(self.n_docs)
        for qt in query_tokens:
            if qt not in self.idf:
                continue
            idf = self.idf[qt]
            for i, toks in enumerate(self.tokenized):
                tf = toks.count(qt)
                if tf == 0:
                    continue
                dl = len(toks)
                result[i] += idf * (
                    tf * (self.k1 + 1)
                    / (tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1)))
                )
        return result


# ═══════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

TITLE_WEIGHT   = 0.4
SUMMARY_WEIGHT = 0.6
SUMMARY_CHARS  = 300
BM25_K1        = 1.5
BM25_B         = 0.75

SEC_ALPHA   = 0.5    # semantic
SEC_BETA    = 0.3    # bm25
SEC_GAMMA   = 0.15   # entity
SEC_DELTA   = 0.05   # depth

CHK_ALPHA   = 0.5
CHK_BETA    = 0.3
CHK_GAMMA   = 0.15
CHK_DELTA   = 0.05

SECTION_TOP_K   = 5
CHUNK_ANCHOR_N  = 10
NEIGHBOR_WINDOW = 1
ENTITY_MATCH_THRESHOLD = 0.85

HNSW_EF_CONSTRUCTION = 200
HNSW_M = 16
HNSW_CANDIDATES = 50


# ═══════════════════════════════════════════════════════════════════════════
# INDEX
# ═══════════════════════════════════════════════════════════════════════════

class DirRAGIndex:

    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.sections: Dict[str, Section] = {}
        self.chunks:   Dict[str, Chunk]   = {}
        self._section_ids_ordered: List[str] = []
        self._chunk_ids_ordered:   List[str] = []
        self.total_passages = 0
        self.total_chunks   = 0

        self._section_emb_matrix: Optional[np.ndarray] = None
        self._chunk_emb_matrix:   Optional[np.ndarray] = None
        self._section_bm25: Optional[BM25Index] = None
        self._chunk_bm25:   Optional[BM25Index] = None
        self._hnsw_section: Optional[hnswlib.Index] = None

    def add_passage(self, title: str, content: str):
        if not title or not content:
            return
        sec_id  = str(uuid.uuid4())
        section = Section(section_id=sec_id, title=title, depth=1)
        # NER 不在此处调用，延迟到 build() 统一计时
        chunks  = split_into_chunks(content, sec_id, len(self._chunk_ids_ordered))
        for chunk in chunks:
            section.chunk_ids.append(chunk.chunk_id)
            self.chunks[chunk.chunk_id] = chunk
            self._chunk_ids_ordered.append(chunk.chunk_id)
        self.sections[sec_id] = section
        self._section_ids_ordered.append(sec_id)
        self.total_passages += 1
        self.total_chunks   += len(chunks)

    def build(self) -> Dict[str, float]:
        sids = self._section_ids_ordered
        cids = self._chunk_ids_ordered

        # ── Embedding ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        title_texts   = [self.sections[s].title for s in sids]
        summary_texts = [
            self.chunks[self.sections[s].chunk_ids[0]].text[:SUMMARY_CHARS]
            if self.sections[s].chunk_ids else self.sections[s].title
            for s in sids
        ]
        title_embs   = self.embed_model.encode(title_texts)
        summary_embs = self.embed_model.encode(summary_texts)
        sec_embs     = TITLE_WEIGHT * title_embs + SUMMARY_WEIGHT * summary_embs
        self._section_emb_matrix = sec_embs
        for i, sid in enumerate(sids):
            self.sections[sid].embedding = sec_embs[i]

        chunk_texts = [self.chunks[c].text for c in cids]
        if chunk_texts:
            chk_embs = self.embed_model.encode(chunk_texts)
            self._chunk_emb_matrix = chk_embs
            for i, cid in enumerate(cids):
                self.chunks[cid].embedding = chk_embs[i]
        t1 = time.perf_counter()

        # ── BM25 ─────────────────────────────────────────────────────────
        t2 = time.perf_counter()
        self._section_bm25 = BM25Index(title_texts,  BM25_K1, BM25_B)
        self._chunk_bm25   = BM25Index(chunk_texts if chunk_texts else [""], BM25_K1, BM25_B)
        t3 = time.perf_counter()

        # ── Entity NER（从此处统一提取，计时准确）────────────────────────
        t4 = time.perf_counter()
        for cid in cids:
            chunk = self.chunks[cid]
            chunk.entities = extract_entities_ner(chunk.text)
        for sid in sids:
            section = self.sections[sid]
            all_ents: List[str] = []
            for cid in section.chunk_ids:
                all_ents.extend(self.chunks[cid].entities)
            section.entities = list(dict.fromkeys(all_ents))
            # 同时补充 title NER
            title_ents = extract_entities_ner(section.title)
            section.entities = list(dict.fromkeys(title_ents + section.entities))
        t5 = time.perf_counter()

        # ── HNSW ─────────────────────────────────────────────────────────
        t6 = time.perf_counter()
        if len(sids) > 0:
            dim = sec_embs.shape[1]
            hnsw = hnswlib.Index(space='ip', dim=dim)
            hnsw.init_index(max_elements=len(sids), ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M)
            hnsw.add_items(sec_embs)
            self._hnsw_section = hnsw
        t7 = time.perf_counter()

        return {
            "build_embedding_time_s": round(t1 - t0, 4),
            "build_bm25_time_s":      round(t3 - t2, 4),
            "build_entity_time_s":    round(t5 - t4, 4),
            "build_hnsw_time_s":      round(t7 - t6, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVER 【已修复 hnswlib API】
# ═══════════════════════════════════════════════════════════════════════════

class DirRAGRetriever:

    def __init__(self, index: DirRAGIndex):
        self.index = index

    def retrieve_sections(
        self,
        q_emb:      np.ndarray,
        q_keywords: List[str],
        q_entities: List[str],
    ) -> List[Tuple[Section, float]]:
        idx  = self.index
        sids = idx._section_ids_ordered
        n    = len(sids)
        if n == 0:
            return []

        # ===================== 【修复版 HNSW】=====================
        top_k_hnsw = min(HNSW_CANDIDATES, n)
        q_emb_2d = q_emb.reshape(1, -1)
        labels, distances = idx._hnsw_section.knn_query(q_emb_2d, k=top_k_hnsw)
        # =========================================================

        candidate_indices = labels[0].tolist()
        sem_scores = {i: 1 - distances[0][idx] for idx, i in enumerate(candidate_indices)}

        bm25_raw = idx._section_bm25.scores(q_keywords)
        bm25_max = bm25_raw.max()
        bm25_scores = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw

        ent_scores = {}
        if q_entities:
            for i in candidate_indices:
                sec_ents = {e.lower() for e in idx.sections[sids[i]].entities}
                matches = sum(1 for qe in q_entities if any(qe.lower() in se for se in sec_ents))
                ent_scores[i] = matches / len(q_entities)
        else:
            ent_scores = {i: 0.0 for i in candidate_indices}

        scored = []
        for i in candidate_indices:
            total = SEC_ALPHA * sem_scores[i] + SEC_BETA * bm25_scores[i] + SEC_GAMMA * ent_scores[i] + SEC_DELTA * 1.0
            scored.append((i, total))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(idx.sections[sids[i]], float(score)) for i, score in scored[:SECTION_TOP_K]]

    def retrieve_chunks_from_sections(
        self,
        q_emb:        np.ndarray,
        q_keywords:   List[str],
        q_entities:   List[str],
        top_sections: List[Section],
    ) -> List[Chunk]:
        idx = self.index
        if not top_sections:
            return []

        candidate_cids: List[str] = []
        for sec in top_sections:
            candidate_cids.extend(sec.chunk_ids)
            for child_sid in sec.children_ids:
                if child_sid in idx.sections:
                    candidate_cids.extend(idx.sections[child_sid].chunk_ids)

        if not candidate_cids:
            return []

        cand_embs  = np.stack([idx.chunks[c].embedding for c in candidate_cids])
        sem_scores = cand_embs @ q_emb

        cand_texts = [idx.chunks[c].text for c in candidate_cids]
        local_bm25 = BM25Index(cand_texts, BM25_K1, BM25_B)
        bm25_raw   = local_bm25.scores(q_keywords)
        bm25_max   = bm25_raw.max()
        bm25_scores = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw

        ent_scores = np.zeros(len(candidate_cids))
        if q_entities:
            for i, cid in enumerate(candidate_cids):
                chunk_ents = {e.lower() for e in idx.chunks[cid].entities}
                matches    = sum(1 for qe in q_entities if any(qe.lower() in ce for ce in chunk_ents))
                ent_scores[i] = matches / len(q_entities)

        pos_scores = np.array([
            1.0 - idx.chunks[c].index_in_doc / max(len(idx.sections[idx.chunks[c].section_id].chunk_ids), 1)
            for c in candidate_cids
        ])

        total = CHK_ALPHA * sem_scores + CHK_BETA * bm25_scores + CHK_GAMMA * ent_scores + CHK_DELTA * pos_scores
        sorted_idx  = np.argsort(total)[::-1]
        anchor_cids = [candidate_cids[i] for i in sorted_idx[:CHUNK_ANCHOR_N]]

        expanded: List[str] = []
        seen: Set[str] = set()
        for anchor_cid in anchor_cids:
            anchor   = idx.chunks[anchor_cid]
            sec_cids = idx.sections[anchor.section_id].chunk_ids
            try:
                pos = sec_cids.index(anchor_cid)
            except ValueError:
                pos = 0
            lo = max(0, pos - NEIGHBOR_WINDOW)
            hi = min(len(sec_cids) - 1, pos + NEIGHBOR_WINDOW)
            for j in range(lo, hi + 1):
                cid = sec_cids[j]
                if cid not in seen:
                    seen.add(cid)
                    expanded.append(cid)

        expanded.sort(key=lambda c: (idx.chunks[c].section_id, idx.chunks[c].index_in_doc))
        return [idx.chunks[c] for c in expanded]


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def load_hotpotqa(input_path: str) -> List[Dict]:
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            samples.append({
                "_id":      raw["_id"],
                "question": raw["input"],
                "passages": raw["passages"],
            })
    return samples


def build_passage_pool(all_samples: List[Dict], focal_idx: int, scale: int, rng: random.Random) -> List[Dict]:
    focal_passages = list(all_samples[focal_idx]["passages"])
    if scale == 1:
        return focal_passages
    other_indices   = [i for i in range(len(all_samples)) if i != focal_idx]
    n_extra         = min(scale - 1, len(other_indices))
    sampled_indices = rng.sample(other_indices, n_extra)
    extra: List[Dict] = []
    for idx in sampled_indices:
        extra.extend(all_samples[idx]["passages"])
    return focal_passages + extra


def run_scalability_single(question: str, passages: List[Dict], embed_model) -> Dict:
    t0 = time.perf_counter()
    index = DirRAGIndex(embed_model)
    for passage in passages:
        index.add_passage(passage.get("title", "").strip(), passage.get("content", "").strip())
    build_breakdown = index.build()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    q_emb      = embed_model.encode([question])[0]
    q_entities = extract_entities_ner(question)
    q_keywords = extract_keywords(question)
    t3 = time.perf_counter()

    retriever = DirRAGRetriever(index)
    t4 = time.perf_counter()
    top_sections = retriever.retrieve_sections(q_emb, q_keywords, q_entities)
    t5 = time.perf_counter()

    t6 = time.perf_counter()
    chunks = retriever.retrieve_chunks_from_sections(q_emb, q_keywords, q_entities, [s for s, _ in top_sections])
    t7 = time.perf_counter()

    sec_t = t5 - t4
    chk_t = t7 - t6

    return {
        "total_passages":           index.total_passages,
        "total_chunks":             index.total_chunks,
        "total_sections":           len(index.sections),
        "q_entity_count":           len(q_entities),
        "q_keyword_count":          len(q_keywords),
        "top_sections_returned":    len(top_sections),
        "chunks_retrieved":         len(chunks),
        "index_build_time_s":       round(t1 - t0, 4),
        "build_embedding_time_s":   build_breakdown["build_embedding_time_s"],
        "build_bm25_time_s":        build_breakdown["build_bm25_time_s"],
        "build_entity_time_s":      build_breakdown["build_entity_time_s"],
        "build_hnsw_time_s":        build_breakdown["build_hnsw_time_s"],
        "query_encode_time_s":      round(t3 - t2, 4),
        "section_retrieval_time_s": round(sec_t, 4),
        "chunk_retrieval_time_s":   round(chk_t, 4),
        "total_retrieval_time_s":   round(sec_t + chk_t, 4),
        "end_to_end_time_s":        round((t1-t0) + (t3-t2) + sec_t + chk_t, 4),
    }


def run_scale_experiment(all_samples: List[Dict], scale: int, n_queries: int, embed_model, output_path: str, seed: int = 42):
    rng = random.Random(seed)
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)
    focal_indices = indices[:n_queries]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    done_ids: Set[str] = set()
    if os.path.exists(output_path):
        is_old = False
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "ok" and "build_hnsw_time_s" not in rec:
                        is_old = True
                        break
                except Exception: pass
        if is_old:
            print(f"  ⚠️  旧结果清空重跑：{output_path}")
            os.remove(output_path)
        else:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try: done_ids.add(json.loads(line)["sample_id"])
                        except Exception: pass

    print(f"\n  [Scale {scale}x]  queries={n_queries}  已完成={len(done_ids)}  输出→ {output_path}")
    timings = []

    for rank, focal_idx in enumerate(tqdm(focal_indices, desc=f"scale={scale}x"), start=1):
        sample    = all_samples[focal_idx]
        sample_id = sample["_id"]
        if sample_id in done_ids: continue

        passages = build_passage_pool(all_samples, focal_idx, scale, rng)
        pool_n_passages = len(passages)
        pool_n_tokens = sum(len((p.get("title","")+" "+p.get("content","")).split()) for p in passages)

        try:
            timing = run_scalability_single(question=sample["question"], passages=passages, embed_model=embed_model)
            status = "ok"
        except Exception as e:
            print(f"    ❌ Error [{sample_id}]: {e}")
            timing, status = {}, f"error: {e}"

        record = {"sample_id": sample_id, "scale": scale, "rank_in_run": rank, "question": sample["question"],
                  "focal_passages": len(sample["passages"]), "pool_n_passages": pool_n_passages,
                  "pool_n_tokens": pool_n_tokens, "status": status,**timing}
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if timing: timings.append({**timing, "pool_n_passages": pool_n_passages, "pool_n_tokens": pool_n_tokens})

    if timings:
        def avg(k): vals = [t[k] for t in timings if k in t]; return round(sum(vals)/len(vals),4) if vals else None
        print(f"\n  ── Scale {scale}x 汇总（{len(timings)} 条）──")
        print(f"    avg pool_n_passages       : {avg('pool_n_passages')}")
        print(f"    avg pool_n_tokens         : {avg('pool_n_tokens')}")
        print(f"    avg total_chunks          : {avg('total_chunks')}")
        print(f"    avg index_build_time_s    : {avg('index_build_time_s')}")
        print(f"      ├ embedding             : {avg('build_embedding_time_s')}")
        print(f"      ├ bm25                  : {avg('build_bm25_time_s')}")
        print(f"      ├ entity                : {avg('build_entity_time_s')}")
        print(f"      └ hnsw                  : {avg('build_hnsw_time_s')}")
        print(f"    avg query_encode_time_s   : {avg('query_encode_time_s')}")
        print(f"    avg section_retrieval_s   : {avg('section_retrieval_time_s')}")
        print(f"    avg chunk_retrieval_s     : {avg('chunk_retrieval_time_s')}")
        print(f"    avg total_retrieval_s     : {avg('total_retrieval_time_s')}")
        print(f"    avg end_to_end_s          : {avg('end_to_end_time_s')}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_INPUT  = "/root/dir-sem-rag/longbench/data_process/hotpotqa_clean.jsonl"
DEFAULT_OUTDIR = "./results_hotpotqa/scalability_v3_hnsw_v2"

SCALE_CONFIG = {1:50,
# 5:50,
# 10:50,
# 20:50,
# 30:50,
# 40:50,
50:50,
# 60:50,
# 70:50,
75: 50,
# 80:50,
# 90:50,
100:30,
150:30,
200:20}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument("--scales", type=int, nargs="+", default=list(SCALE_CONFIG.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import sys
    sys.path.insert(0, "/root/dir-sem-rag/gte-Qwen2-1.5B-instruct")
    from embedding import EmbeddingModel

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"]="false"

    embed_model = EmbeddingModel()
    all_samples = load_hotpotqa(args.input_path)
    print(f"[Data] {len(all_samples)} 条  ←  {args.input_path}")

    print("\n" + "═"*60)
    print("  DirRAG 可扩展性实验 v3  [hnswlib 修复版]")
    print(f"  scales : {sorted(args.scales)}")
    print(f"  seed   : {args.seed}")
    print(f"  output : {args.output_dir}")
    print("═"*60)

    for scale in sorted(args.scales):
        if scale not in SCALE_CONFIG: continue
        run_scale_experiment(all_samples=all_samples, scale=scale, n_queries=SCALE_CONFIG[scale],
                             embed_model=embed_model, output_path=os.path.join(args.output_dir, f"scale_{scale}x.jsonl"), seed=args.seed)

    print("\n" + "═"*60)
    print("  完成。")
    print("═"*60)

if __name__ == "__main__":
    main()
