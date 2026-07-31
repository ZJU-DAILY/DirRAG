"""
dirrag_ablation_musique.py
===========================
DirRAG v3.0 消融实验主入口 —— musique 版本

数据格式（每条 jsonl）：
    {
      "_id":      str,
      "input":    str,           # 问题
      "answers":  List[str],     # 金标答案列表，取第一个作为 gold
      "passages": [
          {"title": str, "content": str},
          ...
      ],
      "dataset":  "musique",
      "language": "en"
    }

目录结构约定（与 Qasper 版的差异）：
    每个 passage 的 title 单独作为一个「一级目录节点」（depth=1），
    其 content 被切分为 chunk 挂载在该目录下。
    不存在多级层次，所有目录节点深度相同。

    示例：
        Section: path=["Case 1"], depth=1  → content chunks
        Section: path=["Case 2"], depth=1  → content chunks
        Section: path=["Case 3"], depth=1  → content chunks

用法示例
--------
# 运行单个变体
python dirrag_ablation_musique.py --variant full_dirrag
python dirrag_ablation_musique.py --variant wo_section_routing
python dirrag_ablation_musique.py --variant wo_iteration
python dirrag_ablation_musique.py --variant title_only
python dirrag_ablation_musique.py --variant summary_only
python dirrag_ablation_musique.py --variant wo_depth_reward
python dirrag_ablation_musique.py --variant semantic_only

# 列出所有可用变体
python dirrag_ablation_musique.py --list

# 从外部 JSON 文件加载配置（自定义消融）
python dirrag_ablation_musique.py --config_path ./my_custom_config.json

# 覆盖子集大小（快速验证用，跑 50 条）
python dirrag_ablation_musique.py --variant full_dirrag --sample_size 50
"""

import os
import re
import json
import math
import uuid
import random
import shutil
import hashlib
import argparse
import numpy as np

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm

from ablation_config import DirRAGConfig, get_config, list_variants, ABLATION_VARIANTS

# ── 项目级单例 ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, "/root/dir-sem-rag/gte-Qwen2-1.5B-instruct")

from embedding import EmbeddingModel, EMBEDDING_DIM
from llm import LocalLLMClient, llm_judge
from data_utils import (
    load_completed_sample_ids,
    save_single_result,
    calculate_qa_metrics,
)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# ═══════════════════════════════════════════════════════════════════════════
# ──────────────────────────  DATA STRUCTURES  ──────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

STOP_WORDS = {
    "a","an","the","is","are","was","were","and","or","but",
    "in","on","at","to","for","of","with","by","from","that",
    "this","it","its","be","as","do","did","has","have","had",
}

ENTITY_COLLECTION_NAME = "entities"
ENTITY_MILVUS_BASE     = "/tmp/dirrag_entity_milvus"


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
# ──────────────────────────  ENTITY CACHE  ─────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class EntityCache:
    """
    本地持久化实体缓存。

    Cache key：md5(text.strip())
    存储格式：单个 JSON 文件，结构为 { "<md5>": ["entity1", "entity2", ...] }

    设计说明
    --------
    - key 基于文本内容而非 chunkId，因为 chunkId 是 uuid4() 动态生成的，
      同一段文本两次运行会产生不同 chunkId，导致缓存永远 miss。
    - 跨变体、跨运行均可复用，只要文本相同就能命中。
    - 写入策略：每次 set() 都立刻落盘（追加写模式），防止崩溃丢失缓存。
    """

    def __init__(self, cache_path: str = "./entity_musique_cache.json"):
        self._path  = cache_path
        self._store: Dict[str, List[str]] = {}
        self._hits  = 0
        self._total = 0
        self._load()

    # ── 持久化 ─────────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._store = json.load(f)
                print(f"[EntityCache] 加载缓存 {len(self._store)} 条  ← {self._path}")
            except Exception as e:
                print(f"[EntityCache] 缓存文件损坏，重新开始：{e}")
                self._store = {}

    def _save(self):
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        # 写临时文件再原子替换，防止写到一半崩溃导致 JSON 损坏
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False)
        os.replace(tmp, self._path)

    # ── 公共接口 ───────────────────────────────────────────────────────────

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[str]]:
        self._total += 1
        key = self._key(text)
        if key in self._store:
            self._hits += 1
            return self._store[key]
        return None

    def set(self, text: str, entities: List[str]):
        key = self._key(text)
        self._store[key] = entities
        self._save()   # 立刻落盘

    def stats(self) -> str:
        rate = self._hits / max(self._total, 1) * 100
        return (
            f"[EntityCache] 命中率 {self._hits}/{self._total} "
            f"({rate:.1f}%)  缓存条目 {len(self._store)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# ──────────────────────────  HELPERS  ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def extract_entities(
    text:  str,
    llm:   LocalLLMClient,
    cache: Optional[EntityCache] = None,
) -> List[str]:
    if not text or len(text.strip()) < 10:
        return []

    # ── 缓存命中：直接返回，跳过 LLM 调用 ────────────────────────────────
    if cache is not None:
        cached = cache.get(text)
        if cached is not None:
            return cached

    prompt = f"""
You are a precise entity extractor.
Extract ALL important named entities from the following text.
Entities include: people, places, organizations, cases, laws, events, titles, proper nouns.
Output ONLY a valid JSON list of strings. No extra words, no markdown.
Text: {text}
""".strip()
    try:
        raw      = llm.chat(prompt)
        cleaned  = re.sub(r"```(?:json)?|```", "", raw).strip()
        entities = json.loads(cleaned)
        if not isinstance(entities, list):
            entities = []
        entities = list(dict.fromkeys(str(e).strip() for e in entities if e))
    except Exception:
        entities = []

    # ── 写入缓存（空列表也存，避免对同一段文本反复调用 LLM）────────────
    if cache is not None:
        cache.set(text, entities)

    return entities


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def extract_keywords(text: str) -> List[str]:
    return list(dict.fromkeys(tokenize(text)))


def split_into_chunks(
    text: str,
    section_id: str,
    doc_offset: int,
    llm: LocalLLMClient,
    cfg: DirRAGConfig,
    cache: Optional[EntityCache] = None,
) -> List[Chunk]:
    CHUNK_MIN_TOKENS = 100
    CHUNK_MAX_TOKENS = 500

    sentences   = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: List[Chunk] = []
    buffer: List[str]   = []
    buf_tokens = 0

    def flush(buf: List[str], idx: int) -> Chunk:
        txt = " ".join(buf)
        return Chunk(
            chunk_id     = str(uuid.uuid4()),
            section_id   = section_id,
            text         = txt,
            entities     = extract_entities(txt, llm, cache),   # ← 传入 cache
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
# ──────────────────────────  BM25  ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class BM25Index:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.corpus    = corpus
        self.n_docs    = len(corpus)
        self.k1, self.b = k1, b
        self.tokenized = [tokenize(doc) for doc in corpus]
        self.avgdl     = sum(len(t) for t in self.tokenized) / max(self.n_docs, 1)
        self._build_idf()

    def _build_idf(self):
        df: Dict[str, int] = defaultdict(int)
        for toks in self.tokenized:
            for tok in set(toks):
                df[tok] += 1
        self.idf: Dict[str, float] = {
            tok: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for tok, freq in df.items()
        }

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
                tf * (self.k1 + 1)
                / (tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1)))
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
    def __init__(self, embed_model: EmbeddingModel, milvus_path: str, cfg: DirRAGConfig):
        from pymilvus import MilvusClient
        self._embed_model = embed_model
        self._milvus_path = milvus_path
        self._cfg         = cfg
        self._client      = MilvusClient(milvus_path)
        self._client.create_collection(
            collection_name = ENTITY_COLLECTION_NAME,
            dimension       = EMBEDDING_DIM,
            metric_type     = "COSINE",
        )
        self._pending: List[Dict] = []

    def add_chunk(self, chunk: Chunk, llm: LocalLLMClient, cache: Optional[EntityCache] = None):
        # 第一句话的实体用于计算 bonus，同样走缓存
        first_sent = chunk.text.split('.')[0]
        first_sent_entities = set(extract_entities(first_sent, llm, cache))
        tf_map: Dict[str, int] = defaultdict(int)
        for ent in chunk.entities:
            tf_map[ent] += 1
        for ent, tf in tf_map.items():
            bonus  = 0.3 if ent in first_sent_entities else 0.0
            self._pending.append({
                "entity_text": ent,
                "chunk_id":    chunk.chunk_id,
                "section_id":  chunk.section_id,
                "weight":      tf + bonus,
            })

    def flush(self):
        if not self._pending:
            return
        texts = [r["entity_text"] for r in self._pending]
        embs  = self._embed_model.encode(texts)
        rows  = [
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
        self._client.insert(collection_name=ENTITY_COLLECTION_NAME, data=rows)
        self._pending.clear()

    def lookup_entities(self, query_entities: List[str]) -> Tuple[Set[str], Set[str]]:
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
                if hit["distance"] >= self._cfg.entity_fuzzy_threshold:
                    matched_chunks.add(hit["entity"]["chunk_id"])
                    matched_sections.add(hit["entity"]["section_id"])
        return matched_chunks, matched_sections

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
            shutil.rmtree(self._milvus_path) if os.path.isdir(self._milvus_path) \
                else os.remove(self._milvus_path)


# ═══════════════════════════════════════════════════════════════════════════
# ──────────────────────────────  INDEX  ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class DirRAGIndex:
    def __init__(self, embed_model: EmbeddingModel, cfg: DirRAGConfig):
        self.embed_model = embed_model
        self.cfg         = cfg
        self.sections: Dict[str, Section] = {}
        self.chunks:   Dict[str, Chunk]   = {}

        milvus_path = f"{ENTITY_MILVUS_BASE}_{uuid.uuid4().hex[:8]}.db"
        self.entity_index = EntityInvertedIndex(embed_model, milvus_path, cfg)

        self._section_ids_ordered: List[str] = []
        self._chunk_ids_ordered:   List[str] = []
        self._section_bm25: Optional[BM25Index] = None
        self._chunk_bm25:   Optional[BM25Index] = None

    def add_passage(
        self,
        title:   str,
        content: str,
        depth:   int,
        path:    List[str],
        llm:     LocalLLMClient,
        cache:   Optional[EntityCache] = None,   # ← 新增
    ):
        sec_id  = str(uuid.uuid4())
        section = Section(
            section_id = sec_id,
            title      = title,
            path       = path,
            depth      = depth,
        )
        chunks = split_into_chunks(
            content, sec_id, len(self._chunk_ids_ordered), llm, self.cfg, cache
        )
        for chunk in chunks:
            section.chunk_ids.append(chunk.chunk_id)
            self.chunks[chunk.chunk_id] = chunk
            self.entity_index.add_chunk(chunk, llm, cache)   # ← 传入 cache
            self._chunk_ids_ordered.append(chunk.chunk_id)

        all_ents: List[str] = []
        for cid in section.chunk_ids:
            all_ents.extend(self.chunks[cid].entities)
        section.entities = list(dict.fromkeys(all_ents))

        self.sections[sec_id] = section
        self._section_ids_ordered.append(sec_id)

        # 建立父子关系
        if len(path) > 1:
            parent_path = path[:-1]
            for existing_sid in self._section_ids_ordered[:-1]:
                if self.sections[existing_sid].path == parent_path:
                    self.sections[existing_sid].children_ids.append(sec_id)
                    section.parent_id = existing_sid
                    break

    def build(self):
        cfg = self.cfg
        # ── Section 嵌入（由 title_weight / summary_weight 控制）──────────
        title_texts   = [self.sections[s].title for s in self._section_ids_ordered]
        summary_texts = []
        for sid in self._section_ids_ordered:
            sec = self.sections[sid]
            raw = self.chunks[sec.chunk_ids[0]].text[:cfg.summary_chars] \
                  if sec.chunk_ids else sec.title
            summary_texts.append(raw)

        title_embs   = self.embed_model.encode(title_texts)
        summary_embs = self.embed_model.encode(summary_texts)

        # ── 关键消融点：嵌入混合策略 ──────────────────────────────────────
        section_embs = cfg.title_weight * title_embs + cfg.summary_weight * summary_embs
        for i, sid in enumerate(self._section_ids_ordered):
            self.sections[sid].embedding = section_embs[i]

        chunk_texts = [self.chunks[c].text for c in self._chunk_ids_ordered]
        if chunk_texts:
            chunk_embs = self.embed_model.encode(chunk_texts)
            for i, cid in enumerate(self._chunk_ids_ordered):
                self.chunks[cid].embedding = chunk_embs[i]

        self._section_bm25 = BM25Index(title_texts, cfg.bm25_k1, cfg.bm25_b)
        self._chunk_bm25   = BM25Index(chunk_texts if chunk_texts else [""],
                                       cfg.bm25_k1, cfg.bm25_b)
        self.entity_index.flush()

    def destroy(self):
        self.entity_index.destroy()


# ═══════════════════════════════════════════════════════════════════════════
# ──────────────────────────────  RETRIEVER  ────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class DirRAGRetriever:
    def __init__(self, index: DirRAGIndex):
        self.index     = index
        self.cfg       = index.cfg
        self.max_depth = max(
            (s.depth for s in index.sections.values()), default=1
        )

    def retrieve_sections(
        self,
        q_emb:            np.ndarray,
        q_keywords:       List[str],
        q_entities:       List[str],
        used_section_ids: Set[str],
    ) -> List[Tuple[Section, float]]:
        idx  = self.index
        cfg  = self.cfg
        sids = idx._section_ids_ordered

        # Semantic
        sem_scores = {
            sid: float(np.dot(q_emb, idx.sections[sid].embedding))
            for sid in sids
            if idx.sections[sid].embedding is not None
        }

        # BM25
        bm25_raw = idx._section_bm25.top_k(q_keywords, k=20)
        bm25_max = bm25_raw[0][1] if bm25_raw and bm25_raw[0][1] > 0 else 1.0
        bm25_scores = {sids[i]: s / bm25_max for i, s in bm25_raw}

        # Entity
        _, ent_section_ids = idx.entity_index.lookup_entities(q_entities)

        scored: List[Tuple[str, float]] = []
        for sid in sids:
            sec    = idx.sections[sid]
            s_sem  = sem_scores.get(sid, 0.0)
            s_bm25 = bm25_scores.get(sid, 0.0)
            s_ent  = (1.0 if sid in ent_section_ids else 0.0) if q_entities else 0.0
            s_dep  = min(sec.depth / max(self.max_depth, 1), 1.0)

            # ── 关键消融点：Section 评分公式 ───────────────────────────────
            total = (
                cfg.sec_alpha * s_sem
                + cfg.sec_beta  * s_bm25
                + cfg.sec_gamma * s_ent
                + cfg.sec_delta * s_dep
            )
            if sid in used_section_ids:
                total *= cfg.used_section_penalty
            scored.append((sid, total))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            (idx.sections[sid], val)
            for sid, val in scored[:cfg.section_top_k]
        ]

    def retrieve_chunks_from_sections(
        self,
        q_emb:        np.ndarray,
        q_keywords:   List[str],
        q_entities:   List[str],
        top_sections: List[Section],
    ) -> List[Chunk]:
        idx = self.index
        cfg = self.cfg

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

        return self._score_and_expand(q_emb, q_keywords, q_entities, candidate_cids)

    def retrieve_chunks_global(
        self,
        q_emb:      np.ndarray,
        q_keywords: List[str],
        q_entities: List[str],
    ) -> List[Chunk]:
        """
        W/O Section Routing 变体：直接在所有 chunk 中检索。
        """
        candidate_cids = set(self.index._chunk_ids_ordered)
        return self._score_and_expand(q_emb, q_keywords, q_entities, candidate_cids)

    def _score_and_expand(
        self,
        q_emb:          np.ndarray,
        q_keywords:     List[str],
        q_entities:     List[str],
        candidate_cids: Set[str],
    ) -> List[Chunk]:
        idx = self.index
        cfg = self.cfg

        cid_list   = list(candidate_cids)
        pool_texts = [idx.chunks[c].text for c in cid_list]
        pool_bm25  = BM25Index(pool_texts, cfg.bm25_k1, cfg.bm25_b)

        bm25_raw = pool_bm25.top_k(q_keywords, k=min(20, len(cid_list)))
        bm25_max = bm25_raw[0][1] if bm25_raw and bm25_raw[0][1] > 0 else 1.0
        bm25_map = {cid_list[i]: s / bm25_max for i, s in bm25_raw}

        ent_chunk_ids, _ = idx.entity_index.lookup_entities(q_entities)
        ent_chunk_ids &= candidate_cids

        scored: List[Tuple[str, float]] = []
        for cid in candidate_cids:
            chunk     = idx.chunks[cid]
            doc_total = len(idx.sections[chunk.section_id].chunk_ids)

            s_sem  = float(np.dot(q_emb, chunk.embedding)) \
                     if chunk.embedding is not None else 0.0
            s_bm25 = bm25_map.get(cid, 0.0)
            s_ent  = (
                sum(1 for qe in q_entities
                    if any(qe.lower() in ce.lower() for ce in chunk.entities))
                / len(q_entities)
            ) if q_entities else 0.0
            s_pos  = 1.0 - chunk.index_in_doc / max(doc_total, 1)

            # ── 关键消融点：Chunk 评分公式 ─────────────────────────────────
            total = (
                cfg.chk_alpha * s_sem
                + cfg.chk_beta  * s_bm25
                + cfg.chk_gamma * s_ent
                + cfg.chk_delta * s_pos
            )
            scored.append((cid, total))

        scored.sort(key=lambda x: x[1], reverse=True)
        anchor_cids = [cid for cid, _ in scored[:cfg.chunk_anchor_n]]

        expanded: List[str] = []
        seen: Set[str] = set()
        for anchor_cid in anchor_cids:
            anchor   = idx.chunks[anchor_cid]
            sec_cids = idx.sections[anchor.section_id].chunk_ids
            try:
                pos = sec_cids.index(anchor_cid)
            except ValueError:
                pos = 0
            lo = max(0, pos - cfg.neighbor_window)
            hi = min(len(sec_cids) - 1, pos + cfg.neighbor_window)
            for i in range(lo, hi + 1):
                cid = sec_cids[i]
                if cid not in seen:
                    seen.add(cid)
                    expanded.append(cid)

        expanded.sort(
            key=lambda c: (idx.chunks[c].section_id, idx.chunks[c].index_in_doc)
        )
        return [idx.chunks[c] for c in expanded]


# ═══════════════════════════════════════════════════════════════════════════
# ──────────────────────────────  PIPELINE  ─────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def build_context_string(chunks: List[Chunk], index: DirRAGIndex) -> str:
    blocks = []
    for chunk in chunks:
        sec  = index.sections[chunk.section_id]
        path = " > ".join(sec.path)
        blocks.append(f"[Source: {path}]\n{chunk.text}")
    return "\n\n".join(blocks)


def _llm_judge_with_history(
    llm:           LocalLLMClient,
    question:      str,
    context_str:   str,
    judge_history: List[Dict],
    iteration:     int,
    new_added:     int,
) -> Dict:
    """
    带历史记忆的 LLM 充分性判断器。

    相比原始 llm_judge：
    1. 把前几轮的推理过程和缺失判断拼入 prompt，LLM 能在前一轮
       思考的基础上继续，而不是每轮"失忆"重新推理。
    2. 要求 LLM 对比「上一轮缺什么」和「本轮新增了什么」，
       从而生成与上一轮不同的 missing_keywords。

    返回 dict（与原 llm_judge 格式兼容）：
        {
          "is_sufficient":    bool,
          "answer":           str,   # is_sufficient=True 时填写
          "reasoning":        str,   # 本轮推理过程（存入历史）
          "missing_aspects":  str,   # 还缺哪些方面（自然语言）
          "missing_keywords": list,  # 用于下一轮检索扩展
        }
    """

    # ── 拼装历史摘要 ───────────────────────────────────────────────────────
    if judge_history:
        history_lines = []
        for h in judge_history:
            history_lines.append(
                f"[Iteration {h['iteration']}]\n"
                f"  Reasoning: {h['reasoning']}\n"
                f"  Still missing: {h['missing_aspects']}\n"
                f"  Keywords tried next: {', '.join(h['missing_keywords'])}"
            )
        history_block = (
            "=== Previous reasoning history ===\n"
            + "\n\n".join(history_lines)
            + "\n==================================="
        )
    else:
        history_block = "(This is the first iteration — no prior reasoning history.)"

    # ── 构造 prompt ────────────────────────────────────────────────────────
    prompt = f"""You are a rigorous reading-comprehension judge working iteratively.

Question: {question}

{history_block}

=== Current context (iteration {iteration}, {new_added} new chunks added) ===
{context_str}
===================================================================

Your task:
1. Read the question and the current context carefully.
2. Review the previous reasoning history above. Note what was ALREADY identified as missing in prior iterations, and whether the new chunks address those gaps.
3. Decide: does the current context contain enough information to answer the question?

If YES — provide the final answer.
If NO — identify what is STILL missing. Your "missing_aspects" and "missing_keywords" MUST differ from previous iterations (do not repeat keywords that were already tried).

Reasoning requirements (MUST follow for every iteration, whether sufficient or not):
- In "reasoning", you MUST explicitly quote the exact sentences or phrases from the context above that are relevant to the question, and state the [Source: ...] section path they come from.
- Format each evidence piece as: [Source: <section path>] "<exact quote>" → <what this tells us>
- If multiple chunks contribute, list each one separately.
- Do NOT paraphrase without quoting — the exact original wording is required.

Respond ONLY with a valid JSON object, no markdown, no extra text, Use \"escaped quotes\" for any internal quotation marks, Double check that the JSON is valid and all quotes are escaped.:
{{
  "is_sufficient": true | false,
  "answer": "<concise answer, or empty string if not sufficient>",
  "reasoning": "<step-by-step reasoning with mandatory source-anchored quotes as described above>",
  "supporting_evidence": [
    {{
      "section_path": "<exact [Source: ...] path as shown in context>",
      "quote": "<exact sentence or phrase quoted from that section>",
      "relevance": "<one sentence: how this quote helps answer the question>"
    }}
  ],
  "missing_aspects": "<if not sufficient: a natural-language description of what is still needed>",
  "missing_keywords": ["<keyword1>", "<keyword2>", ...]
}}""".strip()

    try:
        raw     = llm.chat(prompt)
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        result  = json.loads(cleaned)
        # 保证必要字段存在
        result.setdefault("is_sufficient",      False)
        result.setdefault("answer",             "")
        result.setdefault("reasoning",          "")
        result.setdefault("supporting_evidence",[])
        result.setdefault("missing_aspects",    "")
        result.setdefault("missing_keywords",   [])
        return result
    except Exception as e:
        print(f"  [judge] parse error at iter={iteration}: {e}")
        return {
            "is_sufficient":      False,
            "answer":             "",
            "reasoning":          "",
            "supporting_evidence":[],
            "missing_aspects":    "",
            "missing_keywords":   [],
        }


def run_dirrag_query_musique(
    question:    str,
    passages:    List[Dict],   # [{"title": str, "content": str}, ...]
    embed_model: EmbeddingModel,
    llm:         LocalLLMClient,
    cfg:         DirRAGConfig,
    cache:       Optional[EntityCache] = None,
) -> str:
    """
    musique 版 DirRAG pipeline。

    与 Qasper 版（run_dirrag_query）的唯一区别是索引构建逻辑：
        - Qasper：passage 带有多级 path 字符串，需要解析层次
        - musique：每个 passage 的 title 直接作为独立的一级目录节点（depth=1），
                    所有节点深度相同，不存在父子关系

    索引建好之后，后续的检索、迭代、judge 逻辑与 Qasper 版完全相同。
    """

    # ── 建立 per-query 索引（musique 目录结构）─────────────────────────────
    index = DirRAGIndex(embed_model, cfg)

    for passage in passages:
        title   = passage["title"].strip()
        content = passage["content"].strip()

        if not title or not content:
            continue

        index.add_passage(
            title   = title,
            content = content,
            depth   = 1,
            path    = [title],
            llm     = llm,
            cache   = cache,
        )

    index.build()
    retriever = DirRAGRetriever(index)

    # ── 以下与 run_dirrag_query 完全相同 ──────────────────────────────────
    q_emb      = embed_model.encode([question])[0]
    q_entities = extract_entities(question, llm, cache)
    q_keywords = extract_keywords(question)

    all_chunks:       List[Chunk] = []
    used_section_ids: Set[str]    = set()
    accumulated_cids: Set[str]    = set()
    extra_keywords    = list(q_keywords)
    extra_entities    = list(q_entities)

    judge_history:    List[Dict] = []
    used_missing_kws: Set[str]   = set(extra_keywords)
    reasoning_chain:  List[Dict] = []

    try:
        for iteration in range(1, cfg.max_iter + 1):

            if cfg.use_section_routing:
                top_sections = retriever.retrieve_sections(
                    q_emb            = q_emb,
                    q_keywords       = extra_keywords,
                    q_entities       = extra_entities,
                    used_section_ids = used_section_ids,
                )
                for sec, _ in top_sections:
                    used_section_ids.add(sec.section_id)

                new_chunks = retriever.retrieve_chunks_from_sections(
                    q_emb        = q_emb,
                    q_keywords   = extra_keywords,
                    q_entities   = extra_entities,
                    top_sections = [s for s, _ in top_sections],
                )
            else:
                new_chunks = retriever.retrieve_chunks_global(
                    q_emb      = q_emb,
                    q_keywords = extra_keywords,
                    q_entities = extra_entities,
                )

            truly_new = [c for c in new_chunks if c.chunk_id not in accumulated_cids]
            for c in truly_new:
                accumulated_cids.add(c.chunk_id)
                all_chunks.append(c)

            context_str  = build_context_string(all_chunks, index)
            judge_result = _llm_judge_with_history(
                llm           = llm,
                question      = question,
                context_str   = context_str,
                judge_history = judge_history,
                iteration     = iteration,
                new_added     = len(truly_new),
            )

            iter_record = {
                "iteration":          iteration,
                "new_chunks_added":   len(truly_new),
                "is_sufficient":      judge_result.get("is_sufficient", False),
                "reasoning":          judge_result.get("reasoning", ""),
                "supporting_evidence":judge_result.get("supporting_evidence", []),
                "missing_aspects":    judge_result.get("missing_aspects", ""),
                "missing_keywords":   judge_result.get("missing_keywords", []),
            }
            reasoning_chain.append(iter_record)

            if judge_result.get("is_sufficient"):
                print(f"  ✅  LLM judged sufficient at iteration {iteration}.")
                return judge_result.get("answer", ""), reasoning_chain

            if cfg.use_iteration:
                missing = judge_result.get("missing_keywords", [])
                new_kws = [kw for kw in missing if kw.lower() not in used_missing_kws]
                if new_kws:
                    print(f"  ⚙️   iter={iteration} 新增关键词: {new_kws}")
                    extra_keywords = list(dict.fromkeys(
                        extra_keywords + [kw.lower() for kw in new_kws]
                    ))
                    extra_entities = list(dict.fromkeys(
                        extra_entities + [e for e in new_kws if e[0].isupper()]
                    ))
                    used_missing_kws.update(kw.lower() for kw in new_kws)
                else:
                    print(f"  ⚙️   iter={iteration} 无新关键词，检索方向已收敛。")

                judge_history.append({
                    "iteration":        iteration,
                    "new_chunks_added": len(truly_new),
                    "reasoning":        judge_result.get("reasoning", ""),
                    "missing_aspects":  judge_result.get("missing_aspects", ""),
                    "missing_keywords": missing,
                })

        # ── Fallback ───────────────────────────────────────────────────────
        print(f"  ⚠️   max_iter={cfg.max_iter} reached. Forcing final answer.")
        context_str  = build_context_string(all_chunks, index)
        final_prompt = (
            f"You are a precise reading-comprehension assistant.\n"
            f"Answer the question using ONLY the provided context.\n\n"
            f"Question: {question}\n\nContext:\n{context_str}\n\n"
            f"Key context: <quote(s)>\nReasoning: <step-by-step>\n"
            f"Confidence: <High|Medium|Low> — <reason>\n"
            f"Final answer: <concise answer>\n"
        )
        return llm.chat(final_prompt).strip(), reasoning_chain

    finally:
        index.destroy()


# ═══════════════════════════════════════════════════════════════════════════
# ──────────────────────────────  MAIN  ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def load_musique(input_path: str) -> List[Dict]:
    """
    加载 musique JSONL 数据。

    原始字段映射：
        _id      → _id
        input    → question
        answers  → gold_answer（取 answers[0]；若为列表则 join）
        passages → passages（保持原样，由 run_dirrag_query_musique 处理）
    """
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            answers = raw.get("answers", [])
            if isinstance(answers, list):
                gold = answers[0] if answers else ""
            else:
                gold = str(answers)

            samples.append({
                "_id":         raw["_id"],
                "question":    raw["input"],
                "gold_answer": gold,
                "passages":    raw["passages"],   # List[{"title": str, "content": str}]
            })
    return samples


def sample_dataset(dataset: List[Dict], size: Optional[int], seed: int) -> List[Dict]:
    if size is None or size >= len(dataset):
        return dataset
    rng = random.Random(seed)
    return rng.sample(dataset, size)


def setup_env(cfg: DirRAGConfig):
    os.environ["CUDA_DEVICE_ORDER"]         = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]      = "0,1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]   = "expandable_segments:True,garbage_collection_threshold:0.6"
    os.environ["TOKENIZERS_PARALLELISM"]    = "false"


def main():
    parser = argparse.ArgumentParser(description="DirRAG v3.0 消融实验 —— musique")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--variant",     type=str, help="消融变体名（见 --list）")
    group.add_argument("--config_path", type=str, help="从 JSON 文件加载自定义配置")
    group.add_argument("--list",        action="store_true", help="列出所有可用变体并退出")

    parser.add_argument("--sample_size", type=int, default=None, help="覆盖子集大小")
    parser.add_argument("--output_path", type=str, default=None, help="覆盖输出路径")
    args = parser.parse_args()

    if args.list:
        list_variants()
        return

    # ── 加载配置 ────────────────────────────────────────────────────────────
    if args.config_path:
        cfg = DirRAGConfig.load(args.config_path)
        print(f"[Config] 从文件加载：{args.config_path}")
    elif args.variant:
        cfg = get_config(args.variant)
        print(f"[Config] 使用预设变体：{args.variant}")
    else:
        print("[Config] 未指定变体，使用 Full DirRAG（Baseline）")
        cfg = get_config("full_dirrag")

    # musique 默认路径覆盖（当配置仍指向 Qasper 路径时自动修正）
    WIKIMQA_INPUT  = "/root/dir-sem-rag/longbench/data_process/musique_clean.jsonl"
    WIKIMQA_OUTDIR = "./results_musique"
    if cfg.input_path != WIKIMQA_INPUT:
        print(f"[Config] input_path 已覆盖为 musique 路径：{WIKIMQA_INPUT}")
        cfg.input_path = WIKIMQA_INPUT
    if not cfg.output_path.startswith(WIKIMQA_OUTDIR):
        new_out = cfg.output_path.replace("./results/", f"{WIKIMQA_OUTDIR}/", 1)
        if new_out == cfg.output_path:   # 路径没有 ./results/ 前缀时的兜底
            new_out = f"{WIKIMQA_OUTDIR}/{cfg.variant_name}.jsonl"
        print(f"[Config] output_path 已覆盖为：{new_out}")
        cfg.output_path = new_out

    # 命令行覆盖（优先级最高）
    if args.sample_size is not None:
        cfg.sample_size = args.sample_size
    if args.output_path is not None:
        cfg.output_path = args.output_path

    setup_env(cfg)
    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    cfg.save()

    # ── 打印配置摘要 ────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  DirRAG v3.0 [musique] — 消融实验：{cfg.variant_name}")
    print(f"  {cfg.description}")
    print("═" * 70)
    print(f"  use_section_routing : {cfg.use_section_routing}")
    print(f"  use_iteration       : {cfg.use_iteration}  (max_iter={cfg.max_iter})")
    print(f"  title/summary blend : {cfg.title_weight} / {cfg.summary_weight}")
    print(f"  SEC weights         : α={cfg.sec_alpha} β={cfg.sec_beta} γ={cfg.sec_gamma} δ={cfg.sec_delta}")
    print(f"  CHK weights         : α={cfg.chk_alpha} β={cfg.chk_beta} γ={cfg.chk_gamma} δ={cfg.chk_delta}")
    print(f"  sample_size         : {cfg.sample_size}  (seed={cfg.sample_seed})")
    print(f"  output              : {cfg.output_path}")
    print("═" * 70 + "\n")

    embed_model = EmbeddingModel()
    llm         = LocalLLMClient()

    # ── 实体缓存：与 Qasper 版共享同一份缓存文件 ──────────────────────────
    # 路径固定为 ./results/entity_cache.json，跨数据集、跨变体均可复用
    cache_path   = "./results/entity_cache.json"
    entity_cache = EntityCache(cache_path)

    # ── 数据加载（musique 专用）────────────────────────────────────────────
    dataset  = load_musique(cfg.input_path)
    dataset  = sample_dataset(dataset, cfg.sample_size, cfg.sample_seed)
    done_ids = load_completed_sample_ids(cfg.output_path)
    print(f"  数据集大小：{len(dataset)}  已完成：{len(done_ids)}\n")

    total_em  = 0
    total_f1  = 0.0
    processed = 0

    for sample in tqdm(dataset, desc=cfg.variant_name):
        sid = sample["_id"]
        if sid in done_ids:
            continue

        question = sample["question"]
        gold     = sample["gold_answer"]
        passages = sample["passages"]

        reasoning_chain: List[Dict] = []
        try:
            pred, reasoning_chain = run_dirrag_query_musique(
                question    = question,
                passages    = passages,
                embed_model = embed_model,
                llm         = llm,
                cfg         = cfg,
                cache       = entity_cache,
            )
        except Exception as e:
            print(f"  ❌ Error [{sid}]: {e}")
            pred = ""

        em, f1 = calculate_qa_metrics(pred, gold)

        record = {
            "sample_id":   sid,
            "question":    question,
            "gold_answer": gold,
            "pred_answer": {
                "answer":          pred,
                "reasoning_chain": reasoning_chain,
            },
            "exact_match": em,
            "f1":          f1,
        }
        with open(cfg.output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_em  += em
        total_f1  += f1
        processed += 1

    print("\n" + "═" * 70)
    print(f"  变体：{cfg.variant_name}  [musique]")
    if processed > 0:
        print(f"  已处理  : {processed}")
        print(f"  Avg EM  : {total_em / processed:.4f}")
        print(f"  Avg F1  : {total_f1 / processed:.4f}")
    else:
        print("  无新样本被处理。")
    print(f"  {entity_cache.stats()}")
    print("═" * 70)


if __name__ == "__main__":
    main()