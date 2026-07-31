"""
dirrag_ablation_cloud_dir.py
======================
DirRAG v3.0 消融实验主入口 —— Cloud Dir 版本

与 HotpotQA 版的核心差异
------------------------
1. 全量预建索引：Cloud Dir 语料库在启动时一次性建立全局 DirRAGIndex，
   不再 per-query 动态建索引（HotpotQA 版）。
2. 实体识别直接复用 graph_cache.json：
   - Cache key 采用与 GraphRAG 相同的方式：md5(title + text[:100]) → "chunk_<md5>"
   - 命中时打印 "💾 [缓存命中]"，miss 时回退到 LLM 抽取
3. 语料库目录结构：使用 Cloud Dir 的原生多级目录（title0~title5 + depth）

数据格式（语料库 cloud_dir_manual_output.json）：
    [
      {
        "title0": "Cardiology",
        "title1": "Hypertension",
        ...
        "title5": "",
        "content": "...",
        "path": "/Cardiology/Hypertension/...",
        "depth": 5
      },
      ...
    ]

数据格式（QA multi_hop_qa.json）：
    [{"query": str, "answer": str}, ...]

用法示例
--------
python dirrag_ablation_cloud_dir.py --variant full_dirrag
python dirrag_ablation_cloud_dir.py --variant wo_section_routing
python dirrag_ablation_cloud_dir.py --variant wo_iteration
python dirrag_ablation_cloud_dir.py --variant title_only
python dirrag_ablation_cloud_dir.py --variant summary_only
python dirrag_ablation_cloud_dir.py --variant wo_depth_reward
python dirrag_ablation_cloud_dir.py --variant semantic_only
python dirrag_ablation_cloud_dir.py --list
python dirrag_ablation_cloud_dir.py --variant full_dirrag --sample_size 50
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

import sys
sys.path.insert(0, "/root/dir-sem-rag/gte-Qwen2-1.5B-instruct")

from embedding import EmbeddingModel, EMBEDDING_DIM
from llm import LocalLLMClient, llm_judge
from data_utils import (
    load_completed_sample_ids,
    save_single_result,
    calculate_qa_metrics,
)

# ─────────────────────────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────────────────────────
CLOUD_DIR_CORPUS    = "/root/dir-sem-rag/clouddirwiki/contentWithPath/result.json"
CLOUD_DIR_QA        = "/root/dir-sem-rag/clouddirwiki/contentWithPath/single_hop_qa.json"
GRAPH_CACHE   = "/root/dir-sem-rag/clouddirwiki/test/graph_cache.json"
CLOUD_DIR_OUTDIR    = "./single_results_cloud_dir"

CHUNK_TEXT_MAX_CHARS = 1500   # 与 GraphRAG 保持一致

# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

STOP_WORDS = {
    "a","an","the","is","are","was","were","and","or","but",
    "in","on","at","to","for","of","with","by","from","that",
    "this","it","its","be","as","do","did","has","have","had",
}

ENTITY_COLLECTION_NAME = "entities_cloud_dir"
ENTITY_MILVUS_BASE     = "/tmp/dirrag_cloud_dir_entity_milvus"


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
# GRAPH CACHE（与 GraphRAG 相同的缓存命中方式）
# ═══════════════════════════════════════════════════════════════════════════

class GraphEntityCache:
    """
    复用 graph_cache.json 中已识别的实体。

    Cache key 计算方式与 GraphRAG._make_chunk_id() 完全相同：
        chunk_id = "chunk_" + md5(title + text[:100])

    命中时：直接返回实体名称列表，打印 💾 [缓存命中]
    未命中时：回退到 LLM 抽取，结果不写回 graph_cache.json
              （graph_cache 由 GraphRAG 维护，此处只读）
    """

    def __init__(self, cache_path: str = GRAPH_CACHE):
        self._path  = cache_path
        self._store: Dict[str, Dict] = {}
        self._hits  = 0
        self._total = 0
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                self._store = json.load(f)
            print(f"[GraphEntityCache] 加载 graph_cache: {len(self._store)} 条  ← {self._path}")
        else:
            print(f"[GraphEntityCache] ⚠️  未找到 graph_cache: {self._path}，将全部使用 LLM 抽取")

    @staticmethod
    def make_chunk_id(title: str, text_prefix: str) -> str:
        """与 GraphRAG._make_chunk_id() 完全一致"""
        raw = (title + text_prefix).encode("utf-8")
        return "chunk_" + hashlib.md5(raw).hexdigest()

    def get_entities(self, title: str, text: str) -> Optional[List[str]]:
        """
        返回实体名称列表，未命中返回 None。
        text 应为 chunk 全文（会截取前 100 字符作为 key）。
        """
        self._total += 1
        chunk_id = self.make_chunk_id(title, text[:100])
        if chunk_id in self._store:
            self._hits += 1
            cached = self._store[chunk_id]
            # graph_cache 中 entities 是对象列表，提取 name 字段
            ents = cached.get("entities", [])
            names = [e["name"] for e in ents if isinstance(e, dict) and e.get("name")]
            print(f"    💾 [缓存命中] {title[:50]}")
            return names
        return None

    def stats(self) -> str:
        rate = self._hits / max(self._total, 1) * 100
        return (
            f"[GraphEntityCache] 命中率 {self._hits}/{self._total} "
            f"({rate:.1f}%)  缓存条目 {len(self._store)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# ENTITY EXTRACTION（带 graph_cache 命中）
# ═══════════════════════════════════════════════════════════════════════════

def extract_entities(
    text:         str,
    llm:          LocalLLMClient,
    graph_cache:  Optional[GraphEntityCache] = None,
    title:        str = "",
) -> List[str]:
    """
    实体抽取：优先查 graph_cache，未命中则调用 LLM。
    title 用于构造 graph_cache key（与 GraphRAG 一致）。
    """
    if not text or len(text.strip()) < 10:
        return []

    # ── graph_cache 命中 ──────────────────────────────────────────────────
    if graph_cache is not None and title:
        cached = graph_cache.get_entities(title, text)
        if cached is not None:
            return cached

    # ── LLM 抽取（回退）─────────────────────────────────────────────────
    print(f"    🔍 [LLM抽取] {title[:50]}")
    prompt = f"""
You are a precise entity extractor.
Extract ALL important named entities from the following text.
Entities include: people, places, organizations, diseases, drugs, symptoms, medical procedures, proper nouns.
Output ONLY a valid JSON list of strings. No extra words, no markdown.
Text: {text}
""".strip()
    try:
        raw      = llm.chat(prompt, max_new_tokens=8192)
        cleaned  = re.sub(r"```(?:json)?|```", "", raw).strip()
        entities = json.loads(cleaned)
        if not isinstance(entities, list):
            entities = []
        entities = list(dict.fromkeys(str(e).strip() for e in entities if e))
    except Exception:
        entities = []

    return entities


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def extract_keywords(text: str) -> List[str]:
    return list(dict.fromkeys(tokenize(text)))


def make_doc_chunk(
    full_text:   str,
    section_id:  str,
    doc_offset:  int,
    llm:         LocalLLMClient,
    graph_cache: Optional[GraphEntityCache] = None,
    title:       str = "",
) -> Chunk:
    """
    与 GraphRAG 对齐的 chunking：每条语料记录整体作为一个 chunk。

    GraphRAG 中：
        text = f"{title}\\n{content}"          # 完整文本
        chunk_id = md5(title + text[:100])     # key

    这里 full_text 就是 f"{title}\\n{content}"，与 GraphRAG 完全一致，
    确保 graph_cache key 必然命中。
    """
    # 截断到 CHUNK_TEXT_MAX_CHARS，与 GraphRAG 保持一致
    text_for_index = full_text[:CHUNK_TEXT_MAX_CHARS]

    return Chunk(
        chunk_id     = str(uuid.uuid4()),
        section_id   = section_id,
        text         = text_for_index,
        entities     = extract_entities(text_for_index, llm, graph_cache, title),
        index_in_doc = doc_offset,
    )


# ═══════════════════════════════════════════════════════════════════════════
# BM25
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
# ENTITY INDEX
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

    def add_chunk(self, chunk: Chunk):
        """
        把 chunk 中已经识别好的实体写入 pending 列表。
        实体已在 make_doc_chunk() 中通过 graph_cache 获取，无需再调用 LLM。
        前三个实体（prominence proxy）获得 0.3 的 bonus，其余为 0。
        """
        prominent = set(chunk.entities[:3])
        tf_map: Dict[str, int] = defaultdict(int)
        for ent in chunk.entities:
            tf_map[ent] += 1
        for ent, tf in tf_map.items():
            bonus = 0.3 if ent in prominent else 0.0
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
# INDEX（全量预建，区别于 HotpotQA 版的 per-query 建索引）
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

        # path_str → section_id 映射，用于快速建立父子关系
        self._path_to_sid: Dict[str, str] = {}

    def add_doc(
        self,
        title:       str,
        content:     str,
        depth:       int,
        path:        List[str],
        llm:         LocalLLMClient,
        graph_cache: Optional[GraphEntityCache] = None,
    ):
        """
        添加一条 Cloud Dir 文档记录到索引。
        path 直接来自语料库的多级标题列表（已过滤空字符串）。
        """
        sec_id  = str(uuid.uuid4())
        section = Section(
            section_id = sec_id,
            title      = title,
            path       = path,
            depth      = depth,
        )

        # ── 与 GraphRAG 对齐：full_text = title + "\n" + content ─────────────
        # GraphRAG._make_chunk_id(title, text[:100]) 中的 text 就是这个格式，
        # 因此只要 full_text 一致，graph_cache key 必然命中。
        full_text = (title + "\n" + content).strip()

        chunk = make_doc_chunk(
            full_text   = full_text,
            section_id  = sec_id,
            doc_offset  = len(self._chunk_ids_ordered),
            llm         = llm,
            graph_cache = graph_cache,
            title       = title,
        )
        section.chunk_ids.append(chunk.chunk_id)
        self.chunks[chunk.chunk_id] = chunk
        self.entity_index.add_chunk(chunk)
        self._chunk_ids_ordered.append(chunk.chunk_id)

        all_ents: List[str] = []
        for cid in section.chunk_ids:
            all_ents.extend(self.chunks[cid].entities)
        section.entities = list(dict.fromkeys(all_ents))

        self.sections[sec_id] = section
        self._section_ids_ordered.append(sec_id)

        # 建立父子关系（通过 path 前缀匹配）
        path_str = " > ".join(path)
        self._path_to_sid[path_str] = sec_id

        if len(path) > 1:
            parent_path_str = " > ".join(path[:-1])
            parent_sid = self._path_to_sid.get(parent_path_str)
            if parent_sid:
                self.sections[parent_sid].children_ids.append(sec_id)
                section.parent_id = parent_sid

    def build(self):
        cfg = self.cfg
        print(f"  [Index] 计算 section 嵌入（{len(self._section_ids_ordered)} 个）...")
        title_texts   = [self.sections[s].title for s in self._section_ids_ordered]
        summary_texts = []
        for sid in self._section_ids_ordered:
            sec = self.sections[sid]
            raw = self.chunks[sec.chunk_ids[0]].text[:cfg.summary_chars] \
                  if sec.chunk_ids else sec.title
            summary_texts.append(raw)

        title_embs   = self.embed_model.encode(title_texts)
        summary_embs = self.embed_model.encode(summary_texts)

        section_embs = cfg.title_weight * title_embs + cfg.summary_weight * summary_embs
        for i, sid in enumerate(self._section_ids_ordered):
            self.sections[sid].embedding = section_embs[i]

        print(f"  [Index] 计算 chunk 嵌入（{len(self._chunk_ids_ordered)} 个）...")
        chunk_texts = [self.chunks[c].text for c in self._chunk_ids_ordered]
        if chunk_texts:
            chunk_embs = self.embed_model.encode(chunk_texts)
            for i, cid in enumerate(self._chunk_ids_ordered):
                self.chunks[cid].embedding = chunk_embs[i]

        self._section_bm25 = BM25Index(title_texts, cfg.bm25_k1, cfg.bm25_b)
        self._chunk_bm25   = BM25Index(chunk_texts if chunk_texts else [""],
                                       cfg.bm25_k1, cfg.bm25_b)
        print("  [Index] flush entity index...")
        self.entity_index.flush()
        print("  [Index] ✅ 构建完成。")

    def destroy(self):
        self.entity_index.destroy()


# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVER
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

        sem_scores = {
            sid: float(np.dot(q_emb, idx.sections[sid].embedding))
            for sid in sids
            if idx.sections[sid].embedding is not None
        }

        bm25_raw = idx._section_bm25.top_k(q_keywords, k=20)
        bm25_max = bm25_raw[0][1] if bm25_raw and bm25_raw[0][1] > 0 else 1.0
        bm25_scores = {sids[i]: s / bm25_max for i, s in bm25_raw}

        _, ent_section_ids = idx.entity_index.lookup_entities(q_entities)

        scored: List[Tuple[str, float]] = []
        for sid in sids:
            sec    = idx.sections[sid]
            s_sem  = sem_scores.get(sid, 0.0)
            s_bm25 = bm25_scores.get(sid, 0.0)
            s_ent  = (1.0 if sid in ent_section_ids else 0.0) if q_entities else 0.0
            s_dep  = min(sec.depth / max(self.max_depth, 1), 1.0)

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

    def _collect_descendant_chunk_ids(self, section_id: str, visited: Set[str]) -> Set[str]:
        """
        递归收集某个 section 及其所有后代 section 的 chunk_ids。
        visited 防止循环引用（理论上不应有，但加上更安全）。
        """
        if section_id in visited:
            return set()
        visited.add(section_id)

        idx = self.index
        result: Set[str] = set()

        sec = idx.sections.get(section_id)
        if sec is None:
            return result

        for cid in sec.chunk_ids:
            result.add(cid)

        for child_sid in sec.children_ids:
            result |= self._collect_descendant_chunk_ids(child_sid, visited)

        return result


    def retrieve_chunks_from_sections(
        self,
        q_emb:        np.ndarray,
        q_keywords:   List[str],
        q_entities:   List[str],
        top_sections: List[Section],
    ) -> List[Chunk]:
        candidate_cids: Set[str] = set()
        visited: Set[str] = set()

        for sec in top_sections:
            # 递归收集该 section 及其所有后代的 chunks
            candidate_cids |= self._collect_descendant_chunk_ids(sec.section_id, visited)

        if not candidate_cids:
            return []
        return self._score_and_expand(q_emb, q_keywords, q_entities, candidate_cids)

    def retrieve_chunks_global(
        self,
        q_emb:      np.ndarray,
        q_keywords: List[str],
        q_entities: List[str],
    ) -> List[Chunk]:
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
# PIPELINE
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

Respond ONLY with a valid JSON object, no markdown, no extra text. Use \\"escaped quotes\\" for any internal quotation marks:
{{
  "is_sufficient": true | false,
  "answer": "<concise answer, or empty string if not sufficient>",
  "reasoning": "<step-by-step reasoning with mandatory source-anchored quotes>",
  "supporting_evidence": [
    {{
      "section_path": "<exact [Source: ...] path>",
      "quote": "<exact sentence or phrase quoted>",
      "relevance": "<one sentence: how this quote helps answer the question>"
    }}
  ],
  "missing_aspects": "<if not sufficient: what is still needed>",
  "missing_keywords": ["<keyword1>", "<keyword2>", ...]
}}""".strip()

    try:
        raw     = llm.chat(prompt, max_new_tokens=8192)
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        result  = json.loads(cleaned)
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
            "is_sufficient":       False,
            "answer":              "",
            "reasoning":           "",
            "supporting_evidence": [],
            "missing_aspects":     "",
            "missing_keywords":    [],
        }


def run_dirrag_query_cloud_dir(
    question:    str,
    index:       DirRAGIndex,      # 预建好的全局索引（与 HotpotQA 版最大区别）
    retriever:   DirRAGRetriever,
    embed_model: EmbeddingModel,
    llm:         LocalLLMClient,
    cfg:         DirRAGConfig,
    graph_cache: Optional[GraphEntityCache] = None,
) -> Tuple[str, List[Dict]]:
    """
    Cloud Dir 版 DirRAG query pipeline。

    索引已在 main() 中预建，此函数只做检索+迭代判断。
    """
    q_emb      = embed_model.encode([question])[0]
    q_entities = extract_entities(question, llm, graph_cache, title="")
    q_keywords = extract_keywords(question)

    all_chunks:       List[Chunk] = []
    used_section_ids: Set[str]    = set()
    accumulated_cids: Set[str]    = set()
    extra_keywords    = list(q_keywords)
    extra_entities    = list(q_entities)

    judge_history:    List[Dict] = []
    used_missing_kws: Set[str]   = set(extra_keywords)
    reasoning_chain:  List[Dict] = []

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
            "iteration":           iteration,
            "new_chunks_added":    len(truly_new),
            "is_sufficient":       judge_result.get("is_sufficient", False),
            "reasoning":           judge_result.get("reasoning", ""),
            "supporting_evidence": judge_result.get("supporting_evidence", []),
            "missing_aspects":     judge_result.get("missing_aspects", ""),
            "missing_keywords":    judge_result.get("missing_keywords", []),
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

    # Fallback
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


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_cloud_dir_corpus(corpus_path: str) -> List[Dict]:
    """
    加载 Cloud Dir 语料库，返回结构化文档列表。
    每条记录包含: title, content, depth, path（list）
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for doc in raw:
        titles = [doc.get(f"title{i}", "") for i in range(6)]
        path   = [t.strip() for t in titles if t.strip()]
        if not path:
            continue
        title   = " - ".join(path)
        content = doc.get("content", "").strip()
        depth   = doc.get("depth", len(path))
        if not content:
            continue
        docs.append({
            "title":   title,
            "content": content,
            "depth":   depth,
            "path":    path,
        })
    return docs


def load_cloud_dir_qa(qa_path: str) -> List[Dict]:
    """
    加载 multi_hop_qa.json，格式: [{"query": str, "answer": str}, ...]
    返回统一格式: [{"_id": str, "question": str, "gold_answer": str}, ...]
    """
    with open(qa_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = []
    for i, item in enumerate(raw):
        samples.append({
            "_id":         f"cloud_dir_q{i:05d}",
            "question":    item["query"],
            "gold_answer": item["answer"],
        })
    return samples


def sample_dataset(dataset: List[Dict], size: Optional[int], seed: int) -> List[Dict]:
    if size is None or size >= len(dataset):
        return dataset
    rng = random.Random(seed)
    return rng.sample(dataset, size)


def setup_env(cfg: DirRAGConfig):
    os.environ["CUDA_DEVICE_ORDER"]       = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]    = cfg.cuda_visible_devices
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"
    os.environ["TOKENIZERS_PARALLELISM"]  = "false"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DirRAG v3.0 消融实验 —— Cloud Dir")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--variant",     type=str, help="消融变体名（见 --list）")
    group.add_argument("--config_path", type=str, help="从 JSON 文件加载自定义配置")
    group.add_argument("--list",        action="store_true", help="列出所有可用变体并退出")

    parser.add_argument("--sample_size", type=int,  default=None, help="限制 QA 样本数")
    parser.add_argument("--output_path", type=str,  default=None, help="覆盖输出路径")
    parser.add_argument("--corpus_path", type=str,  default=CLOUD_DIR_CORPUS, help="Cloud Dir 语料库路径")
    parser.add_argument("--qa_path",     type=str,  default=CLOUD_DIR_QA,     help="QA 文件路径")
    parser.add_argument("--cache_path",  type=str,  default=GRAPH_CACHE, help="graph_cache.json 路径")
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

    # 强制覆盖为 Cloud Dir 路径
    cfg.input_path = args.qa_path
    if args.output_path:
        cfg.output_path = args.output_path
    else:
        os.makedirs(CLOUD_DIR_OUTDIR, exist_ok=True)
        cfg.output_path = f"{CLOUD_DIR_OUTDIR}/{cfg.variant_name}.jsonl"

    if args.sample_size is not None:
        cfg.sample_size = args.sample_size

    setup_env(cfg)
    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    cfg.save()

    # ── 打印配置摘要 ────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  DirRAG v3.0 [Cloud Dir] — 消融实验：{cfg.variant_name}")
    print(f"  {cfg.description}")
    print("═" * 70)
    print(f"  use_section_routing : {cfg.use_section_routing}")
    print(f"  use_iteration       : {cfg.use_iteration}  (max_iter={cfg.max_iter})")
    print(f"  title/summary blend : {cfg.title_weight} / {cfg.summary_weight}")
    print(f"  SEC weights         : α={cfg.sec_alpha} β={cfg.sec_beta} γ={cfg.sec_gamma} δ={cfg.sec_delta}")
    print(f"  CHK weights         : α={cfg.chk_alpha} β={cfg.chk_beta} γ={cfg.chk_gamma} δ={cfg.chk_delta}")
    print(f"  sample_size         : {cfg.sample_size}  (seed={cfg.sample_seed})")
    print(f"  corpus              : {args.corpus_path}")
    print(f"  qa                  : {args.qa_path}")
    print(f"  graph_cache         : {args.cache_path}")
    print(f"  output              : {cfg.output_path}")
    print("═" * 70 + "\n")

    embed_model = EmbeddingModel()
    llm         = LocalLLMClient()

    # ── graph_cache（只读，复用 GraphRAG 已识别实体）───────────────────────
    graph_cache = GraphEntityCache(args.cache_path)

    # ── 加载语料库 ─────────────────────────────────────────────────────────
    print(f"📂 加载 Cloud Dir 语料库: {args.corpus_path}")
    corpus = load_cloud_dir_corpus(args.corpus_path)
    print(f"   语料条数: {len(corpus)}")

    # ── 全量预建索引（cloud dir 版核心特征）──────────────────────────────────────
    print("\n🔨 开始构建全量索引...")
    index = DirRAGIndex(embed_model, cfg)
    for i, doc in enumerate(tqdm(corpus, desc="building index")):
        index.add_doc(
            title       = doc["title"],
            content     = doc["content"],
            depth       = doc["depth"],
            path        = doc["path"],
            llm         = llm,
            graph_cache = graph_cache,
        )
    index.build()
    retriever = DirRAGRetriever(index)
    print(f"✅ 索引构建完成: {len(index.sections)} sections, {len(index.chunks)} chunks\n")

    # ── 加载 QA ────────────────────────────────────────────────────────────
    qa_samples = load_cloud_dir_qa(args.qa_path)
    qa_samples = sample_dataset(qa_samples, cfg.sample_size, cfg.sample_seed)
    done_ids   = load_completed_sample_ids(cfg.output_path)
    print(f"📋 QA 样本总数: {len(qa_samples)}  已完成: {len(done_ids)}\n")

    total_em  = 0
    total_f1  = 0.0
    processed = 0

    for sample in tqdm(qa_samples, desc=cfg.variant_name):
        sid = sample["_id"]
        if sid in done_ids:
            continue

        question = sample["question"]
        gold     = sample["gold_answer"]

        reasoning_chain: List[Dict] = []
        pred = ""
        try:
            pred, reasoning_chain = run_dirrag_query_cloud_dir(
                question    = question,
                index       = index,
                retriever   = retriever,
                embed_model = embed_model,
                llm         = llm,
                cfg         = cfg,
                graph_cache = graph_cache,
            )
        except Exception as e:
            print(f"  ❌ Error [{sid}]: {e}")

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

    # ── 清理 Milvus ────────────────────────────────────────────────────────
    index.destroy()

    print("\n" + "═" * 70)
    print(f"  变体：{cfg.variant_name}  [Cloud Dir]")
    if processed > 0:
        print(f"  已处理  : {processed}")
        print(f"  Avg EM  : {total_em / processed:.4f}")
        print(f"  Avg F1  : {total_f1 / processed:.4f}")
    else:
        print("  无新样本被处理。")
    print(f"  {graph_cache.stats()}")
    print("═" * 70)


if __name__ == "__main__":
    main()