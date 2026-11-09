                       

import os
import sys
import json
import re
import time
import string
import gc
import hashlib
import ast
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set, Any

import numpy as np
from scipy.sparse import csr_matrix

                                                                             
MODEL_PATH = "/root/dir-sem-rag/Qwen2.5-7B-Instruct"

sys.path.insert(0, "/root/dir-sem-rag/gte-Qwen2-1.5B-instruct")
sys.path.insert(1, "/root/dir-sem-rag/longbench/test/dir_rag_section_chunk_claude")
from embedding import EmbeddingModel
from llm import LocalLLMClient

           
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
try:
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass


def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    return " ".join(s.split())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def min_max_normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-10:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def text_processing(text_list: List[str]) -> List[str]:
    return [re.sub(r'\s+', ' ', re.sub(r'[%s]' % re.escape(string.punctuation), ' ',
              t.lower())).strip() for t in text_list]


def compute_mdhash(content: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(content.encode("utf-8")).hexdigest()


def parse_llm_json(response: str) -> Optional[Any]:
    text = response.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group()
        try:
            return json.loads(json_str)
        except Exception:
            pass
    return None


def personalized_pagerank_sparse(
    edge_indices: list,
    edge_weights: list,
    num_nodes: int,
    reset: np.ndarray,
    damping: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> np.ndarray:
    if num_nodes == 0:
        return np.array([])
    if not edge_indices:
                            
        p = reset.copy()
        s = p.sum()
        return p / s if s > 0 else np.ones(num_nodes) / num_nodes

                      
    rows, cols, data = [], [], []
    for (u, v), w in zip(edge_indices, edge_weights):
        rows.append(u); cols.append(v); data.append(w)
        rows.append(v); cols.append(u); data.append(w)
    A = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

                       
    out_degree = np.array(A.sum(axis=1)).flatten()

                      
    for i in range(num_nodes):
        if out_degree[i] < 1e-12:
            A[i, i] = 1.0
            out_degree[i] = 1.0

              
    inv_deg = 1.0 / out_degree
    T = A.multiply(inv_deg.reshape(-1, 1))

                         
    p = reset.copy()
    s = p.sum()
    p = p / s if s > 0 else np.ones(num_nodes) / num_nodes

    for _ in range(max_iter):
        p_new = damping * (T.T @ p) + (1 - damping) * reset
        s = p_new.sum()
        p_new = p_new / s if s > 0 else p_new
        delta = np.linalg.norm(p_new - p, ord=1)
        p = p_new
        if delta < tol:
            break
    return p


class Chunk:
    def __init__(self, chunk_id: int, text: str, article_title: str,
                 index_in_article: int, sent_ids: Set[int]):
        self.chunk_id = chunk_id
        self.text = text
        self.article_title = article_title
        self.index_in_article = index_in_article
        self.sent_ids = sent_ids


class OpenIECache:

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, list]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.loads(f.read())
                print(f"🗂️  OpenIE 缓存已加载: {len(self._cache)} 条记录")
            except Exception as e:
                print(f"⚠️ OpenIE 缓存加载失败: {str(e)[:50]}")
                self._cache = {}

    def _save(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(self._cache, ensure_ascii=False))
        except Exception as e:
            print(f"⚠️ OpenIE 缓存保存失败: {str(e)[:50]}")

    @staticmethod
    def make_key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[Dict[str, list]]:
        return self._cache.get(self.make_key(text))

    def set(self, text: str, entities: List[str], triples: List[List[str]]):
        self._cache[self.make_key(text)] = {
            "entities": entities,
            "triples": triples,
        }

    def flush(self):
        self._save()

    def __len__(self):
        return len(self._cache)


class TrackedLLM:

    def __init__(self, llm_client, tokenizer=None):
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
                          
        self.indexing_call_count = 0
        self.indexing_input_tokens = 0
        self.indexing_output_tokens = 0
                                                      
        self.retrieval_call_count = 0
        self.retrieval_input_tokens = 0
        self.retrieval_output_tokens = 0

    def chat(self, prompt: str, stage: str = "query") -> str:
        self.call_count += 1
        response = self.llm_client.chat(prompt)
        if self.tokenizer is not None:
            try:
                in_tok = len(self.tokenizer.encode(prompt))
                out_tok = len(self.tokenizer.encode(response))
            except Exception:
                in_tok = len(prompt.split())
                out_tok = len(response.split())
        else:
            in_tok = len(prompt.split())
            out_tok = len(response.split())

        if stage == "indexing":
            self.indexing_call_count += 1
            self.indexing_input_tokens += in_tok
            self.indexing_output_tokens += out_tok
        elif stage == "retrieval":
                                                 
            self.retrieval_call_count += 1
            self.retrieval_input_tokens += in_tok
            self.retrieval_output_tokens += out_tok
        else:
                                        
            self.total_input_tokens += in_tok
            self.total_output_tokens += out_tok

        return response

    def reset(self):
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.retrieval_call_count = 0
        self.retrieval_input_tokens = 0
        self.retrieval_output_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def indexing_total_tokens(self) -> int:
        return self.indexing_input_tokens + self.indexing_output_tokens

    @property
    def retrieval_total_tokens(self) -> int:
        return self.retrieval_input_tokens + self.retrieval_output_tokens


OPENIE_PROMPT = """You are an information-extraction system.
Given the TEXT below, extract:
1. Named entities (people, organizations, locations, works, events, concepts).
2. Relationship triples in the form (subject, predicate, object).

TEXT: {text}

Respond ONLY with a single valid JSON object — no markdown, no explanation.
Schema:
{{
  "entities": ["entity1", "entity2", ...],
  "triples": [["subject", "predicate", "object"], ...]
}}
JSON:"""

FACT_RERANK_PROMPT = """You are a fact relevance judge.
Given a QUESTION and a list of CANDIDATE FACTS, select the most relevant facts for answering the question.

QUESTION: {query}

CANDIDATE FACTS (0-indexed):
{facts_text}

Respond ONLY with a JSON object:
{{
  "selected_indices": [0, 2, 5, ...]
}}
Select at most {top_k} most relevant facts. JSON:"""

QA_PROMPT = """Answer the question using the provided context or based on your own knowledge.

Context:
{context}

Question: {question}

Instructions:
1. Give the shortest correct answer phrase.
2. If you do NOT know the answer or the context does NOT contain enough information to answer the question, respond with exactly: "I don't know". Please do NOT randomly guess the answer.
3. Output ONLY the answer, nothing else (no explanation, no JSON, no prefix)."""


class HippoRAG2:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_client: TrackedLLM,
        chunk_size: int = 100,
        overlap: int = 32,
        top_k: int = 5,
                       
        linking_top_k: int = 10,
        synonymy_edge_topk: int = 10,
        synonymy_edge_sim_threshold: float = 0.8,
        passage_node_weight: float = 0.05,
        damping: float = 0.5,
        openie_cache: Optional[OpenIECache] = None,
    ):
        self.embedding_model = embedding_model
        self.llm = llm_client
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
                       
        self.linking_top_k = linking_top_k
        self.synonymy_edge_topk = synonymy_edge_topk
        self.synonymy_edge_sim_threshold = synonymy_edge_sim_threshold
        self.passage_node_weight = passage_node_weight
        self.damping = damping
        self.openie_cache = openie_cache

        self.reset()

    def reset(self):
             
        self.chunks: List[Chunk] = []
        self.chunk_id_to_idx: Dict[int, int] = {}
        self._chunk_id_counter = 0
        self._chunk_embeddings: List[np.ndarray] = []

                   
        self._chunk_entities: Dict[int, List[str]] = {}                           
        self._chunk_triples: Dict[int, List[List[str]]] = {}                      

                                  
        self._num_nodes: int = 0
        self._node_names: List[str] = []
        self._edge_indices: List[Tuple[int, int]] = []
        self._edge_weights: List[float] = []

            
        self._entity_names: List[str] = []
        self._entity_embeddings: List[np.ndarray] = []
        self._entity_name_to_idx: Dict[str, int] = {}
        self._entity_hash_to_name: Dict[str, str] = {}                      
        self._entity_hash_to_idx: Dict[str, int] = {}                                  
        self._entity_to_chunks: Dict[str, Set[int]] = {}                                  

                         
        self._fact_texts: List[str] = []
        self._fact_embeddings: List[np.ndarray] = []
        self._fact_to_chunks: Dict[str, Set[int]] = {}                         

              
        self._passage_hashes: List[str] = []                                 
        self._passage_hash_to_chunk_id: Dict[str, int] = {}

        gc.collect()

                                  
    def build_index(self, passages: List[dict]) -> dict:
        t0 = time.time()
        timing = {}
        self.reset()
                                         
        self.llm.indexing_call_count = 0
        self.llm.indexing_input_tokens = 0
        self.llm.indexing_output_tokens = 0

                    
        t1 = time.time()
        for passage in passages:
            title = passage.get("title", "")
            sentences = passage.get("sentence", [])
            if not sentences:
                content = passage.get("content", "")
                if content.strip():
                    self._chunk_text(title, content)
                continue
            word_list: List[str] = []
            word_sent_ids: List[int] = []
            for sid, sent in enumerate(sentences):
                ws = sent.split()
                word_list.extend(ws)
                word_sent_ids.extend([sid] * len(ws))
            if word_list:
                self._chunk_words(title, word_list, word_sent_ids)

        timing["chunking_time"] = round(time.time() - t1, 4)
        if not self.chunks:
            timing["embedding_time"] = 0.0
            timing["openie_time"] = 0.0
            timing["graph_construction_time"] = 0.0
            timing["total_time"] = round(time.time() - t0, 4)
            return timing

                                 
        t1 = time.time()
        try:
            texts = [c.text for c in self.chunks]
            embs = self.embedding_model.encode(texts)
            self._chunk_embeddings = [np.array(e, dtype=np.float32) for e in embs]
        except Exception as e:
            raise RuntimeError(f"Chunk embedding 失败: {str(e)[:80]}")
        timing["embedding_time"] = round(time.time() - t1, 4)

                                        
        t1 = time.time()
        self._perform_openie()
        timing["openie_time"] = round(time.time() - t1, 4)

                                         
        t1 = time.time()
        self._build_entity_and_fact_embeddings()
        timing["entity_fact_embedding_time"] = round(time.time() - t1, 4)

                        
        t1 = time.time()
        self._build_knowledge_graph()
        timing["graph_construction_time"] = round(time.time() - t1, 4)

        timing["total_time"] = round(time.time() - t0, 4)
        return timing

    def _chunk_text(self, title: str, content: str):
        words = content.split()
        sent_ids = [0] * len(words)
        self._chunk_words(title, words, sent_ids)

    def _chunk_words(self, title: str, words: List[str], sent_ids: List[int]):
        start = 0
        idx = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = Chunk(
                chunk_id=self._chunk_id_counter,
                text=" ".join(words[start:end]),
                article_title=title,
                index_in_article=idx,
                sent_ids=set(sent_ids[start:end]),
            )
            self.chunks.append(chunk)
            self.chunk_id_to_idx[chunk.chunk_id] = len(self.chunks) - 1
            self._chunk_id_counter += 1
            idx += 1
            if end == len(words):
                break
            start += self.chunk_size - self.overlap

                                                
    def _perform_openie(self):
        print(f"    🕸️  OpenIE: {len(self.chunks)} 个文本块 ...")
        cache_hit = 0

        for i, chunk in enumerate(self.chunks):
                     
            cached = None
            if self.openie_cache is not None:
                cached = self.openie_cache.get(chunk.text)

            if cached is not None:
                entities = cached.get("entities", [])
                triples = cached.get("triples", [])
                cache_hit += 1
            else:
                              
                entities, triples = self._llm_openie(chunk.text)
                if self.openie_cache is not None:
                    self.openie_cache.set(chunk.text, entities, triples)

            self._chunk_entities[chunk.chunk_id] = entities
            self._chunk_triples[chunk.chunk_id] = triples

               
        if self.openie_cache is not None and cache_hit < len(self.chunks):
            self.openie_cache.flush()

        if cache_hit > 0:
            print(f"    🗂️  OpenIE 缓存命中: {cache_hit}/{len(self.chunks)} 个文本块")

    def _llm_openie(self, text: str) -> Tuple[List[str], List[List[str]]]:
        prompt = OPENIE_PROMPT.format(text=text)
        raw = self.llm.chat(prompt, stage="indexing").strip()

        entities: List[str] = []
        triples: List[List[str]] = []

        parsed = parse_llm_json(raw)
        if isinstance(parsed, dict):
            ents = parsed.get("entities", [])
            if isinstance(ents, list):
                entities = [str(e).strip() for e in ents if e and str(e).strip()]

            tris = parsed.get("triples", [])
            if isinstance(tris, list):
                for t in tris:
                    if isinstance(t, (list, tuple)) and len(t) == 3:
                                                                    
                                                                       
                        raw_triple = [str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()]
                        if all(raw_triple):
                            triples.append(raw_triple)

        return entities, triples

                                                   
    def _build_entity_and_fact_embeddings(self):
                  
                                                 
        all_entities: Set[str] = set()
        for entities in self._chunk_entities.values():
            for e in entities:
                raw_name = e.strip()
                if raw_name:
                    all_entities.add(raw_name)

        self._entity_names = sorted(all_entities)
        if self._entity_names:
            try:
                ent_embs = self.embedding_model.encode(self._entity_names)
                self._entity_embeddings = [np.array(e, dtype=np.float32) for e in ent_embs]
                self._entity_name_to_idx = {name: i for i, name in enumerate(self._entity_names)}
            except Exception as e:
                raise RuntimeError(f"Entity embedding 失败: {str(e)[:80]}")

                                
        all_facts: Set[str] = set()
        for triples in self._chunk_triples.values():
            for t in triples:
                all_facts.add(str(t))

        self._fact_texts = sorted(all_facts)
        if self._fact_texts:
            try:
                fact_embs = self.embedding_model.encode(self._fact_texts)
                self._fact_embeddings = [np.array(e, dtype=np.float32) for e in fact_embs]
            except Exception as e:
                raise RuntimeError(f"Fact embedding 失败: {str(e)[:80]}")

                             
        for chunk_id, triples in self._chunk_triples.items():
            for t in triples:
                fact_key = str(t)
                self._fact_to_chunks.setdefault(fact_key, set()).add(chunk_id)

                                  
    def _build_knowledge_graph(self):
        if not self._entity_names and not self.chunks:
            return

                
        entity_ids: List[str] = []
        for name in self._entity_names:
            entity_ids.append(name)
            self._entity_hash_to_idx[name] = len(entity_ids) - 1
            self._entity_to_chunks[name] = set()

                
        passage_hashes: List[str] = []
        for chunk in self.chunks:
            pass_hash = compute_mdhash(chunk.text, prefix="chunk-")
            passage_hashes.append(pass_hash)
            self._passage_hash_to_chunk_id[pass_hash] = chunk.chunk_id

        self._passage_hashes = passage_hashes

                                
        all_names = entity_ids + passage_hashes
        self._node_names = all_names
        self._num_nodes = len(all_names)

               
        edge_weights: Dict[Tuple[str, str], float] = {}

        def _add_edge(h1: str, h2: str, weight: float):
            key = tuple(sorted((h1, h2)))
            edge_weights[key] = max(edge_weights.get(key, 0.0), weight)

                               
        for chunk_id, triples in self._chunk_triples.items():
            for triple in triples:
                if len(triple) != 3:
                    continue
                subj, _, obj = triple[0], triple[1], triple[2]

                if subj in self._entity_hash_to_idx and obj in self._entity_hash_to_idx:
                    _add_edge(subj, obj, 1.0)
                    self._entity_to_chunks.setdefault(subj, set()).add(chunk_id)
                    self._entity_to_chunks.setdefault(obj, set()).add(chunk_id)

                              
        for chunk_id, entities in self._chunk_entities.items():
            pass_hash = passage_hashes[chunk_id] if chunk_id < len(passage_hashes) else None
            if pass_hash is None:
                continue
            for entity_name in entities:
                raw_name = entity_name.strip()
                if not raw_name:
                    continue
                if raw_name in self._entity_hash_to_idx:
                    _add_edge(pass_hash, raw_name, 1.0)
                    self._entity_to_chunks.setdefault(raw_name, set()).add(chunk_id)

                                              
        if len(self._entity_embeddings) > 1:
            ent_mat = np.array(self._entity_embeddings)
            ent_mat_norm = ent_mat / (np.linalg.norm(ent_mat, axis=1, keepdims=True) + 1e-10)

            for i in range(len(self._entity_names)):
                sims = ent_mat_norm @ ent_mat_norm[i]
                            
                k = min(self.synonymy_edge_topk + 1, len(sims))
                top_idx = np.argpartition(-sims, k - 1)[:k]
                for j in top_idx:
                    if j == i:
                        continue
                    sim = float(sims[j])
                    if sim >= self.synonymy_edge_sim_threshold:
                                                           
                        h1 = entity_ids[i]
                        h2 = entity_ids[j]
                        _add_edge(h1, h2, sim)

                                   
        name_to_idx = {n: i for i, n in enumerate(all_names)}
        edges = list(edge_weights.keys())
        weights = [edge_weights[e] for e in edges]
        self._edge_indices = [(name_to_idx[e[0]], name_to_idx[e[1]]) for e in edges]
        self._edge_weights = weights

        print(f"    ✅ 图谱: {self._num_nodes} nodes, "
              f"{len(self._edge_indices)} edges")

                                
    def get_chunk_count(self) -> int:
        return len(self.chunks)

    def get_index_size_mb(self) -> float:
        total = sum(e.nbytes for e in self._chunk_embeddings)
        total += sum(e.nbytes for e in self._entity_embeddings)
        total += sum(e.nbytes for e in self._fact_embeddings)
        return total / (1024 * 1024)

    def get_stats(self) -> dict:
        return {
            "num_chunks": len(self.chunks),
            "num_entities": len(self._entity_names),
            "num_facts": len(self._fact_texts),
            "num_graph_nodes": self._num_nodes,
            "num_graph_edges": len(self._edge_indices),
        }

                                 
    def answer_question(self, question: str) -> Tuple[str, dict]:
        t_start = time.time()
        self.llm.reset()

        t_ret = time.time()

                                 
        try:
            q_emb = self.embedding_model.encode([question])[0]
        except Exception as e:
            raise RuntimeError(f"查询 embedding 失败，样本中止: {str(e)[:80]}")

        retrieval_info: dict = {"facts": {}, "ppr": {}, "dpr": {}}

                            
        query_fact_scores, top_facts = self._retrieve_facts(q_emb)
        retrieval_info["facts"] = {
            "num_candidates": len(self._fact_texts),
            "num_selected": len(top_facts),
            "top_facts": top_facts[:10],
        }

                                                 
        if top_facts:
            top_facts = self._rerank_facts(question, top_facts)
            retrieval_info["facts"]["num_after_rerank"] = len(top_facts)

                                          
        if top_facts:
            sorted_chunk_ids, ppr_scores = self._graph_search_with_ppr(
                question, q_emb, query_fact_scores, top_facts
            )
            retrieval_info["ppr"] = {"num_passages_scored": len(sorted_chunk_ids)}
        else:
                               
            sorted_chunk_ids = self._dense_passage_retrieval(q_emb)
            retrieval_info["dpr"] = {"used_dpr_only": True}

        retrieval_time = time.time() - t_ret

                              
        final_chunk_ids = sorted_chunk_ids[:self.top_k]

                           
        llm_result = self._answer_with_llm(question, final_chunk_ids)
        pred = llm_result.get("answer", "I don't know")
        if not llm_result.get("isSufficient", False):
            pred = "I don't know"

        e2e = time.time() - t_start

        iteration_details = [{
            "iteration": 1,
            "retrieved_dir_ids": [],
            "retrieved_chunk_ids": final_chunk_ids,
            "expanded_chunk_ids": final_chunk_ids,
            "context": llm_result.get("context", ""),
            "prompt": llm_result.get("prompt", ""),
            "llm_raw_response": llm_result.get("llm_raw_response", ""),
            "isSufficient": llm_result.get("isSufficient", False),
            "answer": llm_result.get("answer", ""),
            "reasoning": llm_result.get("reasoning", ""),
            "new_keywords": llm_result.get("new_keywords", []),
            "new_entities": llm_result.get("new_entities", []),
            "used_chunk_info": llm_result.get("used_chunk_info", []),
            "retrieval_time_s": round(retrieval_time, 4),
            "retrieval_info": retrieval_info,
        }]

        metrics = {
            "token_cost": {
                "input_tokens": self.llm.total_input_tokens,
                "output_tokens": self.llm.total_output_tokens,
                "total_tokens": self.llm.total_tokens,
            },
            "retrieval_token_cost": {
                "input_tokens": self.llm.retrieval_input_tokens,
                "output_tokens": self.llm.retrieval_output_tokens,
                "total_tokens": self.llm.retrieval_total_tokens,
                "llm_call_count": self.llm.retrieval_call_count,
            },
            "indexing_token_cost": {
                "input_tokens": self.llm.indexing_input_tokens,
                "output_tokens": self.llm.indexing_output_tokens,
                "total_tokens": self.llm.indexing_total_tokens,
                "llm_call_count": self.llm.indexing_call_count,
            },
            "llm_call_count": self.llm.call_count,
            "e2e_latency_s": round(e2e, 4),
            "retrieval_time_s": round(retrieval_time, 4),
            "iterations": 1,
            "iteration_details": iteration_details,
        }
        return pred, metrics

    def _retrieve_facts(self, q_emb: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        if not self._fact_embeddings:
            return np.array([]), []

        scores = np.array([cosine_similarity(q_emb, emb) for emb in self._fact_embeddings])
        normalized = min_max_normalize(scores)

                             
        k = min(self.linking_top_k, len(normalized))
        if k == 0:
            return normalized, []

        top_indices = np.argsort(normalized)[-k:][::-1].tolist()
        top_facts = [self._fact_texts[i] for i in top_indices]

        return normalized, top_facts

    def _rerank_facts(self, query: str, candidate_facts: List[str]) -> List[str]:
        if not candidate_facts:
            return []

        facts_text = "\n".join(
            f"{i}. {fact}" for i, fact in enumerate(candidate_facts)
        )
        prompt = FACT_RERANK_PROMPT.format(
            query=query,
            facts_text=facts_text,
            top_k=self.linking_top_k,
        )

        try:
            raw = self.llm.chat(prompt, stage="retrieval").strip()
            parsed = parse_llm_json(raw)
            if isinstance(parsed, dict):
                selected = parsed.get("selected_indices", [])
                if isinstance(selected, list):
                    reranked = []
                    for idx in selected:
                        try:
                            i = int(idx)
                            if 0 <= i < len(candidate_facts):
                                reranked.append(candidate_facts[i])
                        except (ValueError, TypeError):
                            continue
                    if reranked:
                        return reranked
        except Exception as e:
            print(f"⚠️ Fact 重排失败: {str(e)[:50]}")

                       
        return candidate_facts

    def _dense_passage_retrieval(self, q_emb: np.ndarray) -> List[int]:
        if not self._chunk_embeddings:
            return []

        scores = np.array([cosine_similarity(q_emb, emb) for emb in self._chunk_embeddings])
        normalized = min_max_normalize(scores)
        sorted_indices = np.argsort(normalized)[::-1].tolist()
        return sorted_indices

    def _graph_search_with_ppr(self, query: str, q_emb: np.ndarray,
                                 query_fact_scores: np.ndarray,
                                 top_facts: List[str]) -> Tuple[List[int], np.ndarray]:
        num_nodes = self._num_nodes
        if num_nodes == 0:
            return [], np.array([])

        phrase_weights = np.zeros(num_nodes, dtype=np.float64)
        passage_weights = np.zeros(num_nodes, dtype=np.float64)
        occurrence_counts = np.zeros(num_nodes, dtype=np.float64)

                                         
        for fact_text in top_facts:
            try:
                triple = ast.literal_eval(fact_text)
                if len(triple) != 3:
                    continue
            except Exception:
                parts = fact_text.strip("[]()'\" ").split(",")
                if len(parts) < 3:
                    continue
                triple = [p.strip().strip("'\"") for p in parts[:3]]

            subj = triple[0].strip()
            obj = triple[2].strip()

                                       
            fact_idx = -1
            for i, ft in enumerate(self._fact_texts):
                if ft == fact_text:
                    fact_idx = i
                    break
            if fact_idx < 0:
                continue

            fact_score = float(query_fact_scores[fact_idx]) if query_fact_scores.size > 0 else 0.5

            for phrase in [subj, obj]:
                                                                           
                vertex_idx = self._entity_hash_to_idx.get(phrase)
                if vertex_idx is not None:
                                                            
                                              
                    phrase_weights[vertex_idx] += fact_score
                    occurrence_counts[vertex_idx] += 1

                            
        mask = occurrence_counts > 0
        phrase_weights[mask] /= occurrence_counts[mask]

                         
        if self.linking_top_k > 0 and mask.any():
            nonzero_indices = np.where(mask)[0]
            nonzero_scores = phrase_weights[nonzero_indices]
            top_k = min(self.linking_top_k, len(nonzero_indices))
            top_idx = np.argpartition(-nonzero_scores, top_k - 1)[:top_k]
            new_phrase_weights = np.zeros(num_nodes, dtype=np.float64)
            for idx in nonzero_indices[top_idx]:
                new_phrase_weights[idx] = phrase_weights[idx]
            phrase_weights = new_phrase_weights

                                       
        dpr_sorted = self._dense_passage_retrieval(q_emb)
        dpr_scores = np.zeros(len(self.chunks), dtype=np.float64)
        if dpr_sorted:
            raw_scores = np.array([cosine_similarity(q_emb, emb) for emb in self._chunk_embeddings])
            dpr_scores = min_max_normalize(raw_scores)

                            
        num_entities = len(self._entity_names)
        for i, pass_hash in enumerate(self._passage_hashes):
            if i < len(dpr_scores):
                vertex_idx = num_entities + i
                passage_weights[vertex_idx] = dpr_scores[i] * self.passage_node_weight

                                     
        node_weights = phrase_weights + passage_weights

        if np.sum(node_weights) <= 0:
                                            
            return dpr_sorted, np.array([])

                           
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        total = np.sum(reset_prob)
        if total > 0:
            reset_prob = reset_prob / total

        try:
            pagerank_scores = personalized_pagerank_sparse(
                edge_indices=self._edge_indices,
                edge_weights=self._edge_weights,
                num_nodes=num_nodes,
                reset=reset_prob,
                damping=self.damping,
            )
        except Exception as e:
            print(f"⚠️ PPR 失败: {str(e)[:50]}")
            return dpr_sorted, np.array([])

                           
        passage_scores = np.array([pagerank_scores[num_entities + i] for i in range(len(self._passage_hashes))])
        sorted_passage_indices = np.argsort(passage_scores)[::-1]

                      
        sorted_chunk_ids = []
        for idx in sorted_passage_indices:
            pass_hash = self._passage_hashes[idx]
            chunk_id = self._passage_hash_to_chunk_id.get(pass_hash)
            if chunk_id is not None:
                sorted_chunk_ids.append(chunk_id)

        return sorted_chunk_ids, passage_scores[sorted_passage_indices]

                                  
    def _answer_with_llm(self, question: str, chunk_ids: List[int]) -> dict:
        context_parts: List[str] = []
        used_chunk_info: List[dict] = []

        for cid in chunk_ids:
            idx = self.chunk_id_to_idx.get(cid)
            if idx is not None:
                c = self.chunks[idx]
                context_parts.append(f"[{c.article_title}] {c.text}")
                used_chunk_info.append({
                    "chunk_id": c.chunk_id,
                    "article_title": c.article_title,
                    "index_in_article": c.index_in_article,
                    "text_preview": c.text[:200],
                })

        context = "\n\n".join(context_parts) if context_parts else "(no context)"

        prompt = QA_PROMPT.format(context=context, question=question)

        try:
            resp = self.llm.chat(prompt)
        except Exception as e:
            print(f"⚠️ LLM 回答失败: {str(e)[:50]}")
            return {
                "isSufficient": False, "answer": "I don't know", "reasoning": "",
                "new_keywords": [], "new_entities": [],
                "context": context, "prompt": prompt,
                "llm_raw_response": f"ERROR: {str(e)[:100]}",
                "used_chunk_info": used_chunk_info,
            }

                                                   
        answer = resp.strip()
                                                  
        is_sufficient = normalize_answer(answer) != normalize_answer("I don't know")
        if not is_sufficient:
            answer = "I don't know"

        return {
            "isSufficient": is_sufficient,
            "answer": answer,
            "reasoning": "",
            "new_keywords": [],
            "new_entities": [],
            "context": context, "prompt": prompt,
            "llm_raw_response": resp,
            "used_chunk_info": used_chunk_info,
        }

                                        
    def compute_recall_mrr(self, supporting_facts: List[dict],
                           retrieved_ordered: List[int]) -> dict:
        if not supporting_facts or not retrieved_ordered:
            return {"recall_at_k": 0.0, "mrr": 0.0, "hit_count": 0,
                    "total_evidence": len(supporting_facts), "first_hit_rank": 0}

        evidence: Set[Tuple[str, int]] = set()
        for sf in supporting_facts:
            t = sf.get("title", "")
            sid = sf.get("sent_id", -1)
            if t and sid >= 0:
                evidence.add((t, sid))

        total = len(evidence)
        if total == 0:
            return {"recall_at_k": 0.0, "mrr": 0.0, "hit_count": 0,
                    "total_evidence": 0, "first_hit_rank": 0}

        hit = 0
        first_rank = 0
        matched: Set[Tuple[str, int]] = set()

        for rank, cid in enumerate(retrieved_ordered, start=1):
            idx = self.chunk_id_to_idx.get(cid)
            if idx is None:
                continue
            c = self.chunks[idx]
            for sid in c.sent_ids:
                ev = (c.article_title, sid)
                if ev in evidence and ev not in matched:
                    matched.add(ev)
                    hit += 1
                    if first_rank == 0:
                        first_rank = rank

        return {
            "recall_at_k": round(hit / total, 4),
            "mrr": round(1.0 / first_rank if first_rank > 0 else 0.0, 4),
            "hit_count": hit,
            "total_evidence": total,
            "first_hit_rank": first_rank,
        }


def run_evaluation(
    samples: List[dict],
    local_llm: TrackedLLM,
    output_path: str,
    max_samples: Optional[int] = None,
    chunk_size: int = 100,
    overlap: int = 32,
    top_k: int = 5,
                   
    linking_top_k: int = 10,
    synonymy_edge_topk: int = 10,
    synonymy_edge_sim_threshold: float = 0.8,
    passage_node_weight: float = 0.05,
    damping: float = 0.5,
    openie_cache_path: str = "",
):
    print("📦 初始化 EmbeddingModel ...")
    embedding_model = EmbeddingModel()
    print("✅ Embedding model loaded.")

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        print("✅ Tokenizer loaded for token counting.")
    except Exception:
        print("⚠️ Tokenizer 不可用，使用词数估算 token。")

    if not isinstance(local_llm, TrackedLLM):
        local_llm = TrackedLLM(local_llm, tokenizer=tokenizer)

                   
    openie_cache = None
    if openie_cache_path:
        openie_cache = OpenIECache(openie_cache_path)
        print(f"🗂️  OpenIE 缓存已启用: {openie_cache_path}")

    rag = HippoRAG2(
        embedding_model=embedding_model,
        llm_client=local_llm,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        linking_top_k=linking_top_k,
        synonymy_edge_topk=synonymy_edge_topk,
        synonymy_edge_sim_threshold=synonymy_edge_sim_threshold,
        passage_node_weight=passage_node_weight,
        damping=damping,
        openie_cache=openie_cache,
    )

          
    processed_ids: set = set()
    if os.path.exists(output_path):
        print("🔄 加载已处理进度...")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except Exception:
                    continue

    all_target = samples[:max_samples] if max_samples else samples
    to_run = [s for s in all_target if s["_id"] not in processed_ids]
    total = len(to_run)

    print("\n" + "=" * 70)
    print(f"🚀 2wiki HippoRAG 2 | 知识图谱 + PPR | 剩余: {total}")
    print(f"✅ chunk_size={chunk_size}  top_k={top_k}")
    print(f"✅ linking_top_k={linking_top_k}  damping={damping}")
    print("=" * 70)

    for idx, sample in enumerate(to_run, 1):
        _id = sample["_id"]
        question = sample["question"]
        gold = sample["answer"]
        passages = sample["passages"]
        supporting_facts = sample.get("supporting_facts", [])

        print(f"\n📄 [{idx}/{total}] ID: {_id[:20]}...")
        print(f"Q: {question}")
        print(f"📰 文章数: {len(passages)}, 证据数: {len(supporting_facts)}")

        try:
            rag.reset()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

                          
            timing = rag.build_index(passages)
            total_chunks = rag.get_chunk_count()
            index_size = rag.get_index_size_mb()
            stats = rag.get_stats()

            print(f"📦 文本块: {total_chunks}, 实体: {stats['num_entities']}, "
                  f"事实: {stats['num_facts']}, 图节点: {stats['num_graph_nodes']}, "
                  f"图边: {stats['num_graph_edges']} | "
                  f"IndexTime={timing['total_time']:.3f}s, "
                  f"IndexSize={index_size:.2f}MB")

                           
            pred, metrics = rag.answer_question(question)

                                  
            retrieved = metrics["iteration_details"][0]["expanded_chunk_ids"]
            recall_m = rag.compute_recall_mrr(supporting_facts, retrieved)\
                if supporting_facts else {"recall_at_k": 0.0, "mrr": 0.0,
                                          "hit_count": 0, "total_evidence": 0,
                                          "first_hit_rank": 0}

                        
            is_correct = normalize_answer(pred) == normalize_answer(gold)

            result_item = {
                "id": _id,
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred,
                "is_correct": is_correct,
                "metrics": {
                    "num_chunks": total_chunks,
                    "num_directories": 0,
                    "top_d": 0,
                    "top_k": top_k,
                    "indexing_time_s": timing["total_time"],
                    "indexing_time_breakdown": {
                        "total_time": timing["total_time"],
                        "chunking_time": timing.get("chunking_time", 0),
                        "embedding_time_total": timing.get("embedding_time", 0),
                        "openie_time": timing.get("openie_time", 0),
                        "entity_fact_embedding_time": timing.get("entity_fact_embedding_time", 0),
                        "graph_construction_time": timing.get("graph_construction_time", 0),
                        "bm25_time_total": 0.0,
                        "entity_time_total": 0.0,
                    },
                                                           
                    "indexing_token_cost": metrics.get("indexing_token_cost", {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "llm_call_count": 0,
                    }),
                                                       
                    "retrieval_token_cost": metrics.get("retrieval_token_cost", {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "llm_call_count": 0,
                    }),
                    "retrieval_time_s": metrics["retrieval_time_s"],
                    "e2e_latency_s": metrics["e2e_latency_s"],
                    "token_cost": metrics["token_cost"],
                    "llm_call_count": metrics["llm_call_count"],
                    "iterations": metrics["iterations"],
                    "index_size_mb": round(index_size, 3),
                    "recall_at_k": recall_m["recall_at_k"],
                    "mrr": recall_m["mrr"],
                    "hit_count": recall_m["hit_count"],
                    "total_evidence": recall_m["total_evidence"],
                    "first_hit_rank": recall_m["first_hit_rank"],
                },
                "graph_stats": stats,
                "iteration_details": metrics["iteration_details"],
            }
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

            idx_tok = metrics.get("indexing_token_cost", {})
            print(f"A: {pred} | {'✅' if is_correct else '❌'} Gold: {gold} | "
                  f"E2E={metrics['e2e_latency_s']:.3f}s | "
                  f"Retrieval={metrics['retrieval_time_s']:.3f}s | "
                  f"LLMCalls={metrics['llm_call_count']} | "
                  f"Tokens={metrics['token_cost']['total_tokens']} | "
                  f"IndexTokens={idx_tok.get('total_tokens', 0)} | "
                  f"Recall@K={recall_m['recall_at_k']:.2f} | "
                  f"MRR={recall_m['mrr']:.4f}")
            time.sleep(0.5)

        except Exception as e:
            print(f"⚠️ 样本跳过: {str(e)[:80]}")
            import traceback
            traceback.print_exc()
            time.sleep(1)
            continue

    print("\n✅ 评估完成!")


if __name__ == "__main__":
    WIKI_PATH = "/root/dir-sem-rag/longbench/data_recall/hotpotqa_recall_clean.jsonl"
    OUTPUT_PATH = "./hipporag2_hotpotqa_results.jsonl"
            
    OPENIE_CACHE_PATH = "./hipporag2_openie_cache.json"

    MAX_SAMPLES = None
    CHUNK_SIZE = 100                      
    OVERLAP = 32                          
    TOP_K = 5                                                     

                   
    LINKING_TOP_K = 10                                            
    SYNONYMY_EDGE_TOPK = 10                     
    SYNONYMY_EDGE_SIM_THRESHOLD = 0.8            
    PASSAGE_NODE_WEIGHT = 0.05                       
    DAMPING = 0.5                                     

    local_llm = LocalLLMClient(model_path=MODEL_PATH)

    samples = []
    with open(WIKI_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            samples.append({
                "_id": raw.get("_id", str(hash(raw.get("question", "")))),
                "question": raw.get("question", raw.get("input", "")),
                "answer": raw["answers"][0] if raw.get("answers") else "",
                "passages": raw.get("passages", []),
                "supporting_facts": raw.get("supporting_facts", []),
            })

    print(f"📚 共加载 {len(samples)} 个 2wiki 样本")

    run_evaluation(
        samples=samples,
        local_llm=local_llm,
        output_path=OUTPUT_PATH,
        max_samples=MAX_SAMPLES,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        top_k=TOP_K,
        linking_top_k=LINKING_TOP_K,
        synonymy_edge_topk=SYNONYMY_EDGE_TOPK,
        synonymy_edge_sim_threshold=SYNONYMY_EDGE_SIM_THRESHOLD,
        passage_node_weight=PASSAGE_NODE_WEIGHT,
        damping=DAMPING,
        openie_cache_path=OPENIE_CACHE_PATH,
    )
