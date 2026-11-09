                       

import os
import sys
import json
import re
import time
import string
import gc
import pickle
import hashlib
from typing import List, Dict, Optional, Tuple, Set, Any

import numpy as np
import networkx as nx
from community import community_louvain

                                                                             
MODEL_PATH = "/root/dir-sem-rag/Qwen2.5-7B-Instruct"

sys.path.insert(0, "/root/dir-sem-rag/gte-Qwen2-1.5B-instruct")
sys.path.insert(1, "/root/dir-sem-rag/longbench/test/dir_rag_section_chunk_claude")
from embedding import EmbeddingModel
from llm import LocalLLMClient

           
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
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


def safe_json_parse(raw: str) -> Any:
    raw = raw.strip()

                  
    m = re.search(r"```(?:json)?\n?([\s\S]*?)\n?```", raw)
    if m:
        raw = m.group(1).strip()

          
    try:
        return json.loads(raw)
    except Exception:
        pass

                  
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

                  
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    return None


QA_PROMPT = """Answer the question using the provided context or based on your own knowledge.

Context:

{context}

Question: {question}

Instructions:

1. Give the shortest correct answer phrase.

2. If you do NOT know the answer or the context does NOT contain enough information to answer the question, respond with exactly: "I don't know". Please do NOT randomly guess the answer.

3. Output ONLY the answer, nothing else (no explanation, no JSON, no prefix)."""


class Chunk:
    def __init__(self, chunk_id: int, text: str, article_title: str,
                 index_in_article: int, sent_ids: Set[int]):
        self.chunk_id = chunk_id
        self.text = text
        self.article_title = article_title
        self.index_in_article = index_in_article
        self.sent_ids = sent_ids


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

                                      
    def destroy_llm(self):
        if self.llm_client is not None and hasattr(self.llm_client, "destroy"):
            self.llm_client.destroy()

    def ensure_llm_loaded(self):
        if self.llm_client is not None and hasattr(self.llm_client, "ensure_loaded"):
            self.llm_client.ensure_loaded()

    def is_llm_loaded(self) -> bool:
        if self.llm_client is None:
            return False
        if hasattr(self.llm_client, "is_loaded"):
            return self.llm_client.is_loaded()
        return True


class GraphElementCache:

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
                                                                     
        self._cache: Dict[str, Dict[str, list]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.loads(f.read())
                print(f"🗂️  图谱元素缓存已加载: {len(self._cache)} 条记录")
            except Exception as e:
                print(f"⚠️ 图谱元素缓存加载失败: {str(e)[:50]}")
                self._cache = {}

    def _save(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(self._cache, ensure_ascii=False))
        except Exception as e:
            print(f"⚠️ 图谱元素缓存保存失败: {str(e)[:50]}")

    @staticmethod
    def make_key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[Dict[str, list]]:
        key = self.make_key(text)
        return self._cache.get(key)

    def set(self, text: str, entities: List[Dict], relations: List[Dict]) -> None:
        key = self.make_key(text)
                                            
        clean_entities = [
            {k: v for k, v in ent.items() if k != "chunk_idx"}
            for ent in entities
        ]
        self._cache[key] = {
            "entities": clean_entities,
            "relations": relations,
        }

    def flush(self) -> None:
        self._save()

    def __len__(self) -> int:
        return len(self._cache)


class CommunitySummaryCache:

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
                                                                    
        self._cache: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.loads(f.read())
                print(f"🏘️  社区摘要缓存已加载: {len(self._cache)} 条图签名")
            except Exception as e:
                print(f"⚠️ 社区摘要缓存加载失败: {str(e)[:50]}")
                self._cache = {}

    def _save(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(self._cache, ensure_ascii=False))
        except Exception as e:
            print(f"⚠️ 社区摘要缓存保存失败: {str(e)[:50]}")

    @staticmethod
    def compute_graph_signature(chunk_text_hashes: List[str]) -> str:
        sorted_hashes = sorted(chunk_text_hashes)
        combined = "|".join(sorted_hashes)
        return hashlib.md5(combined.encode("utf-8")).hexdigest()

    def get(self, graph_signature: str) -> Optional[Dict[int, str]]:
        raw = self._cache.get(graph_signature)
        if raw is None:
            return None
                              
        return {int(k): v for k, v in raw.items()}

    def set(self, graph_signature: str, summaries: Dict[int, str]) -> None:
                               
        self._cache[graph_signature] = {
            str(k): v for k, v in summaries.items()
        }

    def flush(self) -> None:
        self._save()

    def __len__(self) -> int:
        return len(self._cache)


class GraphRAG:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_client: TrackedLLM,
        chunk_size: int = 100,
        overlap: int = 32,
        top_k: int = 5,
        hop: int = 1,
        graph_element_cache: Optional['GraphElementCache'] = None,
        community_summary_cache: Optional['CommunitySummaryCache'] = None,
    ):
        self.embedding_model = embedding_model
        self.llm = llm_client
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.hop = hop
        self.graph_element_cache = graph_element_cache
        self.community_summary_cache = community_summary_cache

               
        self.chunks: List[Chunk] = []
        self.chunk_id_to_idx: Dict[int, int] = {}
        self._chunk_id_counter = 0

                             
        self._embeddings: List[np.ndarray] = []

              
        self.graph: nx.Graph = nx.Graph()
        self.entity_to_chunks: Dict[str, List[int]] = {}                                   
        self.communities: Dict[int, List[str]] = {}
        self.community_summaries: Dict[int, str] = {}

                            
        self._entity_names: List[str] = []
        self._entity_embeddings: List[np.ndarray] = []
        self._entity_name_to_idx: Dict[str, int] = {}

                              
    def reset(self):
        self.chunks.clear()
        self.chunk_id_to_idx.clear()
        self._embeddings.clear()
        self._chunk_id_counter = 0
        self.graph = nx.Graph()
        self.entity_to_chunks.clear()
        self.communities.clear()
        self.community_summaries.clear()
        self._entity_names.clear()
        self._entity_embeddings.clear()
        self._entity_name_to_idx.clear()
        gc.collect()

                                  
    def build_index(self, passages: List[dict], extract_only: bool = False) -> dict:
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
            timing["graph_extraction_time"] = 0.0
            timing["community_time"] = 0.0
            timing["total_time"] = round(time.time() - t0, 4)
            return timing

                                                                        
        if extract_only:
                                                                     
            t1 = time.time()
            self._build_graph(extract_only=True)
            timing["graph_extraction_time"] = round(time.time() - t1, 4)

                                   
            t1 = time.time()
            self._detect_communities()
            self._summarize_communities()
            timing["community_time"] = round(time.time() - t1, 4)

            timing["embedding_time"] = 0.0
            timing["total_time"] = round(time.time() - t0, 4)
            return timing

                                        
        t1 = time.time()
        try:
            texts = [c.text for c in self.chunks]
            embs = self.embedding_model.encode(texts)
            self._embeddings = [np.array(e, dtype=np.float32) for e in embs]
        except Exception as e:
            print(f"⚠️ Embedding 失败: {str(e)[:50]}")
        timing["embedding_time"] = round(time.time() - t1, 4)

                                    
        t1 = time.time()
        self._build_graph()
        timing["graph_extraction_time"] = round(time.time() - t1, 4)

                           
        t1 = time.time()
        self._detect_communities()
        self._summarize_communities()
        timing["community_time"] = round(time.time() - t1, 4)

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

                                  
    def _build_graph(self, extract_only: bool = False):
        print(f"    🕸️  从 {len(self.chunks)} 个文本块提取图谱元素 ...")
        all_entities: List[Dict] = []
        all_relations: List[Dict] = []
        cache_hit_count = 0
        chunk_text_hashes: List[str] = []                     

        for i, chunk in enumerate(self.chunks):
            chunk_text_hashes.append(GraphElementCache.make_key(chunk.text))

                                  
            cached = None
            if self.graph_element_cache is not None:
                cached = self.graph_element_cache.get(chunk.text)
                if cached is not None:
                    cache_hit_count += 1

            if cached is not None:
                entities = cached.get("entities", [])
                relations = cached.get("relations", [])
                                         
                for ent in entities:
                    ent["chunk_idx"] = i
            else:
                                         
                entities, relations = self._extract_graph_elements(
                    chunk.text, chunk.article_title
                )
                for ent in entities:
                    ent["chunk_idx"] = i

                                      
                if self.graph_element_cache is not None:
                    self.graph_element_cache.set(
                        chunk.text, entities, relations
                    )

            all_entities.extend(entities)
            all_relations.extend(relations)

                            
        if self.graph_element_cache is not None and cache_hit_count < len(self.chunks):
            self.graph_element_cache.flush()

        if cache_hit_count > 0:
            print(f"    🗂️  图谱元素缓存命中: {cache_hit_count}/{len(self.chunks)} 个文本块")

        print(f"    🧩 构建知识图谱 ...")
        self._build_networkx_graph(all_entities, all_relations)

                                
        if not extract_only:
            self._build_entity_embeddings(all_entities)

        print(f"    ✅ 图谱: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

                            
        self._chunk_text_hashes = chunk_text_hashes

    def _build_entity_embeddings(self, all_entities: List[Dict]):
        unique_names = sorted({ent["name"] for ent in all_entities})
        if not unique_names:
            return
        try:
            embs = self.embedding_model.encode(unique_names)
            self._entity_names = unique_names
            self._entity_embeddings = [np.array(e, dtype=np.float32) for e in embs]
            self._entity_name_to_idx = {name: i for i, name in enumerate(unique_names)}
        except Exception as e:
            print(f"⚠️ 实体向量构建失败: {str(e)[:50]}")

    def _extract_graph_elements(self, text: str, title: str) -> Tuple[List[Dict], List[Dict]]:
        prompt = f"""You are an information-extraction assistant.
Given the TEXT below, extract:
1. Named entities (people, organizations, locations, works, events, concepts).
2. Relationships between those entities.

TEXT: {text}

Respond ONLY with a single valid JSON object — no markdown, no explanation.
Schema:
{{
  "entities": [
    {{"name": "<entity name>", "type": "<PERSON|ORG|LOC|WORK|EVENT|CONCEPT>", "description": "<one-sentence description>"}}
  ],
  "relations": [
    {{"head": "<entity name>", "relation": "<verb phrase>", "tail": "<entity name>", "weight": <0.0-1.0>}}
  ]
}}
JSON:"""
        raw = self.llm.chat(prompt, stage="indexing").strip()
        parsed = safe_json_parse(raw)

        entities: List[Dict] = []
        relations: List[Dict] = []

        if isinstance(parsed, dict):
            raw_ents = parsed.get("entities", [])
            raw_rels = parsed.get("relations", [])

            for e in raw_ents:
                if isinstance(e, dict) and e.get("name"):
                    entities.append({
                        "name": str(e["name"]).strip(),
                        "type": str(e.get("type", "UNKNOWN")),
                        "description": str(e.get("description", "")),
                        "source_title": title,
                    })

            entity_names = {e["name"] for e in entities}
            for r in raw_rels:
                if (isinstance(r, dict) and r.get("head") in entity_names and
                        r.get("tail") in entity_names and r.get("relation")):
                    relations.append({
                        "head": str(r["head"]).strip(),
                        "relation": str(r["relation"]).strip(),
                        "tail": str(r["tail"]).strip(),
                        "weight": float(r.get("weight", 0.5)),
                    })

        return entities, relations

    def _build_networkx_graph(self, entities: List[Dict], relations: List[Dict]):
        self.entity_to_chunks = {}
        for ent in entities:
            name = ent["name"]
            idx = ent.get("chunk_idx", -1)

            if not self.graph.has_node(name):
                self.graph.add_node(
                    name, type=ent["type"],
                    description=ent["description"],
                    source_titles=set(),
                )
            self.graph.nodes[name]["source_titles"].add(ent["source_title"])

            if idx >= 0:
                self.entity_to_chunks.setdefault(name, [])
                if idx not in self.entity_to_chunks[name]:
                    self.entity_to_chunks[name].append(idx)

        for rel in relations:
            h, t = rel["head"], rel["tail"]
            if self.graph.has_node(h) and self.graph.has_node(t):
                if self.graph.has_edge(h, t):
                    existing = self.graph[h][t].get("relations", [])
                    existing.append(rel["relation"])
                    self.graph[h][t]["relations"] = existing
                    self.graph[h][t]["weight"] = max(
                        self.graph[h][t].get("weight", 0.5), rel["weight"]
                    )
                else:
                    self.graph.add_edge(
                        h, t, relations=[rel["relation"]], weight=rel["weight"]
                    )

    def _detect_communities(self):
        if self.graph.number_of_nodes() == 0:
            return
        partition = community_louvain.best_partition(
            self.graph, weight="weight", random_state=42
        )
        self.communities = {}
        for node, cid in partition.items():
            self.communities.setdefault(cid, [])
            self.communities[cid].append(node)
        nx.set_node_attributes(self.graph, partition, "community")

    def _summarize_communities(self):
                          
        if self.community_summary_cache is not None and hasattr(self, '_chunk_text_hashes'):
            graph_sig = CommunitySummaryCache.compute_graph_signature(self._chunk_text_hashes)
            cached = self.community_summary_cache.get(graph_sig)
            if cached is not None:
                self.community_summaries = cached
                print(f"    🏘️  社区摘要缓存命中: {len(cached)} 个社区")
                return
        else:
            graph_sig = None

        for cid, members in self.communities.items():
            internal_edges = [
                (u, v, self.graph[u][v])
                for u, v in self.graph.edges(members)
                if u in set(members) and v in set(members)
            ]

            node_lines = []
            for m in members[:15]:
                desc = self.graph.nodes[m].get("description", "")
                ntype = self.graph.nodes[m].get("type", "")
                node_lines.append(f"  - {m} ({ntype}): {desc}")

            edge_lines = []
            for u, v, data in internal_edges[:20]:
                rels = ", ".join(data.get("relations", ["related"]))
                edge_lines.append(f"  - {u} --[{rels}]--> {v}")

            community_text = (
                f"Entities ({len(members)} total, showing up to 15):\n"
                + "\n".join(node_lines)
                + f"\n\nRelationships (showing up to 20):\n"
                + ("\n".join(edge_lines) if edge_lines else "")
            )

            prompt = f"""You are a knowledge-graph analyst.
Below is a community of related entities extracted from a document collection.
Write a concise, self-contained summary (3-5 sentences) describing:
- Who/what the key entities are
- How they are related
- What topic or theme this community represents

Community data:
{community_text}

Summary:"""
            summary = self.llm.chat(prompt, stage="indexing").strip()
            self.community_summaries[cid] = summary if summary != "ERROR" else community_text

                          
        if self.community_summary_cache is not None and graph_sig is not None:
            self.community_summary_cache.set(graph_sig, self.community_summaries)
            self.community_summary_cache.flush()

                                   
    def answer_question(self, question: str) -> Tuple[str, dict]:
        t_start = time.time()
        self.llm.reset()

                                  
        t_ret = time.time()
        try:
            q_emb = self.embedding_model.encode([question])[0]
        except Exception as e:
            print(f"⚠️ 查询 embedding 失败: {str(e)[:40]}")
            q_emb = np.zeros(self._embeddings[0].shape if self._embeddings else 768,
                              dtype=np.float32)

        seed_entities = self._semantic_entity_search(q_emb, n=10)

                               
        subgraph_nodes = self._expand_subgraph(seed_entities, self.hop)
        candidate_entities = list(subgraph_nodes | set(seed_entities))

                                
        entity_scores = self._rank_entities(q_emb, candidate_entities)

                                 
        candidate_chunk_indices: Set[int] = set()
                   
        seed_chunk_ids = self._vector_top_k(question, self.top_k)
        for cid in seed_chunk_ids:
            if cid in self.chunk_id_to_idx:
                candidate_chunk_indices.add(self.chunk_id_to_idx[cid])

                      
        top_entities = [name for name, _ in entity_scores[:20]]
        for ent_name in top_entities:
            for idx in self.entity_to_chunks.get(ent_name, []):
                candidate_chunk_indices.add(idx)

                                 
        ranked_chunks = self._rank_chunks(
            q_emb, candidate_chunk_indices, entity_scores
        )

                                   
        final_chunks = ranked_chunks[:self.top_k]
        final_chunk_ids = [c.chunk_id for c in final_chunks]

              
        relevant_cids = set()
        for ent_name, _ in entity_scores[:10]:
            if self.graph.has_node(ent_name):
                cid = self.graph.nodes[ent_name].get("community")
                if cid is not None:
                    relevant_cids.add(cid)

        retrieval_time = time.time() - t_ret

                                        
        llm_result = self._generate_answer(question, final_chunks, relevant_cids)

        pred = llm_result.get("answer", "I don't know")
        if not llm_result.get("isSufficient", False):
            pred = "I don't know"

        e2e = time.time() - t_start

                                                          
        iteration_details = [{
            "iteration": 1,
            "retrieved_dir_ids": [],
            "retrieved_chunk_ids": seed_chunk_ids,
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
            "graph_info": {
                "seed_entities": seed_entities,
                "subgraph_nodes": list(subgraph_nodes),
                "top_entities": top_entities,
                "community_ids": list(relevant_cids),
            },
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
            "llm_call_count": self.llm.call_count,
            "e2e_latency_s": round(e2e, 4),
            "retrieval_time_s": round(retrieval_time, 4),
            "iterations": 1,
            "iteration_details": iteration_details,
        }
        return pred, metrics

    def _semantic_entity_search(self, q_emb: np.ndarray, n: int = 10) -> List[str]:
        if not self._entity_embeddings:
            return []
        scored = []
        for i, ent_name in enumerate(self._entity_names):
            sim = cosine_similarity(q_emb, self._entity_embeddings[i])
            scored.append((ent_name, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored[:n]]

    def _rank_entities(self, q_emb: np.ndarray,
                       candidate_entities: List[str]) -> List[Tuple[str, float]]:
        if not candidate_entities:
            return []

                       
        max_degree = max(
            (self.graph.degree(n) for n in candidate_entities
             if self.graph.has_node(n)),
            default=1
        )
        if max_degree == 0:
            max_degree = 1

              
        community_sizes = {cid: len(members) for cid, members in self.communities.items()}
        max_comm_size = max(community_sizes.values(), default=1)
        if max_comm_size == 0:
            max_comm_size = 1

        scored = []
        for name in candidate_entities:
            if not self.graph.has_node(name):
                continue
                   
            idx = self._entity_name_to_idx.get(name)
            if idx is not None and idx < len(self._entity_embeddings):
                sim = cosine_similarity(q_emb, self._entity_embeddings[idx])
            else:
                sim = 0.0

                               
            degree = self.graph.degree(name)
            degree_norm = degree / max_degree

                    
            cid = self.graph.nodes[name].get("community")
            comm_factor = (community_sizes.get(cid, 1) / max_comm_size) if cid is not None else 0.5

                  
            score = sim * 0.5 + degree_norm * 0.3 + comm_factor * 0.2
            scored.append((name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _rank_chunks(self, q_emb: np.ndarray, candidate_indices: Set[int],
                     entity_scores: List[Tuple[str, float]]) -> List[Chunk]:
                       
        ent_score_map = {name: score for name, score in entity_scores}

                           
        chunk_entity_scores: Dict[int, float] = {}
        for ent_name, score in entity_scores:
            for idx in self.entity_to_chunks.get(ent_name, []):
                chunk_entity_scores[idx] = chunk_entity_scores.get(idx, 0) + score

                 
        max_ent_score = max(chunk_entity_scores.values(), default=1)
        if max_ent_score == 0:
            max_ent_score = 1

        scored_chunks = []
        for idx in candidate_indices:
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
                         
            ent_score = chunk_entity_scores.get(idx, 0) / max_ent_score
                             
            if idx < len(self._embeddings):
                chunk_sim = cosine_similarity(q_emb, self._embeddings[idx])
            else:
                chunk_sim = 0.0
                  
            total_score = ent_score * 0.6 + chunk_sim * 0.4
            scored_chunks.append((chunk, total_score, idx))

                 
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored_chunks]

    def _vector_top_k(self, query: str, k: int) -> List[int]:
        if not self._embeddings:
            return []
        try:
            q_emb = self.embedding_model.encode([query])[0]
        except Exception as e:
            print(f"⚠️ 查询 embedding 失败: {str(e)[:40]}")
            return []

        scored = [
            (self.chunks[i].chunk_id, cosine_similarity(q_emb, self._embeddings[i]))
            for i in range(len(self._embeddings))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scored[:k]]

    def _link_entities(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = []
        for node in self.graph.nodes():
            if node.lower() in text_lower:
                found.append(node)
        return found

    def _expand_subgraph(self, seed_entities: List[str], hops: int) -> set:
        visited = set()
        frontier = set(e for e in seed_entities if self.graph.has_node(e))
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                if node not in visited:
                    visited.add(node)
                    next_frontier.update(self.graph.neighbors(node))
            frontier = next_frontier - visited
        visited.update(frontier)
        return visited

    def _generate_answer(self, question: str, context_chunks: List[Chunk],
                         relevant_cids: Set[int]) -> dict:
              
        community_context = "\n\n".join(
            f"[Community {cid} Summary]: {self.community_summaries.get(cid, '')}"
            for cid in relevant_cids
        )

                                        
        chunk_parts: List[str] = []
        used_chunk_info: List[dict] = []
        for c in context_chunks:
            chunk_parts.append(f"[{c.article_title}] {c.text}")
            used_chunk_info.append({
                "chunk_id": c.chunk_id,
                "article_title": c.article_title,
                "index_in_article": c.index_in_article,
                "text_preview": c.text[:200],
            })
        chunk_context = "\n\n".join(chunk_parts) if chunk_parts else "(no context)"

                 
        full_context_parts: List[str] = []
        if community_context:
            full_context_parts.append(f"=== Community Knowledge ===\n{community_context}")
        full_context_parts.append(f"=== Source Text ===\n{chunk_context}")
        context = "\n\n".join(full_context_parts)

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

                                
    def get_chunk_count(self) -> int:
        return len(self.chunks)

    def get_index_size_mb(self) -> float:
        total_bytes = sum(e.nbytes for e in self._embeddings)
                           
        index_payload = {
            "graph": nx.node_link_data(self.graph),
            "entity_to_chunks": self.entity_to_chunks,
            "communities": self.communities,
            "community_summaries": self.community_summaries,
        }
        total_bytes += len(pickle.dumps(index_payload))
        return total_bytes / (1024 * 1024)

    def get_graph_stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "communities": len(self.communities),
        }


def run_evaluation(
    samples: List[dict],
    local_llm: TrackedLLM,
    output_path: str,
    max_samples: Optional[int] = None,
    chunk_size: int = 100,
    overlap: int = 32,
    top_k: int = 5,
    hop: int = 1,
    graph_element_cache_path: str = "",
    community_summary_cache_path: str = "",
                                                             
    extract_only: bool = False,
    extract_only_stats_path: str = "",
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

                                     
    graph_element_cache = None
    if graph_element_cache_path:
        graph_element_cache = GraphElementCache(graph_element_cache_path)
        print(f"🗂️  图谱元素缓存已启用: {graph_element_cache_path}")

                              
    community_summary_cache = None
    if community_summary_cache_path:
        community_summary_cache = CommunitySummaryCache(community_summary_cache_path)
        print(f"🏘️  社区摘要缓存已启用: {community_summary_cache_path}")

    graph_rag = GraphRAG(
        embedding_model=embedding_model,
        llm_client=local_llm,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        hop=hop,
        graph_element_cache=graph_element_cache,
        community_summary_cache=community_summary_cache,
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
    print(f"🚀 HotpotQA Graph RAG | 向量检索种子 + 知识图谱扩展 | 剩余: {total}")
    print(f"✅ chunk_size={chunk_size}  top_k={top_k}  hop={hop}")
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
            graph_rag.reset()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

                            
            timing = graph_rag.build_index(passages, extract_only=extract_only)
            total_chunks = graph_rag.get_chunk_count()
            index_size = graph_rag.get_index_size_mb()
            graph_stats = graph_rag.get_graph_stats()

            print(f"📦 文本块: {total_chunks}, "
                  f"IndexTime={timing['total_time']:.3f}s, "
                  f"IndexSize={index_size:.2f}MB")
            print(f"   🕸️  图谱: {graph_stats['nodes']} nodes, "
                  f"{graph_stats['edges']} edges, "
                  f"{graph_stats['communities']} communities")

                                                                 
            if extract_only:
                stats_item = {
                    "id": _id,
                    "question": question,
                    "metrics": {
                        "num_chunks": total_chunks,
                        "indexing_time_s": timing.get("total_time", 0),
                        "indexing_time_breakdown": timing,
                        "indexing_token_cost": {
                            "input_tokens": graph_rag.llm.indexing_input_tokens,
                            "output_tokens": graph_rag.llm.indexing_output_tokens,
                            "total_tokens": graph_rag.llm.indexing_total_tokens,
                            "llm_call_count": graph_rag.llm.indexing_call_count,
                        },
                    },
                }
                stats_path = extract_only_stats_path or output_path.replace(".jsonl", "_extract_only.jsonl")
                with open(stats_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(stats_item, ensure_ascii=False) + "\n")
                print(f"    📝 extract_only: LLM 提取完成, "
                      f"耗时={timing.get('total_time', 0):.3f}s, "
                      f"Token={graph_rag.llm.indexing_total_tokens}")
                continue

                           
            pred, metrics = graph_rag.answer_question(question)

                                  
            retrieved = metrics["iteration_details"][0]["expanded_chunk_ids"]
            recall_m = graph_rag.compute_recall_mrr(supporting_facts, retrieved)\
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
                        "graph_extraction_time": timing.get("graph_extraction_time", 0),
                        "community_time": timing.get("community_time", 0),
                        "bm25_time_total": 0.0,
                        "entity_time_total": 0.0,
                    },
                    "retrieval_time_s": metrics["retrieval_time_s"],
                    "e2e_latency_s": metrics["e2e_latency_s"],
                    "token_cost": metrics["token_cost"],
                    "retrieval_token_cost": metrics["retrieval_token_cost"],
                    "indexing_token_cost": {
                        "input_tokens": graph_rag.llm.indexing_input_tokens,
                        "output_tokens": graph_rag.llm.indexing_output_tokens,
                        "total_tokens": graph_rag.llm.indexing_total_tokens,
                        "llm_call_count": graph_rag.llm.indexing_call_count,
                    },
                    "llm_call_count": metrics["llm_call_count"],
                    "iterations": metrics["iterations"],
                    "index_size_mb": round(index_size, 3),
                    "recall_at_k": recall_m["recall_at_k"],
                    "mrr": recall_m["mrr"],
                    "hit_count": recall_m["hit_count"],
                    "total_evidence": recall_m["total_evidence"],
                    "first_hit_rank": recall_m["first_hit_rank"],
                },
                "graph_stats": graph_stats,
                "iteration_details": metrics["iteration_details"],
            }
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

            print(f"A: {pred} | {'✅' if is_correct else '❌'} Gold: {gold} | "
                  f"E2E={metrics['e2e_latency_s']:.3f}s | "
                  f"Retrieval={metrics['retrieval_time_s']:.3f}s | "
                  f"LLMCalls={metrics['llm_call_count']} | "
                  f"Tokens={metrics['token_cost']['total_tokens']} | "
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
    HOTPOTQA_PATH = "/root/dir-sem-rag/longbench/data_recall/hotpotqa_recall_clean.jsonl"
    OUTPUT_PATH = "./graphrag_trae_hotpotqa_results.jsonl"
                                           
    GRAPH_ELEMENT_CACHE_PATH = "./graphrag_element_cache.json"
                                         
    COMMUNITY_SUMMARY_CACHE_PATH = "./graphrag_community_cache.json"

    MAX_SAMPLES = None
    CHUNK_SIZE = 100
    OVERLAP = 32
    TOP_K = 5
    HOP = 1                    

                                                   
    EXTRACT_ONLY = False
    EXTRACT_ONLY_STATS_PATH = "./graphrag_extract_only_stats.jsonl"

    local_llm = LocalLLMClient(model_path=MODEL_PATH)

    samples = []
    with open(HOTPOTQA_PATH, encoding="utf-8") as f:
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

    print(f"📚 共加载 {len(samples)} 个 HotpotQA 样本")

    run_evaluation(
        samples=samples,
        local_llm=local_llm,
        output_path=OUTPUT_PATH,
        max_samples=MAX_SAMPLES,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        top_k=TOP_K,
        hop=HOP,
        graph_element_cache_path=GRAPH_ELEMENT_CACHE_PATH,
        community_summary_cache_path=COMMUNITY_SUMMARY_CACHE_PATH,
        extract_only=EXTRACT_ONLY,
        extract_only_stats_path=EXTRACT_ONLY_STATS_PATH,
    )
