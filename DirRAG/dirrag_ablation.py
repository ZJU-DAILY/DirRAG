                     
                       
import os
import sys
import json
import re
import time
import string
import gc
import math
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from collections import defaultdict, Counter
from dataclasses import dataclass, field

import numpy as np

                                                                              
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


STOPWORDS: Set[str] = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
    'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just', 'as', 'if', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'any', 'also', 'because',
    'while', 'until', 'them', 'his', 'her', 'its', 'our', 'your', 'their',
    'me', 'him', 'us', 'am', 'my', 'mine', 'yours', 'hers', 'ours',
    'theirs', 'whom', 'whose', 'whatever', 'whichever', 'whoever',
    'whomever', 'whenever', 'wherever', 'however', 's', 't', 'd', 'll',
    'm', 'o', 're', 've', 'y',
}


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 32) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def tokenize(text: str) -> List[str]:
    text = text.lower()
                        
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
                  
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    return " ".join(s.split())


def rrf_score(rank_lists: List[List[Any]], k: int = 60) -> Dict[Any, float]:
    scores: Dict[Any, float] = defaultdict(float)
    for rank_list in rank_lists:
        for rank, item_id in enumerate(rank_list, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return dict(scores)


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
                               
            result = _extract_json_fields(json_str)
            if result is not None:
                return result

                                
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

                               
    result = _extract_json_fields(text)
    if result is not None:
        return result

    return None


def _extract_json_fields(text: str) -> Optional[dict]:
    result: dict = {}

                                    
    m = re.search(r'"isSufficient"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        result["isSufficient"] = m.group(1).lower() == "true"

                                
    m = re.search(r'"answer"\s*:\s*"(.*?)"\s*(?:,|\})', text, re.DOTALL)
    if m:
        result["answer"] = m.group(1).strip()

                                   
    m = re.search(
        r'"reasoning"\s*:\s*"(.*?)"\s*,\s*"(?:answer|new_keywords|new_entities|isSufficient)"',
        text, re.DOTALL
    )
    if not m:
                               
        m = re.search(r'"reasoning"\s*:\s*"(.*?)"\s*\}', text, re.DOTALL)
    if m:
        result["reasoning"] = m.group(1).strip()

                                  
    m = re.search(r'"new_keywords"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if m:
        arr_str = m.group(1)
                    
        keywords = re.findall(r'"([^"]*)"', arr_str)
        result["new_keywords"] = [k.strip() for k in keywords if k.strip()]
    else:
        result["new_keywords"] = []

                                  
    m = re.search(r'"new_entities"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if m:
        arr_str = m.group(1)
        entities = re.findall(r'"([^"]*)"', arr_str)
        result["new_entities"] = [e.strip() for e in entities if e.strip()]
    else:
        result["new_entities"] = []

                               
    if "isSufficient" in result:
        return result
    return None


@dataclass
class Chunk:
    chunk_id: int                         
    text: str                       
    article_title: str                  
    directory_path: str                      
    index_in_article: int                      
    sent_ids: set = None                                 


class DirectoryNode:

    def __init__(self, full_path: str, dir_id: int, parent: Optional['DirectoryNode'] = None):
        self.full_path: str = full_path                       
        self.dir_id: int = dir_id                             
        self.parent: Optional[DirectoryNode] = parent       
        self.children: Dict[str, DirectoryNode] = {}                         
        self.chunk_ids: Set[int] = set()                             
        self.all_chunk_ids: Set[int] = set()                               
        self.access_count: int = 0                                        

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def update_all_chunk_ids(self):
        self.all_chunk_ids = set(self.chunk_ids)
        for child in self.children.values():
            child.update_all_chunk_ids()
            self.all_chunk_ids |= child.all_chunk_ids


class BM25Index:

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1                                       
        self.b = b                                            
        self.doc_ids: List[Any] = []                           
        self.docs: List[List[str]] = []              
        self.doc_len: List[int] = []                      
        self.avgdl: float = 0.0                      
        self.df: Counter = Counter()                               
        self.idf: Dict[str, float] = {}              
        self.inverted: Dict[str, List[Tuple[int, int]]] = defaultdict(list)                                 
        self.N: int = 0                             
        self._idf_computed: bool = False                 

    def add(self, doc_id: Any, tokens: List[str]) -> None:
        doc_idx = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self.docs.append(tokens)
        self.doc_len.append(len(tokens))

                
        tf = Counter(tokens)
        for term, freq in tf.items():
            self.df[term] += 1
            self.inverted[term].append((doc_idx, freq))

        self.N += 1
        self._idf_computed = False                   

    def _compute_idf(self) -> None:
        for term, df in self.df.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        self._idf_computed = True

    def search(self, query_tokens: List[str], top_k: int = 10,
               filter_ids: Optional[Set[Any]] = None) -> List[Tuple[Any, float]]:
        if not self.doc_ids or not query_tokens:
            return []
        if not self._idf_computed:
            self._compute_idf()

        scores: Dict[Any, float] = defaultdict(float)
        filter_set = filter_ids if filter_ids is not None else None

        for term in query_tokens:
            if term not in self.inverted:
                continue
            for doc_idx, tf in self.inverted[term]:
                doc_id = self.doc_ids[doc_idx]
                                        
                if filter_set is not None and doc_id not in filter_set:
                    continue
                dl = self.doc_len[doc_idx]
                idf_val = self.idf.get(term, 0.0)
                           
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-8))
                scores[doc_id] += idf_val * numerator / denominator

                 
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if top_k > 0:
            sorted_results = sorted_results[:top_k]
        return sorted_results

    def clear(self) -> None:
        self.__init__(self.k1, self.b)

    def __len__(self) -> int:
        return self.N


class NEREntityExtractor:

    def __init__(self, model_name: str, batch_size: int = 64,
                 device=None):
        from transformers import pipeline
        if device is None:
            try:
                import torch as _torch
                device = 0 if _torch.cuda.is_available() else -1
            except Exception:
                device = -1
        self.pipe = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=device,
        )
        self.batch_size = batch_size
        print(f"✅ NER 模型已加载: {model_name} (device={'GPU' if device >= 0 else 'CPU'}, "
              f"batch_size={batch_size})")

    @staticmethod
    def _clean(ent_dicts):
        entities = []
        seen = set()
        for ed in ent_dicts:
            word = str(ed.get("word", "")).strip()
            if len(word) <= 1:
                continue
            key = word.lower()
            if key not in seen:
                seen.add(key)
                entities.append(word)
        return entities

    def extract(self, text: str):
        return self.extract_batch([text])[0]

    def extract_batch(self, texts):
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                outs = self.pipe(batch)
            except Exception as e:
                print(f"⚠️ NER 批量推理失败(batch起始={i}): {str(e)[:60]}")
                results.extend([[] for _ in batch])
                continue
            for ent_dicts in outs:
                results.append(self._clean(ent_dicts))
        return results


class VectorStore:

    def __init__(self, dim: int, use_milvus: bool = False,
                 collection_name: str = "dirrag_vectors"):
        self.dim = dim
        self.use_milvus = use_milvus
        self.collection_name = collection_name

                        
        self._ids: List[Any] = []
        self._embeddings: List[np.ndarray] = []
        self._metadatas: List[dict] = []
        self._emb_matrix: Optional[np.ndarray] = None                  
        self._dirty: bool = True              

                              
        if use_milvus:
            self._init_milvus()

    def _init_milvus(self):
        try:
            from pymilvus import (
                connections, Collection, utility,
                FieldSchema, CollectionSchema, DataType
            )
                             
            connections.connect(alias="default", host="localhost", port="19530")

                                
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)

                         
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="item_id", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields, f"DirRAG vector collection: {self.collection_name}")
            self._milvus_collection = Collection(self.collection_name, schema)

                        
            index_params = {
                "index_type": "HNSW",
                "metric_type": "IP",                     
                "params": {"M": 16, "efConstruction": 200},
            }
            self._milvus_collection.create_index("embedding", index_params)
            self._milvus_loaded = False
            print(f"✅ Milvus 集合 '{self.collection_name}' 初始化成功(HNSW索引)")
        except ImportError:
            print("⚠️ pymilvus 未安装，回退到内存 numpy 后端")
            self.use_milvus = False
        except Exception as e:
            print(f"⚠️ Milvus 连接失败: {str(e)[:60]}，回退到内存 numpy 后端")
            self.use_milvus = False

    def add(self, item_id: Any, embedding: np.ndarray, metadata: Optional[dict] = None) -> None:
        self._ids.append(item_id)
        self._embeddings.append(np.array(embedding, dtype=np.float32))
        self._metadatas.append(metadata or {})
        self._dirty = True

        if self.use_milvus:
                                         
            pass

    def _rebuild_matrix(self):
        if not self._embeddings:
            self._emb_matrix = None
            return
        self._emb_matrix = np.stack(self._embeddings)            
                    
        norms = np.linalg.norm(self._emb_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self._emb_matrix_normalized = self._emb_matrix / norms
        self._dirty = False

    def search(self, query_embedding: np.ndarray, top_k: int = 10
               ) -> List[Tuple[Any, float, dict]]:
        if not self._embeddings:
            return []

        if self._dirty:
            self._rebuild_matrix()

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            return []
        query_normalized = query / query_norm

                            
        sims = self._emb_matrix_normalized @ query_normalized

                  
        sorted_idx = np.argsort(sims)[::-1][:top_k]
        return [
            (self._ids[i], float(sims[i]), self._metadatas[i])
            for i in sorted_idx
        ]

    def search_by_ids(self, query_embedding: np.ndarray,
                      candidate_ids: Set[Any], top_k: int = 10
                      ) -> List[Tuple[Any, float, dict]]:
        if not self._embeddings or not candidate_ids:
            return []

        if self._dirty:
            self._rebuild_matrix()

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            return []
        query_normalized = query / query_norm

                 
        sims = self._emb_matrix_normalized @ query_normalized

                         
        results = []
        for i in range(len(self._ids)):
            if self._ids[i] in candidate_ids:
                results.append((self._ids[i], float(sims[i]), self._metadatas[i]))

                  
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        self._ids = []
        self._embeddings = []
        self._metadatas = []
        self._emb_matrix = None
        self._dirty = True

        if self.use_milvus:
            try:
                from pymilvus import utility
                if utility.has_collection(self.collection_name):
                    utility.drop_collection(self.collection_name)
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self._ids)


class TrackedLLM:

    def __init__(self, llm_client, tokenizer=None):
        self.llm = llm_client
        self.tokenizer = tokenizer
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.call_count: int = 0
                       
        self.indexing_call_count: int = 0
        self.indexing_input_tokens: int = 0
        self.indexing_output_tokens: int = 0
                                                  
        self.retrieval_call_count: int = 0
        self.retrieval_input_tokens: int = 0
        self.retrieval_output_tokens: int = 0

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
                                              
        return int(len(text.split()) * 1.3)

    def chat(self, prompt: str, stage: str = "query") -> str:
        self.call_count += 1
        input_tokens = self._count_tokens(prompt)
        response = self.llm.chat(prompt)
        output_tokens = self._count_tokens(response)

        if stage == "indexing":
            self.indexing_call_count += 1
            self.indexing_input_tokens += input_tokens
            self.indexing_output_tokens += output_tokens
        elif stage == "retrieval":
                                                 
            self.retrieval_call_count += 1
            self.retrieval_input_tokens += input_tokens
            self.retrieval_output_tokens += output_tokens
        else:
                                        
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
        return response

    def reset(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
                                   
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
        if self.llm is not None and hasattr(self.llm, "destroy"):
            self.llm.destroy()

    def ensure_llm_loaded(self):
        if self.llm is not None and hasattr(self.llm, "ensure_loaded"):
            self.llm.ensure_loaded()

    def is_llm_loaded(self) -> bool:
        if self.llm is None:
            return False
        if hasattr(self.llm, "is_loaded"):
            return self.llm.is_loaded()
        return True


class EntityCache:

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._cache: Dict[str, List[str]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.loads(f.read())
                print(f"🗂️  实体缓存已加载: {len(self._cache)} 条记录")
            except Exception as e:
                print(f"⚠️ 实体缓存加载失败: {str(e)[:50]}")
                self._cache = {}

    def _save(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(self._cache, ensure_ascii=False))
        except Exception as e:
            print(f"⚠️ 实体缓存保存失败: {str(e)[:50]}")

    @staticmethod
    def make_key(query_id: str, article_title: str, chunk_index: int) -> str:
        return f"{query_id}::{article_title}::{chunk_index}"

    def get(self, query_id: str, article_title: str, chunk_index: int) -> Optional[List[str]]:
        key = self.make_key(query_id, article_title, chunk_index)
        return self._cache.get(key)

    def set(self, query_id: str, article_title: str, chunk_index: int,
            entities: List[str]) -> None:
        key = self.make_key(query_id, article_title, chunk_index)
        self._cache[key] = entities

    def flush(self) -> None:
        self._save()

    def __len__(self) -> int:
        return len(self._cache)


class DirRAG:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_client: TrackedLLM,
        chunk_size: int = 200,
        overlap: int = 32,
        top_d: int = 5,
        top_k: int = 5,
        max_iterations: int = 3,
        entity_sim_threshold: float = 0.75,
        use_milvus: bool = False,
        entity_cache: Optional[EntityCache] = None,
        query_id: str = "",
        rrf_k: int = 60,
                        
        use_dense: bool = True,
        use_bm25: bool = True,
        use_entity: bool = True,
        use_hierarchy: bool = True,
        mask_path_name: bool = False,
    ):
                    
        self.embedding_model = embedding_model
        self.llm_client = llm_client

                    
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_d = top_d
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.entity_sim_threshold = entity_sim_threshold
        self.use_milvus = use_milvus
        self.rrf_k = rrf_k            

                        
        self.use_dense = use_dense
        self.use_bm25 = use_bm25
        self.use_entity = use_entity
        self.use_hierarchy = use_hierarchy
        self.mask_path_name = mask_path_name

                      
        self.entity_cache = entity_cache
        self.query_id = query_id

                       
        self.entity_extractor = NEREntityExtractor(model_name="/root/dir-sem-rag/bert-based-ner", batch_size=64)

                       
        self.chunks: List[Chunk] = []
        self.chunk_id_to_idx: Dict[int, int] = {}                                  
        self._global_chunk_id_counter: int = 0

                     
        self.root: Optional[DirectoryNode] = None
        self.all_directories: List[DirectoryNode] = []                   
        self.dir_id_to_node: Dict[int, DirectoryNode] = {}
        self._global_dir_id_counter: int = 0

                      
        self.chunk_embedding_store: Optional[VectorStore] = None
                             
        self.dir_embedding_store: Optional[VectorStore] = None
                         
        self.chunk_bm25: BM25Index = BM25Index()
                        
        self.dir_bm25: BM25Index = BM25Index()
                                                              
        self.chunk_entity_map: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
                                                  
        self.dir_entity_map: Dict[str, List[int]] = defaultdict(list)
                                
        self.entity_name_store: Optional[VectorStore] = None
                                                                    
        self._chunk_to_entities: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

                                           
        self.visited_sequences: Set[tuple] = set()

                                     
        self._emb_dim: Optional[int] = None

                                              
        self._all_entity_names: Set[str] = set()

                                                        
    def build_index(self, passages: List[dict], extract_only: bool = False) -> dict:
        t0 = time.time()
        timing = {}

                
        self._reset_state()
                                         
        self.llm_client.indexing_call_count = 0
        self.llm_client.indexing_input_tokens = 0
        self.llm_client.indexing_output_tokens = 0

                      
        t_step = time.time()
        self._chunk_passages(passages)
        timing["chunking_time"] = round(time.time() - t_step, 4)
        if not self.chunks:
            timing["total_time"] = round(time.time() - t0, 4)
            return timing

                                                                        
        if extract_only:
            t_step = time.time()
            self._build_chunk_entity_index(extract_only=True)
            timing["chunk_entity_time"] = round(time.time() - t_step, 4)
            timing["entity_time_total"] = timing["chunk_entity_time"]
            timing["directory_tree_time"] = 0.0
            timing["chunk_embedding_time"] = 0.0
            timing["chunk_bm25_time"] = 0.0
            timing["dir_embedding_time"] = 0.0
            timing["dir_bm25_time"] = 0.0
            timing["dir_entity_time"] = 0.0
            timing["embedding_time_total"] = 0.0
            timing["bm25_time_total"] = 0.0
            timing["total_time"] = round(time.time() - t0, 4)
            return timing

                                                     
        t_step = time.time()
        if self.use_hierarchy:
            self._build_directory_tree()
        else:
                                              
            self.root = DirectoryNode(full_path="", dir_id=-1, parent=None)
            for chunk in self.chunks:
                self.root.chunk_ids.add(chunk.chunk_id)
            self.root.update_all_chunk_ids()
        timing["directory_tree_time"] = round(time.time() - t_step, 4)

                                        
        t_step = time.time()
        if self.use_entity:
            self._build_chunk_entity_index()
        timing["chunk_entity_time"] = round(time.time() - t_step, 4)

        t_step = time.time()
        if self.use_dense:
            self._build_chunk_embedding_index()
        timing["chunk_embedding_time"] = round(time.time() - t_step, 4)

                                                       
        if self.use_entity and self.use_dense:
            self._build_entity_name_vectors()

        t_step = time.time()
        if self.use_bm25:
            self._build_chunk_bm25_index()
        timing["chunk_bm25_time"] = round(time.time() - t_step, 4)

                                               
        if self.use_hierarchy:
            t_step = time.time()
            if self.use_dense:
                self._build_directory_embedding_index()
            timing["dir_embedding_time"] = round(time.time() - t_step, 4)

            t_step = time.time()
            if self.use_bm25:
                self._build_directory_bm25_index()
            timing["dir_bm25_time"] = round(time.time() - t_step, 4)

            t_step = time.time()
            if self.use_entity:
                self._build_directory_entity_index()
            timing["dir_entity_time"] = round(time.time() - t_step, 4)
        else:
            timing["dir_embedding_time"] = 0.0
            timing["dir_bm25_time"] = 0.0
            timing["dir_entity_time"] = 0.0

            
        timing["embedding_time_total"] = round(
            timing["chunk_embedding_time"] + timing["dir_embedding_time"], 4)
        timing["bm25_time_total"] = round(
            timing["chunk_bm25_time"] + timing["dir_bm25_time"], 4)
        timing["entity_time_total"] = round(
            timing["chunk_entity_time"] + timing["dir_entity_time"], 4)
        timing["total_time"] = round(time.time() - t0, 4)

        return timing

    def _reset_state(self):
        self.chunks = []
        self.chunk_id_to_idx = {}
        self._global_chunk_id_counter = 0
        self.root = None
        self.all_directories = []
        self.dir_id_to_node = {}
        self._global_dir_id_counter = 0
        self.chunk_bm25 = BM25Index()
        self.dir_bm25 = BM25Index()
        self.chunk_entity_map = defaultdict(list)
        self.dir_entity_map = defaultdict(list)
        self.entity_name_store = None
        self._chunk_to_entities = defaultdict(list)
        self.visited_sequences = set()
        self._emb_dim = None
        self._all_entity_names = set()

    def _chunk_passages(self, passages: List[dict]):
        for passage in passages:
            title = passage.get("title", "")
                                  
            sentences = passage.get("sentence", [])
            if not sentences:
                                   
                content = passage.get("content", "")
                if not content.strip():
                    continue
                                      
                sentences = [content]
                               
                                      
                chunk_strs = chunk_text(content, self.chunk_size, self.overlap)
                for idx_in_article, chunk_text_str in enumerate(chunk_strs):
                    chunk = Chunk(
                        chunk_id=self._global_chunk_id_counter,
                        text=chunk_text_str,
                        article_title=title,
                        directory_path=title,
                        index_in_article=idx_in_article,
                        sent_ids={0},
                    )
                    self.chunks.append(chunk)
                    self.chunk_id_to_idx[chunk.chunk_id] = len(self.chunks) - 1
                    self._global_chunk_id_counter += 1
                continue

                              
            word_list: List[str] = []                    
            word_sent_ids: List[int] = []                      

            for sent_id, sent_text in enumerate(sentences):
                sent_words = sent_text.split()
                word_list.extend(sent_words)
                word_sent_ids.extend([sent_id] * len(sent_words))

            if not word_list:
                continue

                                                        
            start = 0
            idx_in_article = 0
            while start < len(word_list):
                end = min(start + self.chunk_size, len(word_list))
                chunk_words = word_list[start:end]
                chunk_text_str = " ".join(chunk_words)
                                    
                chunk_sent_ids = set(word_sent_ids[start:end])

                chunk = Chunk(
                    chunk_id=self._global_chunk_id_counter,
                    text=chunk_text_str,
                    article_title=title,
                    directory_path=title,                                      
                    index_in_article=idx_in_article,
                    sent_ids=chunk_sent_ids,
                )
                self.chunks.append(chunk)
                self.chunk_id_to_idx[chunk.chunk_id] = len(self.chunks) - 1
                self._global_chunk_id_counter += 1
                idx_in_article += 1

                if end == len(word_list):
                    break
                start += self.chunk_size - self.overlap

    @staticmethod
    def _mask_path(path: str) -> str:
        parts = [p for p in path.split("/") if p]
        return "/" + "/".join(["MASKED"] * len(parts)) if parts else "MASKED"

    def _build_directory_tree(self):
                 
        self.root = DirectoryNode(full_path="", dir_id=-1, parent=None)

                                            
        title_to_node: Dict[str, DirectoryNode] = {}

        for chunk in self.chunks:
            title = chunk.article_title
            if title in title_to_node:
                continue

                                                         
            dir_name = self._mask_path(title) if self.mask_path_name else title

                    
            dir_node = DirectoryNode(
                full_path=dir_name,
                dir_id=self._global_dir_id_counter,
                parent=self.root,
            )
                                                              
                                                            
            child_key = f"MASKED::{dir_node.dir_id}" if self.mask_path_name else dir_name
            self.root.children[child_key] = dir_node
            self.all_directories.append(dir_node)
            self.dir_id_to_node[dir_node.dir_id] = dir_node
            title_to_node[title] = dir_node
            self._global_dir_id_counter += 1

                                                          
        for chunk in self.chunks:
            node = title_to_node.get(chunk.article_title)
            if node is not None:
                node.chunk_ids.add(chunk.chunk_id)

                                 
        self.root.update_all_chunk_ids()

    def _build_chunk_embedding_index(self):
        if not self.chunks:
            return

        try:
            chunk_texts = [c.text for c in self.chunks]
            embeddings = self.embedding_model.encode(chunk_texts)

                             
            if self._emb_dim is None and len(embeddings) > 0:
                self._emb_dim = len(embeddings[0])

                    
            self.chunk_embedding_store = VectorStore(
                dim=self._emb_dim,
                use_milvus=self.use_milvus,
                collection_name="dirrag_chunk_emb",
            )
            for chunk, emb in zip(self.chunks, embeddings):
                self.chunk_embedding_store.add(
                    item_id=chunk.chunk_id,
                    embedding=emb,
                    metadata={"text": chunk.text, "article_title": chunk.article_title},
                )
        except Exception as e:
            print(f"⚠️ 文本块 embedding 构建失败: {str(e)[:60]}")
            self.chunk_embedding_store = VectorStore(
                dim=self._emb_dim or 768, use_milvus=False
            )

    def _build_chunk_bm25_index(self):
        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            self.chunk_bm25.add(doc_id=chunk.chunk_id, tokens=tokens)

    def _build_chunk_entity_index(self, extract_only: bool = False):
        chunk_texts = [c.text for c in self.chunks]
        batch_entities = self.entity_extractor.extract_batch(chunk_texts)

        all_entity_names = set()

        for chunk, entities in zip(self.chunks, batch_entities):
            entity_counts = Counter()
            for ent in entities:
                entity_counts[ent.lower()] += 1
                all_entity_names.add(ent)
            for ent_key, cnt in entity_counts.items():
                self.chunk_entity_map[ent_key].append((chunk.chunk_id, cnt))
                self._chunk_to_entities[chunk.chunk_id].append((ent_key, cnt))

                                          
        if all_entity_names and self._emb_dim is not None and not extract_only:
            try:
                entity_name_list = sorted(all_entity_names)
                entity_embeddings = self.embedding_model.encode(entity_name_list)
                self.entity_name_store = VectorStore(
                    dim=self._emb_dim,
                    use_milvus=self.use_milvus,
                    collection_name="dirrag_entity_names",
                )
                for name, emb in zip(entity_name_list, entity_embeddings):
                    self.entity_name_store.add(
                        item_id=name,
                        embedding=emb,
                        metadata={"entity_name": name},
                    )
            except Exception as e:
                print(f"⚠️ 实体名称向量化失败: {str(e)[:50]}")
                self.entity_name_store = None


    def _build_directory_embedding_index(self):
        if not self.all_directories or self._emb_dim is None:
            return

                             
        dir_paths = [d.full_path for d in self.all_directories]
        try:
            path_embeddings = self.embedding_model.encode(dir_paths)
        except Exception as e:
            print(f"⚠️ 目录路径 embedding 失败: {str(e)[:50]}")
            path_embeddings = [np.zeros(self._emb_dim) for _ in dir_paths]

                              
        self.dir_embedding_store = VectorStore(
            dim=self._emb_dim,
            use_milvus=self.use_milvus,
            collection_name="dirrag_dir_emb",
        )
        for dir_node, emb in zip(self.all_directories, path_embeddings):
            self.dir_embedding_store.add(
                item_id=dir_node.dir_id,
                embedding=np.array(emb, dtype=np.float32),
                metadata={"full_path": dir_node.full_path, "dir_id": dir_node.dir_id},
            )

    def _build_directory_bm25_index(self):
        for dir_node in self.all_directories:
                                          
            all_texts: List[str] = []

                                          
            all_texts.append(dir_node.full_path)

                                       
            for chunk_id in dir_node.chunk_ids:
                chunk_idx = self.chunk_id_to_idx.get(chunk_id)
                if chunk_idx is not None:
                    all_texts.append(self.chunks[chunk_idx].text)

                     
            combined_text = " ".join(all_texts)
            tokens = tokenize(combined_text)
            self.dir_bm25.add(doc_id=dir_node.dir_id, tokens=tokens)

    def _build_directory_entity_index(self):
        for dir_node in self.all_directories:
                                  
            entity_set: Set[str] = set()
            for chunk_id in dir_node.chunk_ids:
                for entity_name, _cnt in self._chunk_to_entities.get(chunk_id, []):
                    entity_set.add(entity_name)

                                                     
            for ent in entity_set:
                self.dir_entity_map[ent].append(dir_node.dir_id)

                                                      
    def answer_question(self, question: str) -> Tuple[str, dict]:
        t_start = time.time()

                    
        self.llm_client.reset()
        self.visited_sequences = set()

              
        total_retrieval_time = 0.0
        iterations_done = 0
        pred_answer = "I don't know"

                     
        iteration_details: List[dict] = []
                                             
        last_retrieved_chunk_ids: List[int] = []
        last_retrieved_dir_ids: List[int] = []

                                 
        keywords = tokenize(question)
        try:
            entities = self.entity_extractor.extract(question)
        except Exception:
            entities = []

        for iteration in range(1, self.max_iterations + 1):
            iterations_done = iteration
            print(f"    🔄 迭代 {iteration}/{self.max_iterations}")

                                  
            query_info = self._parse_query(question, keywords, entities)

                                   
            t_retrieval = time.time()
            top_dir_ids = self._retrieve_directories(query_info)

                                    
            top_chunk_ids = self._retrieve_chunks(query_info, top_dir_ids)
            retrieval_time = time.time() - t_retrieval
            total_retrieval_time += retrieval_time

                                         
            last_retrieved_chunk_ids = list(top_chunk_ids)
            last_retrieved_dir_ids = list(top_dir_ids)

            if not top_chunk_ids:
                                                 
                iteration_details.append({
                    "iteration": iteration,
                    "retrieved_dir_ids": top_dir_ids,
                    "retrieved_chunk_ids": [],
                    "expanded_chunk_ids": [],
                    "context": "",
                    "prompt": "",
                    "llm_raw_response": "",
                    "isSufficient": False,
                    "answer": "I don't know",
                    "reasoning": "",
                    "new_keywords": [],
                    "new_entities": [],
                    "used_chunk_info": [],
                    "retrieval_time_s": round(retrieval_time, 4),
                    "no_new_chunks": True,
                })
                pred_answer = "I don't know"
                break

                                               
            sorted_chunk_tuple = tuple(sorted(top_chunk_ids))
            if sorted_chunk_tuple in self.visited_sequences:
                print(f"    ⚠️ 当前文本块序列已访问过, 无法获得更多信息, 退出检索")
                iteration_details.append({
                    "iteration": iteration,
                    "retrieved_dir_ids": top_dir_ids,
                    "retrieved_chunk_ids": top_chunk_ids,
                    "expanded_chunk_ids": top_chunk_ids,
                    "context": "",
                    "prompt": "",
                    "llm_raw_response": "",
                    "isSufficient": False,
                    "answer": "I don't know",
                    "reasoning": "",
                    "new_keywords": [],
                    "new_entities": [],
                    "used_chunk_info": [],
                    "retrieval_time_s": round(retrieval_time, 4),
                    "duplicate_sequence": True,
                })
                pred_answer = "I don't know"
                break

                            
            self.visited_sequences.add(sorted_chunk_tuple)

                                                
            expanded_chunk_ids = self._expand_chunks(top_chunk_ids)
            llm_result = self._answer_with_llm(question, expanded_chunk_ids)
            answer = llm_result.get("answer", "I don't know")

                      
            iteration_details.append({
                "iteration": iteration,
                "retrieved_dir_ids": top_dir_ids,
                "retrieved_chunk_ids": top_chunk_ids,
                "expanded_chunk_ids": expanded_chunk_ids,
                "context": llm_result.get("context", ""),
                "prompt": llm_result.get("prompt", ""),
                "llm_raw_response": llm_result.get("llm_raw_response", ""),
                "isSufficient": answer != "I don't know",
                "answer": answer,
                "reasoning": "",
                "new_keywords": [],
                "new_entities": [],
                "used_chunk_info": llm_result.get("used_chunk_info", []),
                "retrieval_time_s": round(retrieval_time, 4),
            })

            if answer != "I don't know":
                           
                pred_answer = answer
                break

                                                                
            new_keywords, new_entities = self._extract_keywords_entities(
                question, expanded_chunk_ids
            )
            keywords = list(set(keywords + new_keywords))
            entities = list(set(entities + new_entities))
                                                  
            iteration_details[-1]["new_keywords"] = new_keywords
            iteration_details[-1]["new_entities"] = new_entities
            print(f"    📝 新增关键词: {new_keywords}, 新增实体: {new_entities}")

            if iteration == self.max_iterations:
                           
                pred_answer = "I don't know"
                break

        e2e_latency = time.time() - t_start

              
        metrics = {
            "token_cost": {
                "input_tokens": self.llm_client.total_input_tokens,
                "output_tokens": self.llm_client.total_output_tokens,
                "total_tokens": self.llm_client.total_tokens,
            },
            "retrieval_token_cost": {
                "input_tokens": self.llm_client.retrieval_input_tokens,
                "output_tokens": self.llm_client.retrieval_output_tokens,
                "total_tokens": self.llm_client.retrieval_total_tokens,
                "llm_call_count": self.llm_client.retrieval_call_count,
            },
            "llm_call_count": self.llm_client.call_count,
            "e2e_latency_s": round(e2e_latency, 4),
            "retrieval_time_s": round(total_retrieval_time, 4),
            "iterations": iterations_done,
            "iteration_details": iteration_details,
                                           
            "last_retrieved_chunk_ids": last_retrieved_chunk_ids,
            "last_retrieved_dir_ids": last_retrieved_dir_ids,
        }

        return pred_answer, metrics

    def compute_recall_mrr(self, supporting_facts: List[dict],
                           retrieved_chunk_ids_ordered: List[int]) -> dict:
        if not supporting_facts or not retrieved_chunk_ids_ordered:
            return {
                "recall_at_k": 0.0, "mrr": 0.0,
                "hit_count": 0, "total_evidence": len(supporting_facts),
                "first_hit_rank": 0,
            }

                                                   
        evidence_set: Set[Tuple[str, int]] = set()
        for sf in supporting_facts:
            title = sf.get("title", "")
            sent_id = sf.get("sent_id", -1)
            if title and sent_id >= 0:
                evidence_set.add((title, sent_id))

        total_evidence = len(evidence_set)
        if total_evidence == 0:
            return {
                "recall_at_k": 0.0, "mrr": 0.0,
                "hit_count": 0, "total_evidence": 0,
                "first_hit_rank": 0,
            }

                              
        hit_count = 0
        first_hit_rank = 0                   
        matched_evidence: Set[Tuple[str, int]] = set()

        for rank, chunk_id in enumerate(retrieved_chunk_ids_ordered, start=1):
            chunk_idx = self.chunk_id_to_idx.get(chunk_id)
            if chunk_idx is None:
                continue
            chunk = self.chunks[chunk_idx]
                            
            chunk_hit = False
            for sent_id in chunk.sent_ids:
                ev = (chunk.article_title, sent_id)
                if ev in evidence_set and ev not in matched_evidence:
                    matched_evidence.add(ev)
                    hit_count += 1
                    chunk_hit = True
                    if first_hit_rank == 0:
                        first_hit_rank = rank

        recall_at_k = hit_count / total_evidence
        mrr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0

        return {
            "recall_at_k": round(recall_at_k, 4),
            "mrr": round(mrr, 4),
            "hit_count": hit_count,
            "total_evidence": total_evidence,
            "first_hit_rank": first_hit_rank,
        }

    def compute_directory_recall(self, supporting_facts: List[dict],
                                 retrieved_dir_ids: List[int]) -> dict:
        if not supporting_facts:
            return {
                "directory_recall": 0.0,
                "recalled_dir_count": 0,
                "total_evidence_dirs": 0,
                "evidence_dir_titles": [],
                "recalled_dir_titles": [],
            }

                                              
        evidence_dir_titles: Set[str] = set()
        for sf in supporting_facts:
            title = sf.get("title", "")
            if title:
                evidence_dir_titles.add(title)

        total_evidence_dirs = len(evidence_dir_titles)
        if total_evidence_dirs == 0:
            return {
                "directory_recall": 0.0,
                "recalled_dir_count": 0,
                "total_evidence_dirs": 0,
                "evidence_dir_titles": [],
                "recalled_dir_titles": [],
            }

                    
        title_to_dir_id: Dict[str, int] = {}
        for dir_id, dir_node in self.dir_id_to_node.items():
            title_to_dir_id[dir_node.full_path] = dir_id

                  
        evidence_dir_ids: Set[int] = set()
        for title in evidence_dir_titles:
            dir_id = title_to_dir_id.get(title)
            if dir_id is not None:
                evidence_dir_ids.add(dir_id)

                   
        retrieved_dir_id_set: Set[int] = set(retrieved_dir_ids)

                          
        recalled_dir_ids: Set[int] = evidence_dir_ids & retrieved_dir_id_set
        recalled_dir_count = len(recalled_dir_ids)

                    
        recalled_dir_titles: List[str] = []
        for dir_id in recalled_dir_ids:
            dir_node = self.dir_id_to_node.get(dir_id)
            if dir_node:
                recalled_dir_titles.append(dir_node.full_path)

        directory_recall = recalled_dir_count / total_evidence_dirs

        return {
            "directory_recall": round(directory_recall, 4),
            "recalled_dir_count": recalled_dir_count,
            "total_evidence_dirs": total_evidence_dirs,
            "evidence_dir_titles": sorted(list(evidence_dir_titles)),
            "recalled_dir_titles": recalled_dir_titles,
        }

    def _parse_query(self, query: str, keywords: List[str],
                     entities: List[str]) -> dict:
                        
        try:
            query_emb = self.embedding_model.encode([query])[0]
        except Exception as e:
            print(f"⚠️ 查询 embedding 失败: {str(e)[:40]}")
            query_emb = np.zeros(self._emb_dim or 768, dtype=np.float32)

        return {
            "embedding": query_emb,
            "keywords": keywords,
            "entities": entities,
        }

    def _retrieve_directories(self, query_info: dict) -> List[int]:
                                 
        if not self.use_hierarchy:
            return []

        query_emb = query_info["embedding"]
        query_keywords = query_info["keywords"]
        query_entities = query_info["entities"]

                               
        emb_ranked: List[int] = []
        bm25_ranked: List[int] = []
        entity_ranked: List[int] = []

                                                   
        if self.use_dense and self.dir_embedding_store and len(self.dir_embedding_store) > 0:
            emb_results = self.dir_embedding_store.search(query_emb, top_k=self.top_d * 2)
            emb_ranked = [dir_id for dir_id, _, _ in emb_results]

                                             
        if self.use_bm25 and self.dir_bm25 and len(self.dir_bm25) > 0 and query_keywords:
            bm25_results = self.dir_bm25.search(query_keywords, top_k=self.top_d * 2)
            bm25_ranked = [dir_id for dir_id, _ in bm25_results]

                                            
        if self.use_entity and query_entities and self.entity_name_store and len(self.entity_name_store) > 0:
            entity_scores = self._entity_retrieve_directory(query_entities, self.top_d * 2)
            entity_ranked = [
                dir_id for dir_id, _ in sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
            ][:self.top_d * 2]

                        
        rank_lists = [rl for rl in [emb_ranked, bm25_ranked, entity_ranked] if rl]
        if not rank_lists:
            return []

        final_scores = rrf_score(rank_lists, k=self.rrf_k)

                   
        sorted_dirs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_dir_ids = [d_id for d_id, _ in sorted_dirs[:self.top_d]]

        return top_dir_ids

    def _entity_retrieve_directory(self, query_entities: List[str],
                                   top_k: int) -> Dict[int, float]:
        if not query_entities:
            return {}

                         
        dir_covered_entities: Dict[int, Set[str]] = defaultdict(set)

        for entity in query_entities:
                          
            matched_entity_names: Set[str] = set()
            try:
                entity_emb = self.embedding_model.encode([entity])[0]
            except Exception:
                continue

            matches = self.entity_name_store.search(entity_emb, top_k=top_k)
            for matched_name, sim, _ in matches:
                if sim < self.entity_sim_threshold:
                    continue
                matched_entity_names.add(matched_name)

                                  
            for matched_name in matched_entity_names:
                for dir_id in self.dir_entity_map.get(matched_name, []):
                    dir_covered_entities[dir_id].add(entity)

                                     
        total_query_entities = len(query_entities)
        coverage_scores: Dict[int, float] = {}
        for dir_id, covered in dir_covered_entities.items():
            coverage_scores[dir_id] = len(covered) / total_query_entities

        return coverage_scores

    def _retrieve_chunks(self, query_info: dict,
                         top_dir_ids: List[int]) -> List[int]:
        query_emb = query_info["embedding"]
        query_keywords = query_info["keywords"]
        query_entities = query_info["entities"]

                                 
        if self.use_hierarchy and top_dir_ids:
                               
            candidate_chunk_ids: Set[int] = set()
            for dir_id in top_dir_ids:
                dir_node = self.dir_id_to_node.get(dir_id)
                if dir_node:
                    candidate_chunk_ids |= dir_node.all_chunk_ids
            if not candidate_chunk_ids:
                return []
            candidate_set = candidate_chunk_ids
        else:
                                         
            candidate_set = set(chunk.chunk_id for chunk in self.chunks)
            if not candidate_set:
                return []

                               
        emb_ranked: List[int] = []
        bm25_ranked: List[int] = []
        entity_ranked: List[int] = []

                                                   
        if self.use_dense and self.chunk_embedding_store and len(self.chunk_embedding_store) > 0:
            emb_results = self.chunk_embedding_store.search_by_ids(
                query_emb, candidate_set, top_k=self.top_k * 2
            )
            emb_ranked = [chunk_id for chunk_id, _, _ in emb_results]

                                             
        if self.use_bm25 and self.chunk_bm25 and len(self.chunk_bm25) > 0 and query_keywords:
            bm25_results = self.chunk_bm25.search(
                query_keywords, top_k=self.top_k * 2, filter_ids=candidate_set
            )
            bm25_ranked = [chunk_id for chunk_id, _ in bm25_results]

                                            
        if self.use_entity and query_entities and self.entity_name_store and len(self.entity_name_store) > 0:
            entity_scores = self._entity_retrieve_chunk(
                query_entities, candidate_set, self.top_k * 2
            )
            entity_ranked = [
                chunk_id for chunk_id, _ in sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
            ][:self.top_k * 2]

                        
        rank_lists = [rl for rl in [emb_ranked, bm25_ranked, entity_ranked] if rl]
        if not rank_lists:
            return []

        final_scores = rrf_score(rank_lists, k=self.rrf_k)

                   
        sorted_chunks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_chunk_ids = [c_id for c_id, _ in sorted_chunks[:self.top_k]]

        return top_chunk_ids

    def _entity_retrieve_chunk(self, query_entities: List[str],
                               candidate_ids: Set[int],
                               top_k: int) -> Dict[int, float]:
        if not query_entities:
            return {}

                          
        chunk_covered_entities: Dict[int, Set[str]] = defaultdict(set)

        for entity in query_entities:
            matched_entity_names: Set[str] = set()
            try:
                entity_emb = self.embedding_model.encode([entity])[0]
            except Exception:
                continue

            matches = self.entity_name_store.search(entity_emb, top_k=top_k)
            for matched_name, sim, _ in matches:
                if sim < self.entity_sim_threshold:
                    continue
                matched_entity_names.add(matched_name)

                                   
            for matched_name in matched_entity_names:
                for chunk_id, _cnt in self.chunk_entity_map.get(matched_name, []):
                    if chunk_id in candidate_ids:
                        chunk_covered_entities[chunk_id].add(entity)

                                     
        total_query_entities = len(query_entities)
        coverage_scores: Dict[int, float] = {}
        for chunk_id, covered in chunk_covered_entities.items():
            coverage_scores[chunk_id] = len(covered) / total_query_entities

        return coverage_scores

    def _expand_chunks(self, chunk_ids: List[int]) -> List[int]:
        unique_ids = set(chunk_ids)
        sorted_ids = sorted(
            unique_ids,
            key=lambda cid: (
                self.chunks[self.chunk_id_to_idx[cid]].article_title,
                self.chunks[self.chunk_id_to_idx[cid]].index_in_article,
            ),
        )
        return sorted_ids

    def _answer_with_llm(self, question: str,
                         chunk_ids: List[int]) -> dict:
                                             
        context_parts: List[str] = []
        used_chunk_info: List[dict] = []
        for chunk_id in chunk_ids:
            chunk_idx = self.chunk_id_to_idx.get(chunk_id)
            if chunk_idx is not None:
                chunk = self.chunks[chunk_idx]
                source_tag = f"[Article: {chunk.article_title} | Path: {chunk.directory_path}]"
                context_parts.append(f"{source_tag}\n{chunk.text}")
                used_chunk_info.append({
                    "chunk_id": chunk.chunk_id,
                    "article_title": chunk.article_title,
                    "directory_path": chunk.directory_path,
                    "index_in_article": chunk.index_in_article,
                    "text_preview": chunk.text[:200],
                })

        context = "\n\n".join(context_parts) if context_parts else "(no context available)"

                                                  
        prompt = f"""Answer the question using the provided context or based on your own knowledge.

Context:

{context}

Question: {question}

Instructions:

1. Give the shortest correct answer phrase.

2. If you do NOT know the answer or the context does NOT contain enough information to answer the question, respond with exactly: "I don't know". Please do NOT randomly guess the answer.

3. Output ONLY the answer, nothing else (no explanation, no JSON, no prefix)."""

        try:
            response = self.llm_client.chat(prompt)
        except Exception as e:
            print(f"⚠️ LLM 回答失败: {str(e)[:50]}")
            return {
                "answer": "I don't know",
                "context": context, "prompt": prompt,
                "llm_raw_response": f"ERROR: {str(e)[:100]}",
                "used_chunk_info": used_chunk_info,
            }

                                                  
        answer = response.strip()
                                                  
        if normalize_answer(answer) == normalize_answer("I don't know"):
            answer = "I don't know"

        return {
            "answer": answer,
            "context": context,
            "prompt": prompt,
            "llm_raw_response": response,
            "used_chunk_info": used_chunk_info,
        }

    def _extract_keywords_entities(self, question: str,
                                   chunk_ids: List[int]) -> Tuple[List[str], List[str]]:
        context_parts: List[str] = []
        for chunk_id in chunk_ids:
            chunk_idx = self.chunk_id_to_idx.get(chunk_id)
            if chunk_idx is not None:
                chunk = self.chunks[chunk_idx]
                context_parts.append(chunk.text)

        context = "\n\n".join(context_parts) if context_parts else "(no context)"

        prompt = f"""Based on the question and the context below, extract useful keywords and named entities that could help retrieve more relevant documents in the next search round.

Question: {question}

Context:
{context}

Return ONLY a valid JSON object with this format, no markdown, no explanation:
{{"keywords": ["keyword1", "keyword2"], "entities": ["entity1", "entity2"]}}

Rules:
- Keywords should be single words or short phrases that could help find relevant documents.
- Entities should be specific named entities (person names, organization names, locations, dates, concepts, etc.).
- Only extract terms that are actually present or strongly implied in the context.
- Do NOT repeat keywords or entities.

JSON:"""

        try:
            response = self.llm_client.chat(prompt, stage="retrieval")
        except Exception as e:
            print(f"⚠️ 关键词/实体提取 LLM 调用失败: {str(e)[:50]}")
            return [], []

        result = parse_llm_json(response)
        if isinstance(result, dict):
            keywords = [str(k).strip() for k in result.get("keywords", []) if k]
            entities = [str(e).strip() for e in result.get("entities", []) if e]
            return keywords, entities
        return [], []

                                                      
    def get_chunk_count(self) -> int:
        return len(self.chunks)

    def get_directory_count(self) -> int:
        return len(self.all_directories)

    def get_index_size_mb(self) -> float:
        total_bytes = 0
        if self.chunk_embedding_store:
            for emb in self.chunk_embedding_store._embeddings:
                total_bytes += emb.nbytes
        if self.dir_embedding_store:
            for emb in self.dir_embedding_store._embeddings:
                total_bytes += emb.nbytes
        if self.entity_name_store:
            for emb in self.entity_name_store._embeddings:
                total_bytes += emb.nbytes
        return total_bytes / (1024 * 1024)

    def reset(self) -> None:
        self._reset_state()
        if self.chunk_embedding_store:
            self.chunk_embedding_store.clear()
        if self.dir_embedding_store:
            self.dir_embedding_store.clear()
        if self.entity_name_store:
            self.entity_name_store.clear()
        self.chunk_bm25.clear()
        self.dir_bm25.clear()
        gc.collect()


def run_evaluation(
    samples: List[dict],
    local_llm: TrackedLLM,
    output_path: str,
    max_samples: Optional[int] = None,
    chunk_size: int = 200,
    overlap: int = 32,
    top_d: int = 5,
    top_k: int = 5,
    max_iterations: int = 3,
    use_milvus: bool = False,
    entity_cache_path: str = "",
                                                   
    extract_only: bool = False,
    extract_only_stats_path: str = "",
                    
    use_dense: bool = True,
    use_bm25: bool = True,
    use_entity: bool = True,
    use_hierarchy: bool = True,
    mask_path_name: bool = False,
    variant_name: str = "full",
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

                      
    entity_cache = None
    if entity_cache_path:
        entity_cache = EntityCache(entity_cache_path)
        print(f"🗂️  实体缓存已启用: {entity_cache_path}")

                                                     
    dir_rag = DirRAG(
        embedding_model=embedding_model,
        llm_client=local_llm,
        chunk_size=chunk_size,
        overlap=overlap,
        top_d=top_d,
        top_k=top_k,
        max_iterations=max_iterations,
        use_milvus=use_milvus,
        entity_cache=entity_cache,
        query_id="",
        use_dense=use_dense,
        use_bm25=use_bm25,
        use_entity=use_entity,
        use_hierarchy=use_hierarchy,
        mask_path_name=mask_path_name,
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

    all_target_samples = samples[:max_samples] if max_samples else samples
    samples_to_run = [s for s in all_target_samples if s["_id"] not in processed_ids]
    total_to_run = len(samples_to_run)

    print("\n" + "=" * 70)
    print(f"🚀 hotpotqa DirRAG [{variant_name}] | 目录级检索版 | 剩余: {total_to_run}")
    print(f"✅ chunk_size={chunk_size}  top_d={top_d}  top_k={top_k}  max_iter={max_iterations}")
    print(f"   dense={use_dense}  bm25={use_bm25}  entity={use_entity}  "
          f"hierarchy={use_hierarchy}  mask_path={mask_path_name}")
    print("=" * 70)

    for idx, sample in enumerate(samples_to_run, 1):
        _id = sample["_id"]
        question = sample["question"]
        gold = sample["answer"]
        passages = sample["passages"]
        supporting_facts = sample.get("supporting_facts", [])

        print(f"\n📄 [{idx}/{total_to_run}] ID: {_id[:20]}...")
        print(f"Q: {question}")
        print(f"📰 文章数: {len(passages)}, 证据数: {len(supporting_facts)}")

        try:
                    
            dir_rag.reset()
                                         
            dir_rag.query_id = _id
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

                                       
            index_timing = dir_rag.build_index(passages, extract_only=extract_only)
            total_chunks = dir_rag.get_chunk_count()
            total_dirs = dir_rag.get_directory_count()
            index_size_mb = dir_rag.get_index_size_mb()

            print(f"📦 文本块: {total_chunks}, 目录: {total_dirs}, "
                  f"IndexTime={index_timing.get('total_time', 0):.3f}s, "
                  f"IndexSize={index_size_mb:.2f}MB")
            print(f"   ⏱️  Emb={index_timing.get('embedding_time_total', 0):.3f}s, "
                  f"BM25={index_timing.get('bm25_time_total', 0):.3f}s, "
                  f"Entity={index_timing.get('entity_time_total', 0):.3f}s")

                                                                 
            if extract_only:
                stats_item = {
                    "id": _id,
                    "question": question,
                    "metrics": {
                        "num_chunks": total_chunks,
                        "indexing_time_s": index_timing.get("total_time", 0),
                        "indexing_time_breakdown": index_timing,
                        "indexing_token_cost": {
                            "input_tokens": dir_rag.llm_client.indexing_input_tokens,
                            "output_tokens": dir_rag.llm_client.indexing_output_tokens,
                            "total_tokens": dir_rag.llm_client.indexing_total_tokens,
                            "llm_call_count": dir_rag.llm_client.indexing_call_count,
                        },
                    },
                }
                stats_path = extract_only_stats_path or output_path.replace(".jsonl", "_extract_only.jsonl")
                with open(stats_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(stats_item, ensure_ascii=False) + "\n")
                print(f"    📝 extract_only: LLM 提取完成, "
                      f"耗时={index_timing.get('total_time', 0):.3f}s, "
                      f"Token={dir_rag.llm_client.indexing_total_tokens}")
                continue

                             
            pred_answer, metrics = dir_rag.answer_question(question)

                                       
            retrieved_ordered: List[int] = metrics.get("last_retrieved_chunk_ids", [])
                                  
            all_retrieved_dir_ids: List[int] = []
            seen_dir_ids: set = set()
            for it_detail in metrics.get("iteration_details", []):
                for did in it_detail.get("retrieved_dir_ids", []):
                    if did not in seen_dir_ids:
                        seen_dir_ids.add(did)
                        all_retrieved_dir_ids.append(did)

            recall_metrics = dir_rag.compute_recall_mrr(
                supporting_facts=supporting_facts,
                retrieved_chunk_ids_ordered=retrieved_ordered,
            ) if supporting_facts else {
                "recall_at_k": 0.0, "mrr": 0.0,
                "hit_count": 0, "total_evidence": 0, "first_hit_rank": 0,
            }

                                         
            dir_recall_metrics = dir_rag.compute_directory_recall(
                supporting_facts=supporting_facts,
                retrieved_dir_ids=all_retrieved_dir_ids,
            ) if supporting_facts else {
                "directory_recall": 0.0,
                "recalled_dir_count": 0,
                "total_evidence_dirs": 0,
                "evidence_dir_titles": [],
                "recalled_dir_titles": [],
            }

                
            pred_norm = normalize_answer(pred_answer)
            gold_norm = normalize_answer(gold)
            is_correct = pred_norm == gold_norm

                                                   
            result_item = {
                "id": _id,
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred_answer,
                "is_correct": is_correct,
                "metrics": {
                    "num_chunks": total_chunks,
                    "num_directories": total_dirs,
                    "top_d": top_d,
                    "top_k": top_k,
                    "indexing_time_s": index_timing.get("total_time", 0),
                    "indexing_time_breakdown": index_timing,           
                    "retrieval_time_s": metrics["retrieval_time_s"],
                    "e2e_latency_s": metrics["e2e_latency_s"],
                    "token_cost": metrics["token_cost"],
                    "retrieval_token_cost": metrics.get("retrieval_token_cost", {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "llm_call_count": 0,
                    }),
                    "indexing_token_cost": {
                        "input_tokens": dir_rag.llm_client.indexing_input_tokens,
                        "output_tokens": dir_rag.llm_client.indexing_output_tokens,
                        "total_tokens": dir_rag.llm_client.indexing_total_tokens,
                        "llm_call_count": dir_rag.llm_client.indexing_call_count,
                    },
                    "llm_call_count": metrics["llm_call_count"],
                    "iterations": metrics["iterations"],
                    "index_size_mb": round(index_size_mb, 3),
                                       
                    "recall_at_k": recall_metrics["recall_at_k"],
                    "mrr": recall_metrics["mrr"],
                    "hit_count": recall_metrics["hit_count"],
                    "total_evidence": recall_metrics["total_evidence"],
                    "first_hit_rank": recall_metrics["first_hit_rank"],
                                         
                    "directory_recall": dir_recall_metrics["directory_recall"],
                    "recalled_dir_count": dir_recall_metrics["recalled_dir_count"],
                    "total_evidence_dirs": dir_recall_metrics["total_evidence_dirs"],
                    "evidence_dir_titles": dir_recall_metrics["evidence_dir_titles"],
                    "recalled_dir_titles": dir_recall_metrics["recalled_dir_titles"],
                },
                                           
                "iteration_details": metrics.get("iteration_details", []),
            }
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

            print(f"A: {pred_answer} | {'✅' if is_correct else '❌'} Gold: {gold} | "
                  f"E2E={metrics['e2e_latency_s']:.3f}s | "
                  f"Retrieval={metrics['retrieval_time_s']:.3f}s | "
                  f"LLMCalls={metrics['llm_call_count']} | "
                  f"Tokens={metrics['token_cost']['total_tokens']} | "
                  f"Iters={metrics['iterations']} | "
                  f"Recall@K={recall_metrics['recall_at_k']:.2f} | "
                  f"MRR={recall_metrics['mrr']:.4f} | "
                  f"DirRecall={dir_recall_metrics['directory_recall']:.2f}"
                  f"({dir_recall_metrics['recalled_dir_count']}/{dir_recall_metrics['total_evidence_dirs']})")
            time.sleep(0.5)

        except Exception as e:
            print(f"⚠️ 样本跳过: {str(e)[:80]}")
            import traceback
            traceback.print_exc()
            time.sleep(1)
            continue

    print("\n✅ 评估完成!")


if __name__ == "__main__":
                  
    LONGBENCH_hotpotqa_PATH = "/root/dir-sem-rag/longbench/data_recall/hotpotqa_recall_clean.jsonl"
                           
    OUTPUT_DIR = "./dirrag_hotpotqa_ablation_results"
                              
    ENTITY_CACHE_PATH = "./dirrag_entity_cache.json"

                  
    MAX_SAMPLES = None                        
    CHUNK_SIZE = 100                
    OVERLAP = 32                    
    TOP_D = 5                              
    TOP_K = 5                               
    MAX_ITERATIONS = 3                
    USE_MILVUS = False
    EXTRACT_ONLY = False
    EXTRACT_ONLY_STATS_PATH = "./dirrag_extract_only_stats.jsonl"

                      
    ABLATION_VARIANTS = [
           
                             
        {
            "name": "wo_path_name",
            "desc": "w/o Path Name (all directory names masked)",
            "use_dense": True, "use_bm25": True, "use_entity": True,
            "use_hierarchy": True, "mask_path_name": True,
            "max_iterations": MAX_ITERATIONS,
        },
    ]

                     
    os.makedirs(OUTPUT_DIR, exist_ok=True)

                              
    local_llm = LocalLLMClient(model_path=MODEL_PATH)

                                     
    samples = []
    with open(LONGBENCH_hotpotqa_PATH, encoding="utf-8") as f:
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
    print(f"📚 共加载 {len(samples)} 个 hotpotqa 样本")

                        
    for variant in ABLATION_VARIANTS:
        v_name = variant["name"]
        v_desc = variant["desc"]
        output_path = os.path.join(OUTPUT_DIR, f"{v_name}.jsonl")

        print("\n" + "#" * 70)
        print(f"#  消融变体: {v_name} — {v_desc}")
        print("#" * 70)

        run_evaluation(
            samples=samples,
            local_llm=local_llm,
            output_path=output_path,
            max_samples=MAX_SAMPLES,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
            top_d=TOP_D,
            top_k=TOP_K,
            max_iterations=variant["max_iterations"],
            use_milvus=USE_MILVUS,
            entity_cache_path=ENTITY_CACHE_PATH,
            extract_only=EXTRACT_ONLY,
            extract_only_stats_path=EXTRACT_ONLY_STATS_PATH,
            use_dense=variant["use_dense"],
            use_bm25=variant["use_bm25"],
            use_entity=variant["use_entity"],
            use_hierarchy=variant["use_hierarchy"],
            mask_path_name=variant["mask_path_name"],
            variant_name=v_name,
        )

              
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("✅ 所有消融实验完成!")
    print(f"   结果目录: {OUTPUT_DIR}/")
    for v in ABLATION_VARIANTS:
        print(f"   - {v['name']}.jsonl")
    print("=" * 70)
