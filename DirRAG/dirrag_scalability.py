                     
                       
import os
import sys
import json
import re
import time
import gc
import math
import random
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np

                                                                              
sys.path.insert(0, "/root/dir-sem-rag/gte-Qwen2-1.5B-instruct")
sys.path.insert(1, "/root/dir-sem-rag/longbench/test/dir_rag_section_chunk_claude")
from embedding import EmbeddingModel

                   
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
                           
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def rrf_score(rank_lists: List[List[Any]], k: int = 60) -> Dict[Any, float]:
    scores: Dict[Any, float] = defaultdict(float)
    for rank_list in rank_lists:
        for rank, item_id in enumerate(rank_list, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return dict(scores)


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
        self.doc_tf: List[Dict[str, int]] = []                             
        self._doc_id_to_idx: Dict[Any, int] = {}
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
        self._doc_id_to_idx[doc_id] = doc_idx
        tf = Counter(tokens)
        self.doc_tf.append(dict(tf))
        for term, freq in tf.items():
            self.df[term] += 1
            self.inverted[term].append((doc_idx, freq))
        self.N += 1
        self._idf_computed = False

    def finalize(self) -> None:
        if not self._idf_computed:
            self._compute_idf()

    def _compute_idf(self) -> None:
        for term, df in self.df.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        self._idf_computed = True
                                                
        self._doc_len_arr = np.asarray(self.doc_len, dtype=np.float64)
        self._term_arr: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        for term, postings in self.inverted.items():
            idx = np.fromiter((p[0] for p in postings), dtype=np.int32, count=len(postings))
            tf = np.fromiter((p[1] for p in postings), dtype=np.float64, count=len(postings))
            self._term_arr[term] = (idx, tf, self.idf[term])

    def search(self, query_tokens: List[str], top_k: int = 10,
               filter_ids: Optional[Set[Any]] = None) -> List[Tuple[Any, float]]:
        if not self.doc_ids or not query_tokens:
            return []
        if not self._idf_computed:
            self._compute_idf()

        k1p = self.k1 + 1.0
        denom_const = self.k1 * (1.0 - self.b)
        denom_coef = self.k1 * self.b / max(self.avgdl, 1e-8)
        scores = np.zeros(self.N, dtype=np.float64)

        for term in set(query_tokens):
            entry = self._term_arr.get(term)
            if entry is None:
                continue
            doc_idx, tf, idf_val = entry
            denominator = tf + denom_const + denom_coef * self._doc_len_arr[doc_idx]
            scores[doc_idx] += idf_val * (tf * k1p) / denominator

        if filter_ids is not None:
            keep = [self._doc_id_to_idx[d] for d in filter_ids
                    if d in self._doc_id_to_idx]
            mask = np.zeros(self.N, dtype=bool)
            if keep:
                mask[np.asarray(keep, dtype=np.int32)] = True
            scores = scores * mask

        nz = np.flatnonzero(scores)
        if nz.size == 0:
            return []
        if 0 < top_k < nz.size:
            part = nz[np.argpartition(scores[nz], -top_k)[-top_k:]]
            order = np.argsort(scores[part])[::-1]
            top = part[order]
        else:
            top = nz[np.argsort(scores[nz])[::-1]]
        return [(self.doc_ids[i], float(scores[i])) for i in top]

    def search_in_candidates(self, query_tokens: List[str],
                             candidate_ids: Set[Any],
                             top_k: int = 10) -> List[Tuple[Any, float]]:
        if not self.doc_ids or not query_tokens or not candidate_ids:
            return []
        if not self._idf_computed:
            self._compute_idf()

                                
        query_terms = set(query_tokens)
        scores: Dict[Any, float] = {}

        for doc_id in candidate_ids:
            doc_idx = self._doc_id_to_idx.get(doc_id)
            if doc_idx is None:
                continue
            tf_dict = self.doc_tf[doc_idx]
            dl = self.doc_len[doc_idx]
            score = 0.0
            for term in query_terms:
                tf = tf_dict.get(term)
                if not tf:
                    continue
                idf_val = self.idf.get(term, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-8))
                score += idf_val * numerator / denominator
            if score > 0:
                scores[doc_id] = score

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
                 device: Optional[int] = None):
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
    def _clean(ent_dicts: List[dict]) -> List[str]:
        entities: List[str] = []
        seen: Set[str] = set()
        for ed in ent_dicts:
            word = str(ed.get("word", "")).strip()
                         
            if len(word) <= 1:
                continue
            key = word.lower()
            if key not in seen:
                seen.add(key)
                entities.append(word)
        return entities

    def extract(self, text: str) -> List[str]:
        return self.extract_batch([text])[0]

    def extract_batch(self, texts: List[str]) -> List[List[str]]:
        results: List[List[str]] = []
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

    def __init__(self, dim: int, collection_name: str = "vectors",
                 hnsw_m: int = 16, hnsw_ef_construction: int = 200,
                 hnsw_ef_search: int = 64):
        self.dim = dim
        self.collection_name = collection_name
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        self._ids: List[int] = []
        self._embeddings: List[np.ndarray] = []
        self._matrix: Optional[np.ndarray] = None                     
        self._id_to_pos: Dict[int, int] = {}
        self._index = None                                         
        self._finalized: bool = False

    def add(self, item_id: int, embedding: np.ndarray) -> None:
        emb = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            return
        self._ids.append(int(item_id))
        self._embeddings.append(emb / norm)

    def finalize(self) -> None:
        if self._finalized:
            return
        n = len(self._ids)
        if n > 0:
            import hnswlib
            self._matrix = np.stack(self._embeddings)
            self._id_to_pos = {iid: pos for pos, iid in enumerate(self._ids)}
            self._index = hnswlib.Index(space="cosine", dim=self.dim)
            self._index.init_index(
                max_elements=n,
                ef_construction=self.hnsw_ef_construction,
                M=self.hnsw_m,
                random_seed=42,
            )
            self._index.add_items(self._matrix, np.array(self._ids))
            self._index.set_ef(self.hnsw_ef_search)
                                             
        self._embeddings = []
        self._finalized = True

    def search(self, query_embedding: np.ndarray, top_k: int = 10
               ) -> List[Tuple[int, float, dict]]:
        if not self._finalized:
            self.finalize()
        if self._index is None:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return []
        q = q / q_norm

        k = max(1, min(top_k, len(self._ids)))
        labels, distances = self._index.knn_query(q, k=k)
                                         
        return [
            (int(label), float(1.0 - dist), {})
            for label, dist in zip(labels[0], distances[0])
        ]

    def search_by_ids(self, query_embedding: np.ndarray,
                      candidate_ids: Set[int], top_k: int = 10
                      ) -> List[Tuple[int, float, dict]]:
        if not self._finalized:
            self.finalize()
        if self._matrix is None or not candidate_ids:
            return []

        positions = [self._id_to_pos[i] for i in candidate_ids if i in self._id_to_pos]
        if not positions:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return []
        q = q / q_norm

        sims = self._matrix[positions] @ q
        order = np.argsort(sims)[::-1][:top_k]
        return [
            (self._ids[positions[i]], float(sims[i]), {})
            for i in order
        ]

    def clear(self) -> None:
        self._ids = []
        self._embeddings = []
        self._matrix = None
        self._id_to_pos = {}
        self._index = None
        self._finalized = False

    def __len__(self) -> int:
        return len(self._ids)


class DirRAG:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        ner_extractor: NEREntityExtractor,
        chunk_size: int = 100,
        overlap: int = 32,
        top_d: int = 5,
        top_k: int = 5,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        rrf_k: int = 60,
    ):
        self.embedding_model = embedding_model
        self.ner_extractor = ner_extractor

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_d = top_d
        self.top_k = top_k
        self.rrf_k = rrf_k

        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

                       
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
        self._chunk_to_entities: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

                                     
        self._emb_dim: Optional[int] = None

    def _make_vector_store(self, collection_name: str) -> VectorStore:
        return VectorStore(
            dim=self._emb_dim,
            collection_name=collection_name,
            hnsw_m=self.hnsw_m,
            hnsw_ef_construction=self.hnsw_ef_construction,
            hnsw_ef_search=self.hnsw_ef_search,
        )

                                                        
    def build_index(self, passages: List[dict]) -> dict:
        t0 = time.time()
        timing: Dict[str, float] = {}

        self._reset_state()

                      
        t_step = time.time()
        self._chunk_passages(passages)
        timing["chunking_time"] = round(time.time() - t_step, 4)
        if not self.chunks:
            timing["total_time"] = round(time.time() - t0, 4)
            return timing

                       
        t_step = time.time()
        self._build_directory_tree()
        timing["directory_tree_time"] = round(time.time() - t_step, 4)

                                      
        t_par = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_emb = executor.submit(self._route_embedding)
            fut_bm25 = executor.submit(self._route_bm25)
            fut_entity = executor.submit(self._route_entity)
            emb_timing = fut_emb.result()
            bm25_timing = fut_bm25.result()
            entity_timing = fut_entity.result()
        timing["parallel_routes_time"] = round(time.time() - t_par, 4)

        timing.update(emb_timing)
        timing.update(bm25_timing)
        timing.update(entity_timing)

            
        timing["embedding_time_total"] = round(
            timing.get("chunk_embedding_time", 0) + timing.get("dir_embedding_time", 0), 4)
        timing["bm25_time_total"] = round(
            timing.get("chunk_bm25_time", 0) + timing.get("dir_bm25_time", 0), 4)
        timing["entity_time_total"] = round(
            timing.get("chunk_entity_time", 0) + timing.get("dir_entity_time", 0), 4)
        timing["total_time"] = round(time.time() - t0, 4)

        return timing

    def _route_embedding(self) -> dict:
        chunk_t = self._build_chunk_embedding_index()
        dir_t = self._build_directory_embedding_index()
        return {
            "chunk_embedding_time": round(chunk_t["total"], 4),
            "dir_embedding_time": round(dir_t["total"], 4),
            "encode_time": round(chunk_t["encode"] + dir_t["encode"], 4),
            "hnsw_index_time": round(chunk_t["insert"] + dir_t["insert"], 4),
        }

    def _route_bm25(self) -> dict:
        t = time.perf_counter()
        self._build_chunk_bm25_index()
        chunk_time = time.perf_counter() - t
        t = time.perf_counter()
        self._build_directory_bm25_index()
        dir_time = time.perf_counter() - t
        return {
            "chunk_bm25_time": round(chunk_time, 4),
            "dir_bm25_time": round(dir_time, 4),
        }

    def _route_entity(self) -> dict:
        t = time.perf_counter()
        self._build_chunk_entity_index()
        chunk_time = time.perf_counter() - t
        t = time.perf_counter()
        self._build_directory_entity_index()
        dir_time = time.perf_counter() - t
        return {
            "chunk_entity_time": round(chunk_time, 4),
            "dir_entity_time": round(dir_time, 4),
        }

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
        self._chunk_to_entities = defaultdict(list)
        self._emb_dim = None

    def _chunk_passages(self, passages: List[dict]):
        for passage in passages:
            title = passage.get("title", "")
            sentences = passage.get("sentence", [])
            if not sentences:
                                   
                content = passage.get("content", "")
                if not content.strip():
                    continue
                sentences = [content]

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

    def _build_directory_tree(self):
        self.root = DirectoryNode(full_path="", dir_id=-1, parent=None)

        seen_titles = set()
        for chunk in self.chunks:
            title = chunk.article_title
            if title in seen_titles:
                continue
            seen_titles.add(title)
            dir_node = DirectoryNode(
                full_path=title,
                dir_id=self._global_dir_id_counter,
                parent=self.root,
            )
            self.root.children[title] = dir_node
            self.all_directories.append(dir_node)
            self.dir_id_to_node[dir_node.dir_id] = dir_node
            self._global_dir_id_counter += 1

        for chunk in self.chunks:
            title = chunk.article_title
            if title in self.root.children:
                self.root.children[title].chunk_ids.add(chunk.chunk_id)

        self.root.update_all_chunk_ids()

    def _build_chunk_embedding_index(self) -> dict:
        t_total = time.time()
        t_encode = time.time()
        chunk_texts = [c.text for c in self.chunks]
        embeddings = self.embedding_model.encode(chunk_texts)
        encode_time = time.time() - t_encode

        if self._emb_dim is None and len(embeddings) > 0:
            self._emb_dim = len(embeddings[0])

        t_insert = time.time()
        self.chunk_embedding_store = self._make_vector_store("dirrag_scal_chunk_emb")
        for chunk, emb in zip(self.chunks, embeddings):
            self.chunk_embedding_store.add(item_id=chunk.chunk_id, embedding=emb)
        self.chunk_embedding_store.finalize()
        insert_time = time.time() - t_insert

        return {"total": time.time() - t_total, "encode": encode_time, "insert": insert_time}

    def _build_chunk_bm25_index(self):
        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            self.chunk_bm25.add(doc_id=chunk.chunk_id, tokens=tokens)
        self.chunk_bm25.finalize()

    def _build_chunk_entity_index(self):
        chunk_texts = [c.text for c in self.chunks]
        batch_entities = self.ner_extractor.extract_batch(chunk_texts)

        for chunk, entities in zip(self.chunks, batch_entities):
            entity_counts = Counter()
            for ent in entities:
                entity_counts[ent.lower()] += 1
            for ent_key, cnt in entity_counts.items():
                self.chunk_entity_map[ent_key].append((chunk.chunk_id, cnt))
                self._chunk_to_entities[chunk.chunk_id].append((ent_key, cnt))

    def _build_directory_embedding_index(self) -> dict:
        t_total = time.time()
        t_encode = time.time()
        dir_paths = [d.full_path for d in self.all_directories]
        path_embeddings = self.embedding_model.encode(dir_paths)
        encode_time = time.time() - t_encode

        t_insert = time.time()
        self.dir_embedding_store = self._make_vector_store("dirrag_scal_dir_emb")
        for dir_node, emb in zip(self.all_directories, path_embeddings):
            self.dir_embedding_store.add(item_id=dir_node.dir_id, embedding=emb)
        self.dir_embedding_store.finalize()
        insert_time = time.time() - t_insert

        return {"total": time.time() - t_total, "encode": encode_time, "insert": insert_time}

    def _build_directory_bm25_index(self):
        for dir_node in self.all_directories:
            all_texts: List[str] = [dir_node.full_path]
            for chunk_id in dir_node.chunk_ids:
                chunk_idx = self.chunk_id_to_idx.get(chunk_id)
                if chunk_idx is not None:
                    all_texts.append(self.chunks[chunk_idx].text)
            combined_text = " ".join(all_texts)
            tokens = tokenize(combined_text)
            self.dir_bm25.add(doc_id=dir_node.dir_id, tokens=tokens)
        self.dir_bm25.finalize()

    def _build_directory_entity_index(self):
        for dir_node in self.all_directories:
            entity_set: Set[str] = set()
            for chunk_id in dir_node.chunk_ids:
                for entity_name, _cnt in self._chunk_to_entities.get(chunk_id, []):
                    entity_set.add(entity_name)
            for ent in entity_set:
                self.dir_entity_map[ent].append(dir_node.dir_id)

                                                                 
    def retrieve_only(self, question: str) -> dict:
                                                                   
        t0 = time.perf_counter()

                      
        t = time.perf_counter()
        query_emb = self.embedding_model.encode([question])[0]
        query_embed_time = time.perf_counter() - t

        keywords = tokenize(question)

        t = time.perf_counter()
        query_entities = [e.lower() for e in self.ner_extractor.extract(question)]
        query_ner_time = time.perf_counter() - t

        query_info = {
            "embedding": query_emb,
            "keywords": keywords,
            "entities": query_entities,
        }

                           
        t_dir = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_emb = executor.submit(self._dir_emb_search, query_info)
            fut_bm25 = executor.submit(self._dir_bm25_search, query_info)
            fut_entity = executor.submit(self._dir_entity_search, query_info)
            emb_ranked, dir_emb_time = fut_emb.result()
            bm25_ranked, dir_bm25_time = fut_bm25.result()
            entity_ranked, dir_entity_time = fut_entity.result()
        dir_level_time = time.perf_counter() - t_dir

        rank_lists = [rl for rl in [emb_ranked, bm25_ranked, entity_ranked] if rl]
        top_dir_ids: List[int] = []
        t_fuse = time.perf_counter()
        if rank_lists:
            final_scores = rrf_score(rank_lists, k=self.rrf_k)
            sorted_dirs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            top_dir_ids = [d_id for d_id, _ in sorted_dirs[:self.top_d]]
        dir_rrf_time = time.perf_counter() - t_fuse

                                    
        t_cand = time.perf_counter()
        candidate_chunk_ids: Set[int] = set()
        for dir_id in top_dir_ids:
            dir_node = self.dir_id_to_node.get(dir_id)
            if dir_node:
                candidate_chunk_ids |= dir_node.all_chunk_ids
        candidate_build_time = time.perf_counter() - t_cand

        t_chunk = time.perf_counter()
        chunk_emb_time = chunk_bm25_time = chunk_entity_time = 0.0
        top_chunk_ids: List[int] = []
        if candidate_chunk_ids:
            with ThreadPoolExecutor(max_workers=3) as executor:
                fut_emb = executor.submit(self._chunk_emb_search, query_info, candidate_chunk_ids)
                fut_bm25 = executor.submit(self._chunk_bm25_search, query_info, candidate_chunk_ids)
                fut_entity = executor.submit(self._chunk_entity_search, query_info, candidate_chunk_ids)
                c_emb_ranked, chunk_emb_time = fut_emb.result()
                c_bm25_ranked, chunk_bm25_time = fut_bm25.result()
                c_entity_ranked, chunk_entity_time = fut_entity.result()
        chunk_level_time = time.perf_counter() - t_chunk

        t_fuse = time.perf_counter()
        if candidate_chunk_ids:
            rank_lists = [rl for rl in [c_emb_ranked, c_bm25_ranked, c_entity_ranked] if rl]
            if rank_lists:
                final_scores = rrf_score(rank_lists, k=self.rrf_k)
                sorted_chunks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
                top_chunk_ids = [c_id for c_id, _ in sorted_chunks[:self.top_k]]
        chunk_rrf_time = time.perf_counter() - t_fuse

                              
        _, dir_emb_solo_time = self._dir_emb_search(query_info)
        _, dir_bm25_solo_time = self._dir_bm25_search(query_info)
        _, dir_entity_solo_time = self._dir_entity_search(query_info)
        chunk_emb_solo_time = chunk_bm25_solo_time = chunk_entity_solo_time = 0.0
        if candidate_chunk_ids:
            _, chunk_emb_solo_time = self._chunk_emb_search(query_info, candidate_chunk_ids)
            _, chunk_bm25_solo_time = self._chunk_bm25_search(query_info, candidate_chunk_ids)
            _, chunk_entity_solo_time = self._chunk_entity_search(query_info, candidate_chunk_ids)

        return {
            "retrieval_time_total": round(time.perf_counter() - t0, 4),
            "query_embed_time": round(query_embed_time, 6),
            "query_ner_time": round(query_ner_time, 6),
            "dir_emb_time": round(dir_emb_time, 6),
            "dir_bm25_time": round(dir_bm25_time, 6),
            "dir_entity_time": round(dir_entity_time, 6),
            "dir_emb_solo_time": round(dir_emb_solo_time, 6),
            "dir_bm25_solo_time": round(dir_bm25_solo_time, 6),
            "dir_entity_solo_time": round(dir_entity_solo_time, 6),
            "chunk_emb_solo_time": round(chunk_emb_solo_time, 6),
            "chunk_bm25_solo_time": round(chunk_bm25_solo_time, 6),
            "chunk_entity_solo_time": round(chunk_entity_solo_time, 6),
            "dir_level_time": round(dir_level_time, 6),
            "dir_rrf_time": round(dir_rrf_time, 6),
            "candidate_build_time": round(candidate_build_time, 6),
            "chunk_emb_time": round(chunk_emb_time, 6),
            "chunk_bm25_time": round(chunk_bm25_time, 6),
            "chunk_entity_time": round(chunk_entity_time, 6),
            "chunk_level_time": round(chunk_level_time, 6),
            "chunk_rrf_time": round(chunk_rrf_time, 6),
            "top_dir_ids": top_dir_ids,
            "top_chunk_ids": top_chunk_ids,
        }

                     
    def _dir_emb_search(self, query_info: dict) -> Tuple[List[int], float]:
        t = time.perf_counter()
        ranked: List[int] = []
        if self.dir_embedding_store and len(self.dir_embedding_store) > 0:
            results = self.dir_embedding_store.search(
                query_info["embedding"], top_k=self.top_d * 2)
            ranked = [dir_id for dir_id, _, _ in results]
        return ranked, time.perf_counter() - t

    def _dir_bm25_search(self, query_info: dict) -> Tuple[List[int], float]:
        t = time.perf_counter()
        ranked: List[int] = []
        if len(self.dir_bm25) > 0 and query_info["keywords"]:
            results = self.dir_bm25.search(query_info["keywords"], top_k=self.top_d * 2)
            ranked = [dir_id for dir_id, _ in results]
        return ranked, time.perf_counter() - t

    def _dir_entity_search(self, query_info: dict) -> Tuple[List[int], float]:
        t = time.perf_counter()
        ranked: List[int] = []
        query_entities = query_info["entities"]
        if query_entities and self.dir_entity_map:
            dir_covered: Dict[int, Set[str]] = defaultdict(set)
            for ent in query_entities:
                for dir_id in self.dir_entity_map.get(ent, []):
                    dir_covered[dir_id].add(ent)
            total = len(query_entities)
            coverage = {d: len(c) / total for d, c in dir_covered.items()}
            ranked = [
                d_id for d_id, _ in sorted(coverage.items(), key=lambda x: x[1], reverse=True)
            ][:self.top_d * 2]
        return ranked, time.perf_counter() - t

                      
    def _chunk_emb_search(self, query_info: dict,
                          candidate_ids: Set[int]) -> Tuple[List[int], float]:
        t = time.perf_counter()
        ranked: List[int] = []
        if self.chunk_embedding_store and len(self.chunk_embedding_store) > 0:
            results = self.chunk_embedding_store.search_by_ids(
                query_info["embedding"], candidate_ids, top_k=self.top_k * 2)
            ranked = [chunk_id for chunk_id, _, _ in results]
        return ranked, time.perf_counter() - t

    def _chunk_bm25_search(self, query_info: dict,
                           candidate_ids: Set[int]) -> Tuple[List[int], float]:
        t = time.perf_counter()
        ranked: List[int] = []
        if len(self.chunk_bm25) > 0 and query_info["keywords"]:
                                 
            results = self.chunk_bm25.search_in_candidates(
                query_info["keywords"], candidate_ids, top_k=self.top_k * 2)
            ranked = [chunk_id for chunk_id, _ in results]
        return ranked, time.perf_counter() - t

    def _chunk_entity_search(self, query_info: dict,
                             candidate_ids: Set[int]) -> Tuple[List[int], float]:
        t = time.perf_counter()
        ranked: List[int] = []
        query_entities = query_info["entities"]
        if query_entities and candidate_ids:
            ent_set = set(query_entities)
            chunk_covered: Dict[int, Set[str]] = defaultdict(set)
            for chunk_id in candidate_ids:
                for ent, _cnt in self._chunk_to_entities.get(chunk_id, []):
                    if ent in ent_set:
                        chunk_covered[chunk_id].add(ent)
            total = len(query_entities)
            coverage = {c: len(cov) / total for c, cov in chunk_covered.items()}
            ranked = [
                c_id for c_id, _ in sorted(coverage.items(), key=lambda x: x[1], reverse=True)
            ][:self.top_k * 2]
        return ranked, time.perf_counter() - t

                                                      
    def get_chunk_count(self) -> int:
        return len(self.chunks)

    def get_directory_count(self) -> int:
        return len(self.all_directories)

    def reset(self) -> None:
        if self.chunk_embedding_store:
            self.chunk_embedding_store.clear()
            self.chunk_embedding_store = None
        if self.dir_embedding_store:
            self.dir_embedding_store.clear()
            self.dir_embedding_store = None
        self._reset_state()
        gc.collect()


def assemble_corpus(target_idx: int, factor: int, samples: List[dict],
                    seed: int) -> List[dict]:
    rng = random.Random(seed + factor * 1000 + target_idx)
    corpus_passages: List[dict] = list(samples[target_idx]["passages"])

    if factor > 1:
        other_indices = [j for j in range(len(samples)) if j != target_idx]
        k = factor - 1
        if k <= len(other_indices):
                                              
            picked = rng.sample(other_indices, k)
        else:
                                                
                                       
            picked = list(other_indices)
            picked.extend(rng.choices(other_indices, k=k - len(other_indices)))
        for j in picked:
            corpus_passages.extend(samples[j]["passages"])

           
    seen_titles: Set[str] = set()
    deduped: List[dict] = []
    for p in corpus_passages:
        title = p.get("title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        deduped.append(p)
    return deduped


def _numeric_mean(values: List[float], trim_ratio: float = 0.1) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = int(len(vals) * trim_ratio)
    if k > 0 and len(vals) > 2 * k:
        vals = vals[k:-k]
    return round(sum(vals) / len(vals), 6)


def _repeat_retrieve_median(dir_rag, question: str, repeats: int) -> dict:
    runs = [dir_rag.retrieve_only(question) for _ in range(max(1, repeats))]
    med: Dict[str, float] = {}
    for k in runs[0]:
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
        if vals:
            med[k] = round(float(np.median(vals)), 6)
    return med


def write_summary(output_path: str, summary_path: str, factor: int) -> None:
    records: List[dict] = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    if not records:
        return

    averages: Dict[str, float] = {}
    for key in ("num_articles", "num_directories", "num_chunks"):
        averages[key] = _numeric_mean([r.get(key, 0) for r in records])
    for section in ("indexing", "retrieval"):
        keys: Set[str] = set()
        for r in records:
            keys.update(r.get(section, {}).keys())
        for k in sorted(keys):
            vals = [r[section][k] for r in records
                    if isinstance(r.get(section, {}).get(k), (int, float))]
            averages[f"{section}.{k}"] = _numeric_mean(vals)

    summary = {"factor": factor, "num_records": len(records), "averages": averages}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def run_scalability(
    samples: List[dict],
    factors: List[int],
    num_queries: int,
    seed: int,
    out_dir: str,
    chunk_size: int = 100,
    overlap: int = 32,
    top_d: int = 5,
    top_k: int = 5,
    ner_model: str = "/root/dir-sem-rag/bert-based-ner",
    ner_batch_size: int = 64,
    retrieval_repeats: int = 5,
):
    print("📦 初始化 EmbeddingModel ...")
    embedding_model = EmbeddingModel()
    print("✅ Embedding model loaded.")

    ner_extractor = NEREntityExtractor(ner_model, batch_size=ner_batch_size)

    dir_rag = DirRAG(
        embedding_model=embedding_model,
        ner_extractor=ner_extractor,
        chunk_size=chunk_size,
        overlap=overlap,
        top_d=top_d,
        top_k=top_k,
    )

    os.makedirs(out_dir, exist_ok=True)
    target_samples = samples[:num_queries]
                                                 
    index_of: Dict[int, int] = {id(s): i for i, s in enumerate(samples)}

    for factor in factors:
        output_path = os.path.join(out_dir, f"dirrag_scalability_{factor}x.jsonl")
        summary_path = os.path.join(out_dir, f"dirrag_scalability_{factor}x_summary.json")

                       
        processed_ids: Set[str] = set()
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        processed_ids.add(json.loads(line)["id"])
                    except Exception:
                        continue

        samples_to_run = [s for s in target_samples if s["_id"] not in processed_ids]
        print("\n" + "=" * 70)
        print(f"🚀 扩展因子 {factor}x | 剩余 {len(samples_to_run)}/{len(target_samples)} 条 query")
        if factor > 1 and factor - 1 > len(samples) - 1:
            print(f"⚠️ 因子超出数据集规模({len(samples)} 条): 超出部分为重复抽样, "
                  f"标题去重后目录/文章数将饱和, 不再随 factor 增长")
        print("=" * 70)

        for idx, sample in enumerate(samples_to_run, 1):
            _id = sample["_id"]
            question = sample["question"]
            target_idx = index_of[id(sample)]

            try:
                passages = assemble_corpus(target_idx, factor, samples, seed)

                dir_rag.reset()
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass

                              
                index_timing = dir_rag.build_index(passages)

                                         
                retrieval_timing = _repeat_retrieve_median(
                    dir_rag, question, repeats=retrieval_repeats)

                record = {
                    "id": _id,
                    "factor": factor,
                    "num_articles": len(passages),
                    "num_directories": dir_rag.get_directory_count(),
                    "num_chunks": dir_rag.get_chunk_count(),
                    "indexing": index_timing,
                    "retrieval": retrieval_timing,
                }
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"📄 [{idx}/{len(samples_to_run)}] {factor}x | "
                      f"articles={record['num_articles']} dirs={record['num_directories']} "
                      f"chunks={record['num_chunks']} | "
                      f"IndexTotal={index_timing.get('total_time', 0):.3f}s "
                      f"(emb={index_timing.get('embedding_time_total', 0):.3f} "
                      f"bm25={index_timing.get('bm25_time_total', 0):.3f} "
                      f"ent={index_timing.get('entity_time_total', 0):.3f} "
                      f"par={index_timing.get('parallel_routes_time', 0):.3f}) | "
                      f"RetrTotal={retrieval_timing.get('retrieval_time_total', 0):.3f}s "
                      f"(dir={retrieval_timing.get('dir_level_time', 0):.3f} "
                      f"chunk={retrieval_timing.get('chunk_level_time', 0):.3f})")

            except Exception as e:
                print(f"⚠️ 样本跳过({factor}x, {_id[:20]}): {str(e)[:80]}")
                import traceback
                traceback.print_exc()
                continue

                    
        if os.path.exists(output_path):
            write_summary(output_path, summary_path, factor)
            print(f"📊 因子 {factor}x 汇总已写入: {summary_path}")

    print("\n✅ 可扩展性实验完成!")


if __name__ == "__main__":
                  
    DATA_PATH = "/root/dir-sem-rag/longbench/data_recall/hotpotqa_recall_clean.jsonl"
    OUT_DIR = "./dirrag_trae_scalability_results_hotpotqa_1000_with_solo"

                  
    FACTORS = [1]
    NUM_QUERIES = 30                             
    RETRIEVAL_REPEATS = 5                               
    SEED = 42                                 

                   
    CHUNK_SIZE = 100
    OVERLAP = 32

                  
    TOP_D = 5
    TOP_K = 5

                          
    NER_MODEL = "/root/dir-sem-rag/bert-based-ner"
    NER_BATCH_SIZE = 64

                  
    samples = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            samples.append({
                "_id": raw.get("_id", str(hash(raw.get("question", "")))),
                "question": raw.get("question", raw.get("input", "")),
                "passages": raw.get("passages", []),
            })

    print(f"📚 共加载 {len(samples)} 个 HotpotQA 样本")

                      
    run_scalability(
        samples=samples,
        factors=FACTORS,
        num_queries=NUM_QUERIES,
        seed=SEED,
        out_dir=OUT_DIR,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        top_d=TOP_D,
        top_k=TOP_K,
        ner_model=NER_MODEL,
        ner_batch_size=NER_BATCH_SIZE,
        retrieval_repeats=RETRIEVAL_REPEATS,
    )
