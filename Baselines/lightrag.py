                       

import os
import sys
import json
import re
import time
import string
import gc
import hashlib
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set, Any

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


def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def topk_cosine(query_vec: np.ndarray, mat_norm: np.ndarray, k: int) -> List[Tuple[int, float]]:
    if mat_norm.size == 0:
        return []
    q = query_vec / (np.linalg.norm(query_vec) or 1.0)
    scores = mat_norm @ q
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


TUPLE_DELIM = "<|>"
REC_DELIM = "###"
COMPLETE_DELIM = "<|COMPLETE|>"
ENTITY_TYPES = "organization, person, geo, event"
DESC_SEP = " | "

                                                                              
ENTITY_EXTRACTION_PROMPT = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-

1. Identify all entities. For each identified entity, extract the following information:

- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:

- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_keywords><|><relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"<|><high_level_keywords>)
4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **###** as the list delimiter.
5. When finished, output <|COMPLETE|>

-Real Data-
Entity_types: {entity_types}
Text: {input_text}

Output:"""

GLEANING_PROMPT = """Some entities and relationships may have been missed in the previous extraction.
Add ONLY the missing ones, using exactly the same output format and delimiters (### as list delimiter, <|COMPLETE|> when finished).
If nothing is missing, output only <|COMPLETE|>.

Output:"""

                                                                              
KEYWORD_PROMPT = """---Role---
You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---
Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.
*Keywords Generate Instruction Prompt*

- Output the keywords in JSON format.
- The JSON should have two keys:
- "high_level_keywords" for overarching concepts or themes.
- "low_level_keywords" for specific entities or details.

-Examples-
Example 1:
Query: "How does international trade influence global economic stability?"
Output: {{"high_level_keywords": ["International trade", "Global economic stability", "Economic impact"], "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]}}

Example 2:
Query: "What are the environmental consequences of deforestation on biodiversity?"
Output: {{"high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"], "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]}}

Example 3:
Query: "What is the role of education in reducing poverty?"
Output: {{"high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"], "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]}}

-Real Data-
Query: {query}
Output:"""


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


class LightRAG:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_client: TrackedLLM,
        chunk_size: int = 100,
        overlap: int = 32,
        top_k: int = 5,
        max_gleaning: int = 0,
        summarize_threshold: int = 999,
        cache_dir: str = "./lightrag_cache",
    ):
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.max_gleaning = max_gleaning
        self.summarize_threshold = summarize_threshold
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.reset()

    def reset(self):
             
        self.chunks: List[Chunk] = []
        self.chunk_id_to_idx: Dict[int, int] = {}
        self._chunk_id_counter = 0
        self._embeddings: List[np.ndarray] = []

              
        self.entities: Dict[str, Dict[str, Any]] = {}
                                                                 
        self.relations: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.adj: Dict[str, Set[str]] = defaultdict(set)

              
        self.entity_names: List[str] = []
        self.entity_mat: np.ndarray = np.zeros((0, 0), dtype=np.float32)
                                                     
        self.relation_entries: List[Tuple[Tuple[str, str, str], str]] = []
        self.relation_mat: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.chunk_mat: np.ndarray = np.zeros((0, 0), dtype=np.float32)

                     
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        gc.collect()

                                  
    def build_index(self, passages: List[dict], extract_only: bool = False) -> dict:
        t0 = time.time()
        timing = {}
        self.reset()
                                         
        self.llm_client.indexing_call_count = 0
        self.llm_client.indexing_input_tokens = 0
        self.llm_client.indexing_output_tokens = 0

                                                           
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
            timing["total_time"] = round(time.time() - t0, 4)
            return timing

                                                                              
        if extract_only:
            t1 = time.time()
            for cid, chunk in enumerate(self.chunks):
                self._extract_from_chunk(cid, chunk.text)
            self._maybe_summarize_descriptions()
            timing["graph_extraction_time"] = round(time.time() - t1, 4)
            timing["cache_hits"] = self._cache_hits
            timing["cache_misses"] = self._cache_misses
            timing["embedding_time"] = 0.0
            timing["total_time"] = round(time.time() - t0, 4)
            return timing

                                 
        t1 = time.time()
        try:
            texts = [c.text for c in self.chunks]
            embs = self.embedding_model.encode(texts)
            self._embeddings = [np.array(e, dtype=np.float32) for e in embs]
            self.chunk_mat = normalize_matrix(np.vstack(self._embeddings))
        except Exception as e:
            print(f"⚠️ Embedding 失败: {str(e)[:50]}")
        timing["embedding_time"] = round(time.time() - t1, 4)

                                                             
        t1 = time.time()
        for cid, chunk in enumerate(self.chunks):
            self._extract_from_chunk(cid, chunk.text)
        self._maybe_summarize_descriptions()
        self._build_vector_index()
        timing["graph_extraction_time"] = round(time.time() - t1, 4)
        timing["cache_hits"] = self._cache_hits
        timing["cache_misses"] = self._cache_misses

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

    def _cache_key(self, chunk_text: str) -> str:
        return hashlib.md5(chunk_text.strip().encode("utf-8")).hexdigest()

    def _load_cache(self, chunk_text: str) -> Optional[List[str]]:
        key = self._cache_key(chunk_text)
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                                   
            if data.get("chunk_text", "").strip() != chunk_text.strip():
                return None
            return data.get("responses", [])
        except Exception:
            return None

    def _save_cache(self, chunk_text: str, responses: List[str]) -> None:
        key = self._cache_key(chunk_text)
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "chunk_text": chunk_text,
                    "responses": responses,
                }, f, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 缓存写入失败: {str(e)[:50]}")

    def _extract_from_chunk(self, cid: int, chunk_text: str) -> None:
                        
        cached_responses = self._load_cache(chunk_text)
        if cached_responses is not None:
            self._cache_hits += 1
            for resp in cached_responses:
                self._ingest_records(self._parse_records(resp), cid)
            return

                              
        self._cache_misses += 1
        all_responses: List[str] = []

        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_types=ENTITY_TYPES, input_text=chunk_text,
        )
        raw = self.llm_client.chat(prompt, stage="indexing")
        all_responses.append(raw)
        history = prompt + "\n" + raw
        self._ingest_records(self._parse_records(raw), cid)

        for _ in range(self.max_gleaning):
            glean_prompt = history + "\n\n" + GLEANING_PROMPT
            glean = self.llm_client.chat(glean_prompt, stage="indexing")
            if not glean or glean.replace(COMPLETE_DELIM, "").strip() == "":
                break
            all_responses.append(glean)
            self._ingest_records(self._parse_records(glean), cid)
            history = glean_prompt + "\n" + glean

                      
        self._save_cache(chunk_text, all_responses)

    @staticmethod
    def _parse_records(raw: str) -> List[List[str]]:
        if not raw:
            return []
        raw = raw.replace(COMPLETE_DELIM, "")
        out: List[List[str]] = []
        for rec in raw.split(REC_DELIM):
            rec = rec.strip()
            if not rec:
                continue
            m = re.search(r"\((.*)\)", rec, flags=re.S)
            if m:
                rec = m.group(1)
            fields = [f.strip().strip('"').strip("'").strip() for f in rec.split(TUPLE_DELIM)]
                                                                            
            if len(fields) >= 3:
                out.append(fields)
        return out

    def _ingest_records(self, records: List[List[str]], cid: int) -> None:
        for fields in records:
            tag = fields[0].lower()
                                                                   
            if tag == "content_keywords":
                continue
            if tag == "entity" and len(fields) >= 4:
                name = fields[1].upper().strip()
                if not name:
                    continue
                node = self.entities.setdefault(
                    name, {"type": fields[2] or "unknown", "descriptions": [], "chunks": set()}
                )
                if fields[3]:
                    node["descriptions"].append(fields[3])
                node["chunks"].add(cid)

            elif tag == "relationship" and len(fields) >= 5:
                src, tgt = fields[1].upper().strip(), fields[2].upper().strip()
                if not src or not tgt or src == tgt:
                    continue
                for n in (src, tgt):
                    node = self.entities.setdefault(
                        n, {"type": "unknown", "descriptions": [], "chunks": set()}
                    )
                    node["chunks"].add(cid)

                                                 
                desc_raw = fields[3] or ""
                desc_norm = normalize_answer(desc_raw) if desc_raw else "_no_desc_"
                s, t = (src, tgt) if src <= tgt else (tgt, src)
                key = (s, t, desc_norm)
                try:
                    weight = float(fields[5]) if len(fields) >= 6 else 5.0
                except Exception:
                    weight = 5.0
                edge = self.relations.setdefault(
                    key, {"descriptions": [], "keywords": set(), "weight": 0.0, "chunks": set()}
                )
                if fields[3]:
                    edge["descriptions"].append(fields[3])
                for kw in re.split(r"[,;，、]", fields[4] or ""):
                    kw = kw.strip()
                    if kw:
                        edge["keywords"].add(kw)
                edge["weight"] = max(edge["weight"], weight)
                edge["chunks"].add(cid)
                self.adj[src].add(tgt)
                self.adj[tgt].add(src)

    def _merge_desc(self, kind: str, name: str, descs: List[str]) -> str:
        descs = [d for d in dict.fromkeys(descs) if d]
        if not descs:
            return ""
        if len(descs) < self.summarize_threshold:
            return DESC_SEP.join(descs)
                           
        return DESC_SEP.join(descs[:self.summarize_threshold])

    def _maybe_summarize_descriptions(self) -> None:
        for name, node in self.entities.items():
            node["description"] = self._merge_desc("entity", name, node["descriptions"])
            node.pop("descriptions", None)
        for key, edge in self.relations.items():
            edge["description"] = self._merge_desc(
                "relationship", f"{key[0]} - {key[1]}", edge["descriptions"]
            )
            edge.pop("descriptions", None)

    def _build_vector_index(self) -> None:
                
        if self.entities:
            self.entity_names = list(self.entities.keys())
            ent_texts = [
                f"{n}\n{self.entities[n].get('description', '')}"
                for n in self.entity_names
            ]
            try:
                self.entity_mat = normalize_matrix(
                    np.vstack([
                        np.array(e, dtype=np.float32)
                        for e in self.embedding_model.encode(ent_texts)
                    ])
                )
            except Exception as e:
                print(f"⚠️ 实体向量构建失败: {str(e)[:50]}")

                                                                   
        if self.relations:
            self.relation_entries = []
            rel_texts: List[str] = []
            for k, e in self.relations.items():
                keywords = sorted(e.get("keywords", []))
                if not keywords:
                                     
                    keywords = [e.get("description", "")[:50] or "_no_key_"]
                for kw in keywords:
                    self.relation_entries.append((k, kw))
                    rel_texts.append(
                        f"{kw}\t{k[0]}\n{k[1]}\n{e.get('description', '')}"
                    )
            try:
                self.relation_mat = normalize_matrix(
                    np.vstack([
                        np.array(e, dtype=np.float32)
                        for e in self.embedding_model.encode(rel_texts)
                    ])
                )
            except Exception as e:
                print(f"⚠️ 关系向量构建失败: {str(e)[:50]}")

                                
    def get_chunk_count(self) -> int:
        return len(self.chunks)

    def get_index_size_mb(self) -> float:
        total = self.chunk_mat.nbytes + self.entity_mat.nbytes + self.relation_mat.nbytes
        return total / (1024 * 1024)

    def get_stats(self) -> dict:
        return {
            "num_chunks": len(self.chunks),
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
        }

    def _degree(self, name: str) -> int:
        return len(self.adj.get(name, ()))

                                
    def _extract_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        raw = self.llm_client.chat(KEYWORD_PROMPT.format(query=query), stage="retrieval")
        hl: List[str] = []
        ll: List[str] = []
        m = re.search(r"\{.*\}", raw or "", flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                hl = [str(x) for x in obj.get("high_level_keywords", []) if str(x).strip()]
                ll = [str(x) for x in obj.get("low_level_keywords", []) if str(x).strip()]
            except Exception:
                pass
        if not hl:
            hl = [query]
        if not ll:
            ll = [query]
        return hl, ll

    def _local_retrieve(self, ll_query: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        if not self.entity_names:
            return [], []
        try:
            qv = self.embedding_model.encode([ll_query])[0]
        except Exception:
            return [], []
        hits = topk_cosine(qv, self.entity_mat, self.top_k)
        seed_ents = [self.entity_names[i] for i, _ in hits]

                                         
        expanded_ents: Set[str] = set(seed_ents)
        for ent in seed_ents:
            for nbr in self.adj.get(ent, ()):
                expanded_ents.add(nbr)

                          
        cand = []
        for key, edge in self.relations.items():
            if key[0] in expanded_ents or key[1] in expanded_ents:
                rank = self._degree(key[0]) + self._degree(key[1])
                cand.append((rank, edge["weight"], key))
        cand.sort(key=lambda x: (-x[0], -x[1]))
        top_rels = [k for _, _, k in cand[:self.top_k]]

                                
        ents_ordered = list(dict.fromkeys(seed_ents + sorted(expanded_ents - set(seed_ents))))
        return ents_ordered[:self.top_k * 2], top_rels

    def _global_retrieve(self, hl_query: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        if not self.relation_entries:
            return [], []
        try:
            qv = self.embedding_model.encode([hl_query])[0]
        except Exception:
            return [], []
        hits = topk_cosine(qv, self.relation_mat, self.top_k)
                                                 
        rels: List[Tuple[str, str, str]] = []
        seen_keys: set = set()
        for i, _ in hits:
            rel_key = self.relation_entries[i][0]
            if rel_key not in seen_keys:
                seen_keys.add(rel_key)
                rels.append(rel_key)

                   
        seed_ents: List[str] = []
        for rk in rels:
            for n in (rk[0], rk[1]):
                if n not in seed_ents and n in self.entities:
                    seed_ents.append(n)

                            
        expanded_ents: Set[str] = set(seed_ents)
        for ent in seed_ents:
            for nbr in self.adj.get(ent, ()):
                expanded_ents.add(nbr)

        ents_ordered = list(dict.fromkeys(seed_ents + sorted(expanded_ents - set(seed_ents))))
        return ents_ordered[:self.top_k * 2], rels

    def _select_chunks(self, ents: List[str], rels: List[Tuple[str, str, str]]) -> List[int]:
        score: Dict[int, float] = defaultdict(float)
        for n in ents:
            for cid in self.entities.get(n, {}).get("chunks", ()):
                score[cid] += 1.0
        for k in rels:
                                                       
            for cid in self.relations.get(k, {}).get("chunks", ()):
                score[cid] += 1.5
        ordered = sorted(score.items(), key=lambda x: (-x[1], x[0]))
        return [cid for cid, _ in ordered if 0 <= cid < len(self.chunks)]

                                  
    def answer_question(self, question: str) -> Tuple[str, dict]:
        t_start = time.time()
        self.llm_client.reset()

                                                   
        t_ret = time.time()
        hl_kw, ll_kw = self._extract_keywords(question)
        ll_query = ", ".join(ll_kw)
        hl_query = ", ".join(hl_kw)

                                                      
        ents: List[str] = []
        rels: List[Tuple[str, str, str]] = []
        e, r = self._local_retrieve(ll_query)
        ents += e
        rels += r
        e, r = self._global_retrieve(hl_query)
        ents += e
        rels += r
        ents = list(dict.fromkeys(ents))
        rels = list(dict.fromkeys(rels))

                                                                   
        graph_chunk_ids = self._select_chunks(ents, rels)

                                           
        final_chunk_ids = graph_chunk_ids[:self.top_k]

        retrieval_time = time.time() - t_ret

                                        
        llm_result = self._answer_with_llm(question, final_chunk_ids, ents, rels)
        pred = llm_result.get("answer", "I don't know")
        if not llm_result.get("isSufficient", False):
            pred = "I don't know"

        e2e = time.time() - t_start

        iteration_details = [{
            "iteration": 1,
            "retrieved_dir_ids": [],
            "retrieved_chunk_ids": graph_chunk_ids,
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
                "hl_keywords": hl_kw,
                "ll_keywords": ll_kw,
                "entities": ents,
                "relations": [list(r) for r in rels],
            },
        }]

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
            "e2e_latency_s": round(e2e, 4),
            "retrieval_time_s": round(retrieval_time, 4),
            "iterations": 1,
            "iteration_details": iteration_details,
        }
        return pred, metrics

    def _answer_with_llm(self, question: str, chunk_ids: List[int],
                          ents: List[str], rels: List[Tuple[str, str, str]]) -> dict:
              
        ent_lines: List[str] = []
        for n in ents[:10]:
            if n in self.entities:
                desc = self.entities[n].get("description", "")
                ntype = self.entities[n].get("type", "unknown")
                deg = self._degree(n)
                ent_lines.append(f"  - {n} ({ntype}, degree={deg}): {desc}")

              
        rel_lines: List[str] = []
        for k in rels[:10]:
            if k in self.relations:
                desc = self.relations[k].get("description", "")
                kws = ", ".join(sorted(self.relations[k].get("keywords", [])))
                w = self.relations[k].get("weight", 0)
                rel_lines.append(f"  - {k[0]} --[{kws}]--> {k[1]} (w={w:.0f}): {desc}")

                              
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

                 
        full_context_parts: List[str] = []
        if ent_lines:
            full_context_parts.append("=== Entities ===\n" + "\n".join(ent_lines))
        if rel_lines:
            full_context_parts.append("=== Relationships ===\n" + "\n".join(rel_lines))
        if context_parts:
            full_context_parts.append("=== Source Text ===\n" + "\n\n".join(context_parts))
        context = "\n\n".join(full_context_parts) if full_context_parts else "(no context)"

        prompt = QA_PROMPT.format(context=context, question=question)

        try:
            resp = self.llm_client.chat(prompt)
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
    max_gleaning: int = 0,
    cache_dir: str = "./lightrag_cache",
                                                      
    extract_only: bool = False,
    extract_only_stats_path: str = "",
):
    print("📦 初始化 EmbeddingModel ...")
    embedding_model = EmbeddingModel()
    print("✅ Embedding model loaded. (LLM 与 embedding 常驻 GPU)")

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        print("✅ Tokenizer loaded for token counting.")
    except Exception:
        print("⚠️ Tokenizer 不可用，使用词数估算 token。")

    if not isinstance(local_llm, TrackedLLM):
        local_llm = TrackedLLM(local_llm, tokenizer=tokenizer)

    rag = LightRAG(
        embedding_model=embedding_model,
        llm_client=local_llm,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        max_gleaning=max_gleaning,
        cache_dir=cache_dir,
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
    print(f"🚀 HotpotQA LightRAG | 双层检索 + 知识图谱 | 剩余: {total}")
    print(f"✅ chunk_size={chunk_size}  top_k={top_k}  gleaning={max_gleaning}")
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

                          
            timing = rag.build_index(passages, extract_only=extract_only)
            total_chunks = rag.get_chunk_count()
            index_size = rag.get_index_size_mb()
            stats = rag.get_stats()

            print(f"📦 文本块: {total_chunks}, 实体: {stats['num_entities']}, "
                  f"关系: {stats['num_relations']} | "
                  f"IndexTime={timing['total_time']:.3f}s, "
                  f"IndexSize={index_size:.2f}MB | "
                  f"Cache: {timing.get('cache_hits', 0)} hit / {timing.get('cache_misses', 0)} miss")

                                                                 
            if extract_only:
                stats_item = {
                    "id": _id,
                    "question": question,
                    "metrics": {
                        "num_chunks": total_chunks,
                        "indexing_time_s": timing.get("total_time", 0),
                        "indexing_time_breakdown": timing,
                        "indexing_token_cost": {
                            "input_tokens": rag.llm_client.indexing_input_tokens,
                            "output_tokens": rag.llm_client.indexing_output_tokens,
                            "total_tokens": rag.llm_client.indexing_total_tokens,
                            "llm_call_count": rag.llm_client.indexing_call_count,
                        },
                    },
                }
                stats_path = extract_only_stats_path or output_path.replace(".jsonl", "_extract_only.jsonl")
                with open(stats_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(stats_item, ensure_ascii=False) + "\n")
                print(f"    📝 extract_only: LLM 提取完成, "
                      f"耗时={timing.get('total_time', 0):.3f}s, "
                      f"Token={rag.llm_client.indexing_total_tokens}")
                continue

                           
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
                        "graph_extraction_time": timing.get("graph_extraction_time", 0),
                        "bm25_time_total": 0.0,
                        "entity_time_total": 0.0,
                        "cache_hits": timing.get("cache_hits", 0),
                        "cache_misses": timing.get("cache_misses", 0),
                    },
                    "retrieval_time_s": metrics["retrieval_time_s"],
                    "e2e_latency_s": metrics["e2e_latency_s"],
                    "token_cost": metrics["token_cost"],
                    "retrieval_token_cost": metrics["retrieval_token_cost"],
                    "indexing_token_cost": {
                        "input_tokens": rag.llm_client.indexing_input_tokens,
                        "output_tokens": rag.llm_client.indexing_output_tokens,
                        "total_tokens": rag.llm_client.indexing_total_tokens,
                        "llm_call_count": rag.llm_client.indexing_call_count,
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
                "graph_stats": stats,
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
    OUTPUT_PATH = "./lightrag_hotpotqa_results.jsonl"

    MAX_SAMPLES = None
    CHUNK_SIZE = 100
    OVERLAP = 32
    TOP_K = 5
    MAX_GLEANING = 0
    CACHE_DIR = "./lightrag_cache"

                                            
    EXTRACT_ONLY = False
    EXTRACT_ONLY_STATS_PATH = "./lightrag_extract_only_stats.jsonl"

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
        max_gleaning=MAX_GLEANING,
        cache_dir=CACHE_DIR,
        extract_only=EXTRACT_ONLY,
        extract_only_stats_path=EXTRACT_ONLY_STATS_PATH,
    )
