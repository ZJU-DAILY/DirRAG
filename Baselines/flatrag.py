                       

import os
import sys
import json
import re
import time
import string
import gc
from typing import List, Dict, Optional, Tuple, Set, Any

import numpy as np

                                                                             
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


def parse_llm_json(response: str) -> Optional[Any]:
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

    def chat(self, prompt: str) -> str:
        self.call_count += 1
        response = self.llm_client.chat(prompt)
        if self.tokenizer is not None:
            try:
                self.total_input_tokens += len(self.tokenizer.encode(prompt))
                self.total_output_tokens += len(self.tokenizer.encode(response))
            except Exception:
                self.total_input_tokens += len(prompt.split())
                self.total_output_tokens += len(response.split())
        else:
            self.total_input_tokens += len(prompt.split())
            self.total_output_tokens += len(response.split())
        return response

    def reset(self):
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


class FlatRAG:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_client: TrackedLLM,
        chunk_size: int = 100,
        overlap: int = 32,
        top_k: int = 10,
    ):
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

        self.chunks: List[Chunk] = []
        self.chunk_id_to_idx: Dict[int, int] = {}
        self._embeddings: List[np.ndarray] = []                  
        self._chunk_id_counter = 0

                                  
    def reset(self):
        self.chunks.clear()
        self.chunk_id_to_idx.clear()
        self._embeddings.clear()
        self._chunk_id_counter = 0
        gc.collect()

    def build_index(self, passages: List[dict]) -> dict:
        t0 = time.time()
        timing = {}
        self.reset()

                    
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

    def get_chunk_count(self) -> int:
        return len(self.chunks)

    def get_index_size_mb(self) -> float:
        total = sum(e.nbytes for e in self._embeddings)
        return total / (1024 * 1024)

                                
    def retrieve(self, query: str) -> List[int]:
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
        return [cid for cid, _ in scored[:self.top_k]]

                                  
    def answer_question(self, question: str) -> Tuple[str, dict]:
        t_start = time.time()
        self.llm_client.reset()

        t_ret = time.time()
        top_chunk_ids = self.retrieve(question)
        retrieval_time = time.time() - t_ret

        llm_result = self._answer_with_llm(question, top_chunk_ids)
        pred = llm_result.get("answer", "I don't know")
        if not llm_result.get("isSufficient", False):
            pred = "I don't know"

        e2e = time.time() - t_start

        iteration_details = [{
            "iteration": 1,
            "retrieved_dir_ids": [],
            "retrieved_chunk_ids": top_chunk_ids,
            "expanded_chunk_ids": top_chunk_ids,              
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
        }]

        metrics = {
            "token_cost": {
                "input_tokens": self.llm_client.total_input_tokens,
                "output_tokens": self.llm_client.total_output_tokens,
                "total_tokens": self.llm_client.total_tokens,
            },
            "llm_call_count": self.llm_client.call_count,
            "e2e_latency_s": round(e2e, 4),
            "retrieval_time_s": round(retrieval_time, 4),
            "iterations": 1,
            "iteration_details": iteration_details,
        }
        return pred, metrics

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

    flat_rag = FlatRAG(
        embedding_model=embedding_model,
        llm_client=local_llm,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
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
    print(f"🚀 HotpotQA Flat RAG | 纯向量检索基线 | 剩余: {total}")
    print(f"✅ chunk_size={chunk_size}  top_k={top_k}")
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
            flat_rag.reset()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

                          
            timing = flat_rag.build_index(passages)
            total_chunks = flat_rag.get_chunk_count()
            index_size = flat_rag.get_index_size_mb()

            print(f"📦 文本块: {total_chunks}, "
                  f"IndexTime={timing['total_time']:.3f}s, "
                  f"IndexSize={index_size:.2f}MB")

                           
            pred, metrics = flat_rag.answer_question(question)

                                  
            retrieved = metrics["iteration_details"][0]["expanded_chunk_ids"]
            recall_m = flat_rag.compute_recall_mrr(supporting_facts, retrieved)\
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
                        "chunking_time": timing["chunking_time"],
                        "embedding_time_total": timing["embedding_time"],
                        "bm25_time_total": 0.0,
                        "entity_time_total": 0.0,
                    },
                    "retrieval_time_s": metrics["retrieval_time_s"],
                    "e2e_latency_s": metrics["e2e_latency_s"],
                    "token_cost": metrics["token_cost"],
                    "indexing_token_cost": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "llm_call_count": 0,
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
    OUTPUT_PATH = "./flatrag_dense_hotpotqa_results_trae.jsonl"

    MAX_SAMPLES = None
    CHUNK_SIZE = 100
    OVERLAP = 32
    TOP_K = 3

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
    )
