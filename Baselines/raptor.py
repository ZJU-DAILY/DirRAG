                       

import os
import sys
import json
import re
import time
import string
import threading
from typing import List, Optional, Dict, Tuple, Set

import numpy as np

from raptor import (
    BaseSummarizationModel,
    BaseQAModel,
    BaseEmbeddingModel,
    RetrievalAugmentationConfig,
    RetrievalAugmentation,
)

                                                                             
MODEL_PATH = "/root/dir-sem-rag/Qwen2.5-7B-Instruct"

sys.path.insert(0, "/root/dir-sem-rag/gte-Qwen2-1.5B-instruct")
sys.path.insert(1, "/root/dir-sem-rag/longbench/test/dir_rag_section_chunk_claude")
from embedding import EmbeddingModel
from llm import LocalLLMClient

EMBEDDING_KEY = "EMB"                                                    


def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    return " ".join(s.split())


def normalize_for_match(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    return " ".join(s.split())


QA_PROMPT = """Answer the question using the provided context or based on your own knowledge.

Context:

{context}

Question: {question}

Instructions:

1. Give the shortest correct answer phrase.

2. If you do NOT know the answer or the context does NOT contain enough information to answer the question, respond with exactly: "I don't know". Please do NOT randomly guess the answer.

3. Output ONLY the answer, nothing else (no explanation, no JSON, no prefix)."""

                                                              
SUMMARY_PROMPT = """Summarize the following text segments into a single concise paragraph.
Preserve key facts and do not add information not present in the text.
Keep the summary to approximately {max_tokens} words.

Text:
{context}

Summary:"""


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

    def snapshot(self) -> dict:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "llm_call_count": self.call_count,
        }


_TIME_LOCK = threading.Lock()
_TIMERS = {"chunking_time": 0.0, "embedding_time": 0.0, "summarization_time": 0.0}


def reset_timers():
    with _TIME_LOCK:
        for k in _TIMERS:
            _TIMERS[k] = 0.0


def _add_time(key: str, dt: float):
    with _TIME_LOCK:
        _TIMERS[key] += dt


def install_split_text_timer():
    import raptor.tree_builder as _tb
    if getattr(_tb, "_orig_split_text_timer", None) is None:
        _tb._orig_split_text_timer = _tb.split_text

        def _timed_split_text(*args, **kwargs):
            t0 = time.time()
            result = _tb._orig_split_text_timer(*args, **kwargs)
            _add_time("chunking_time", time.time() - t0)
            return result

        _tb.split_text = _timed_split_text


class MyLocalEmbedding(BaseEmbeddingModel):

    def __init__(self):
        print("📦 Initializing EmbeddingModel ...")
        self.model = EmbeddingModel()
        print("✅ Embedding model loaded.")

    def create_embedding(self, text: str):
        t0 = time.time()
        emb = self.model.encode([text])[0]
        _add_time("embedding_time", time.time() - t0)
        return emb


class MyLocalSummarizer(BaseSummarizationModel):

    def __init__(self, client: TrackedLLM):
        self.client = client

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        prompt = SUMMARY_PROMPT.format(context=context, max_tokens=max_tokens)
        t0 = time.time()
        try:
            response = self.client.chat(prompt)
        finally:
            _add_time("summarization_time", time.time() - t0)
        return response


class MyLocalQA(BaseQAModel):

    def __init__(self, client: TrackedLLM):
        self.client = client

    def answer_question(self, context: str, question: str) -> str:
        prompt = QA_PROMPT.format(context=context, question=question)
        return self.client.chat(prompt)


def build_passages_by_title(passages: List[dict]) -> Dict[str, List[str]]:
    pbt: Dict[str, List[str]] = {}
    for p in passages:
        title = p.get("title", "")
        if title and title not in pbt:
            pbt[title] = p.get("sentence", [])
    return pbt


def compute_recall_mrr_text(
    supporting_facts: List[dict],
    passages_by_title: Dict[str, List[str]],
    unit_texts: List[str],
) -> dict:
    zero = {"recall_at_k": 0.0, "mrr": 0.0, "hit_count": 0,
            "total_evidence": 0, "first_hit_rank": 0}

    gold: Dict[Tuple[str, int], str] = {}
    for sf in supporting_facts:
        title = sf.get("title", "")
        try:
            sid = int(sf.get("sent_id", -1))
        except Exception:
            sid = -1
        if not title or sid < 0:
            continue
        key = (title, sid)
        if key in gold:
            continue
        sentences = passages_by_title.get(title, [])
        if not sentences or sid >= len(sentences):
            continue
        gs = normalize_for_match(sentences[sid])
        if gs:
            gold[key] = gs

    total = len(gold)
    if total == 0 or not unit_texts:
        out = dict(zero)
        out["total_evidence"] = total
        return out

    norm_units = [normalize_for_match(u) for u in unit_texts]

    hit = 0
    first_rank = 0
    matched: Set[Tuple[str, int]] = set()

    for rank, unit in enumerate(norm_units, start=1):
        for key, gs in gold.items():
            if key in matched:
                continue
            if gs in unit:
                matched.add(key)
                hit += 1
        if first_rank == 0 and hit > 0:
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
    top_k: int = 5,
):
    print("📦 初始化 EmbeddingModel ...")
    embedding_model = MyLocalEmbedding()
    print("✅ Embedding model loaded.")

    install_split_text_timer()

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        print("✅ Tokenizer loaded for token counting.")
    except Exception:
        print("⚠️ Tokenizer 不可用，使用词数估算 token。")

    if not isinstance(local_llm, TrackedLLM):
        local_llm = TrackedLLM(local_llm, tokenizer=tokenizer)

                              
    raptor_config = RetrievalAugmentationConfig(
        summarization_model=MyLocalSummarizer(local_llm),
        qa_model=MyLocalQA(local_llm),
        embedding_model=embedding_model,
        tb_max_tokens=chunk_size,
        tb_summarization_length=100,
        tr_top_k=top_k,
    )

                                                                
    max_context_tokens = top_k * chunk_size

                                                                            
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
    print(f"🚀 HotpotQA RAPTOR | 分层摘要树基线 | 剩余: {total}")
    print(f"✅ chunk_size={chunk_size}  top_k={top_k}  "
          f"max_context_tokens={max_context_tokens}")
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
                                                              
            parts: List[str] = []
            for p in passages:
                title = p.get("title", "")
                sentences = p.get("sentence", [])
                body = " ".join(sentences) if sentences else p.get("content", "")
                if body.strip():
                    parts.append(f"{title}\n{body}")
            raw_text = "\n\n".join(parts)

            passages_by_title = build_passages_by_title(passages)

                                               
            RA = RetrievalAugmentation(config=raptor_config)

                                                                        
            reset_timers()
            local_llm.reset()
            t_index = time.time()
            RA.add_documents(raw_text)
            indexing_time = time.time() - t_index

            chunking_time = _TIMERS["chunking_time"]
            embedding_time = _TIMERS["embedding_time"]
            summarization_time = _TIMERS["summarization_time"]
            clustering_time = max(
                0.0, indexing_time - chunking_time - embedding_time - summarization_time
            )

            index_tokens = local_llm.snapshot()                             

            tree = RA.tree
            num_chunks = len(tree.layer_to_nodes.get(0, []))
            num_nodes = len(tree.all_nodes)
            num_layers = tree.num_layers
            index_size_mb = 0.0
            for node in tree.all_nodes.values():
                try:
                    index_size_mb += np.asarray(
                        node.embeddings[EMBEDDING_KEY]
                    ).nbytes
                except Exception:
                    continue
            index_size_mb = index_size_mb / (1024 * 1024)

                                                                       
            local_llm.reset()                             
            t_ret = time.time()
            context, layer_info = RA.retrieve(
                question,
                top_k=top_k,
                max_tokens=max_context_tokens,
                collapse_tree=True,
                return_layer_information=True,
            )
            retrieval_time = time.time() - t_ret

            selected_nodes = [tree.all_nodes[li["node_index"]] for li in layer_info]
            node_indices = [li["node_index"] for li in layer_info]
            unit_texts = [n.text for n in selected_nodes]

                                                                     
            prompt = QA_PROMPT.format(context=context, question=question)
            t_ans = time.time()
            llm_result = RA.qa_model.answer_question(context, question)
            answer_time = time.time() - t_ans

            e2e_latency = retrieval_time + answer_time                      
            token_cost = local_llm.snapshot()

            answer = (llm_result or "").strip()
            is_sufficient = normalize_answer(answer) != normalize_answer("I don't know")
            pred = answer if is_sufficient else "I don't know"

                                                                       
            recall_m = compute_recall_mrr_text(
                supporting_facts, passages_by_title, unit_texts
            ) if supporting_facts else {
                "recall_at_k": 0.0, "mrr": 0.0, "hit_count": 0,
                "total_evidence": 0, "first_hit_rank": 0,
            }

            is_correct = normalize_answer(pred) == normalize_answer(gold)

            used_chunk_info = [
                {
                    "chunk_id": li["node_index"],
                    "article_title": "",                                   
                    "index_in_article": li["layer_number"],
                    "text_preview": selected_nodes[i].text[:200],
                    "layer": li["layer_number"],
                    "is_summary": li["layer_number"] > 0,
                }
                for i, li in enumerate(layer_info)
            ]

            iteration_details = [{
                "iteration": 1,
                "retrieved_dir_ids": [],
                "retrieved_chunk_ids": node_indices,
                "expanded_chunk_ids": node_indices,               
                "context": context,
                "prompt": prompt,
                "llm_raw_response": answer,
                "isSufficient": is_sufficient,
                "answer": pred,
                "reasoning": "",
                "new_keywords": [],
                "new_entities": [],
                "used_chunk_info": used_chunk_info,
                "retrieval_time_s": round(retrieval_time, 4),
                "retrieved_layer_info": layer_info,
            }]

            result_item = {
                "id": _id,
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred,
                "is_correct": is_correct,
                "metrics": {
                    "num_chunks": num_chunks,
                    "num_directories": 0,
                    "top_d": 0,
                    "top_k": top_k,
                    "indexing_time_s": round(indexing_time, 4),
                    "indexing_time_breakdown": {
                        "total_time": round(indexing_time, 4),
                        "chunking_time": round(chunking_time, 4),
                        "embedding_time_total": round(embedding_time, 4),
                        "bm25_time_total": 0.0,
                        "entity_time_total": 0.0,
                        "clustering_time_total": round(clustering_time, 4),
                        "summarization_time_total": round(summarization_time, 4),
                    },
                    "retrieval_time_s": round(retrieval_time, 4),
                    "e2e_latency_s": round(e2e_latency, 4),
                    "token_cost": {
                        "input_tokens": token_cost["input_tokens"],
                        "output_tokens": token_cost["output_tokens"],
                        "total_tokens": token_cost["total_tokens"],
                    },
                    "indexing_token_cost": index_tokens,
                    "retrieval_token_cost": {                                       
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "llm_call_count": token_cost["llm_call_count"],
                    "iterations": 1,
                    "index_size_mb": round(index_size_mb, 3),
                    "recall_at_k": recall_m["recall_at_k"],
                    "mrr": recall_m["mrr"],
                    "hit_count": recall_m["hit_count"],
                    "total_evidence": recall_m["total_evidence"],
                    "first_hit_rank": recall_m["first_hit_rank"],
                                   
                    "num_nodes": num_nodes,
                    "num_layers": num_layers,
                    "num_summary_nodes": num_nodes - num_chunks,
                    "max_context_tokens": max_context_tokens,
                    "context_node_count": len(selected_nodes),
                },
                "iteration_details": iteration_details,
            }

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

            print(f"A: {pred} | {'✅' if is_correct else '❌'} Gold: {gold} | "
                  f"E2E={e2e_latency:.3f}s | "
                  f"Index={indexing_time:.3f}s | "
                  f"Retrieval={retrieval_time:.3f}s | "
                  f"LLMCalls={token_cost['llm_call_count']} | "
                  f"IndexTokens={index_tokens['total_tokens']} | "
                  f"AnswerTokens={token_cost['total_tokens']} | "
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
    OUTPUT_PATH = "./raptor_hotpotqa_results.jsonl"

    MAX_SAMPLES = None
    CHUNK_SIZE = 100
    TOP_K = 5

    local_llm = LocalLLMClient(model_path=MODEL_PATH)

    samples: List[dict] = []
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
        top_k=TOP_K,
    )
