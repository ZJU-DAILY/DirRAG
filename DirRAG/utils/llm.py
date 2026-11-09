import os
import sys
import json
import re
import torch
from typing import Optional
LLM_MAX_NEW_TOKENS = 8192
                                                         
                                          
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

torch.cuda.empty_cache()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

                                                
AGENT_PATH = "/root/dir-sem-rag/Qwen2.5-7B-Instruct"
MODEL_PATH = "/root/dir-sem-rag/Qwen2.5-7B-Instruct"
                                                 

if AGENT_PATH not in sys.path:
    sys.path.append(AGENT_PATH)

from qwenModel import QwenAgent

                                                        
class LocalLLMClient:
    def __init__(self, model_path: str = MODEL_PATH):
                                             
        self._model_path = model_path
        self.agent = None
        self._load()

    def _load(self):
        if self.agent is not None:
            return
        print(f"🚀 加载 LLM: {self._model_path}")
                        
        self.agent = QwenAgent(model_path=self._model_path)
        print("✅ LLM 已就绪")

    def destroy(self):
        if self.agent is not None:
            self.agent.destroy()
            self.agent = None
                                      
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🔥 LocalLLMClient destroyed, GPU cache cleared.")

    def ensure_loaded(self):
        if self.agent is None:
            self._load()

    def is_loaded(self) -> bool:
        return self.agent is not None

    def chat(self, prompt: str, max_new_tokens: int = LLM_MAX_NEW_TOKENS) -> str:
        self.ensure_loaded()
        try:
            response = self.agent.answer(prompt, max_new_tokens=max_new_tokens)

                            
            torch.cuda.empty_cache()
            return response

        except Exception as exc:
            print(f"❌ LLM error: {exc}")
            torch.cuda.empty_cache()
            return "ERROR"


def llm_judge(
    llm:      LocalLLMClient,
    question: str,
    context:  str,
) -> dict:
    prompt = (
        "You are a question-answering assistant. "
        "Given the question and context below, determine whether the context "
        "contains enough information to answer the question.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Respond ONLY with a JSON object in the following format "
        "(no markdown fences, no extra text):\n"
        '{"is_sufficient": true or false, '
        '"answer": "your final answer here", '
        '"reasoning": "explain your reasoning step by step", '
        '"key_context": "the most important evidence sentences from context", '
        '"missing_keywords": ["keyword1", "keyword2"]}'
        "\nIf is_sufficient is false, keep answer and key_context as empty strings."
    )

    raw = llm.chat(prompt)

    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        res = json.loads(cleaned)

        if res.get("is_sufficient"):
            combined = (
                f"Answer: {res.get('answer', '')}\n\n"
                f"Reasoning: {res.get('reasoning', '')}\n\n"
                f"Key Context: {res.get('key_context', '')}"
            )
            res["answer"] = combined
        return res
    except Exception:
        return {"is_sufficient": False, "answer": "", "missing_keywords": []}


if __name__ == "__main__":
    llm = LocalLLMClient()
    print(llm.chat("Who founded Apple Inc.?"))

    result = llm_judge(
        llm,
        question="Who founded Apple?",
        context="Apple Inc. was founded by Steve Jobs in 1976.",
    )
    print("Judge result:", result)