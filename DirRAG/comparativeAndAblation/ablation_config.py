"""
ablation_config.py
==================
DirRAG v3.0 消融实验配置文件

使用方式：
    from ablation_config import get_config, ABLATION_VARIANTS
    cfg = get_config("full_dirrag")          # 直接按变体名取
    cfg = get_config("wo_section_routing")
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import os


# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────  CONFIG DATACLASS  ──────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DirRAGConfig:
    # ── 变体标识 ───────────────────────────────────────────────────────────
    variant_name: str = "full_dirrag"
    description:  str = "Full DirRAG with all components enabled"

    # ── 架构开关 ───────────────────────────────────────────────────────────
    use_section_routing: bool = True
    # NOTE: use_chunk_retrieval 不参与消融（故意不暴露此开关），
    #       Chunk Retrieval 始终开启，见 Q3 分析。

    # ── 迭代检索 ───────────────────────────────────────────────────────────
    use_iteration: bool  = True    # False → 强制 MAX_ITER = 1
    max_iter:      int   = 5       # use_iteration=True 时生效

    # ── Section 嵌入策略 ───────────────────────────────────────────────────
    title_weight:   float = 0.7    # title_weight + summary_weight 应 = 1.0
    summary_weight: float = 0.3
    summary_chars:  int   = 200

    # ── Section 评分权重 ───────────────────────────────────────────────────
    sec_alpha: float = 0.5   # semantic similarity
    sec_beta:  float = 0.2   # BM25
    sec_gamma: float = 0.2   # entity match
    sec_delta: float = 0.1   # depth reward

    # ── Chunk 评分权重 ─────────────────────────────────────────────────────
    chk_alpha: float = 0.5   # semantic similarity
    chk_beta:  float = 0.2   # BM25
    chk_gamma: float = 0.2   # entity match
    chk_delta: float = 0.1   # position prior（参与消融但优先级低）

    # ── 检索超参 ───────────────────────────────────────────────────────────
    section_top_k:   int = 15
    chunk_anchor_n:  int = 30 # 这个要根据数据集调整，避免一次性召回了所有的chunk，这样做实验没意义
    neighbor_window: int = 2

    # ── BM25 超参 ──────────────────────────────────────────────────────────
    bm25_k1: float = 1.5
    bm25_b:  float = 0.75

    # ── 实体模糊匹配阈值 ───────────────────────────────────────────────────
    entity_fuzzy_threshold: float = 0.85

    # ── 已用 Section 惩罚 ──────────────────────────────────────────────────
    used_section_penalty: float = 0.1

    # ── I/O 路径 ───────────────────────────────────────────────────────────
    input_path:  str = "/root/dir-sem-rag/longbench/test/clean_qasper/qasper_cleaned.jsonl"
    output_path: str = "./results/full_dirrag.jsonl"

    # ── 数据采样（消融实验用固定子集）─────────────────────────────────────
    sample_size:   Optional[int] = 500   # None → 跑全量数据集
    sample_seed:   int           = 42

    # ── 设备 ───────────────────────────────────────────────────────────────
    cuda_visible_devices: str = "0,1"

    # ── 衍生属性（自动计算，不需手动设置）────────────────────────────────
    def __post_init__(self):
        # use_iteration=False 时强制 max_iter=1
        if not self.use_iteration:
            self.max_iter = 1

        # 确保权重归一化提示（不强制，允许研究者故意调低总权重）
        sec_sum = self.sec_alpha + self.sec_beta + self.sec_gamma + self.sec_delta
        chk_sum = self.chk_alpha + self.chk_beta + self.chk_gamma + self.chk_delta
        if abs(sec_sum - 1.0) > 1e-6 or abs(chk_sum - 1.0) > 1e-6:
            print(
                f"[Config Warning] 权重之和不为 1.0 → "
                f"SEC={sec_sum:.3f}, CHK={chk_sum:.3f}。"
                f"消融实验中将某项置 0 时，其余权重保持原值（不重归一化），"
                f"这是有意为之，以便单独衡量每个信号的边际贡献。"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Optional[str] = None):
        save_path = path or self.output_path.replace(".jsonl", "_config.json")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[Config] 配置已保存至 {save_path}")

    @classmethod
    def from_dict(cls, d: dict) -> "DirRAGConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str) -> "DirRAGConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────  PRE-DEFINED ABLATION VARIANTS  ─────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _make(variant_name: str, description: str, **overrides) -> DirRAGConfig:
    """工厂函数：从 Full DirRAG 出发，仅覆盖指定字段。"""
    return DirRAGConfig(
        variant_name = variant_name,
        description  = description,
        output_path  = f"./results/{variant_name}.jsonl",
        **overrides,
    )


ABLATION_VARIANTS: dict[str, DirRAGConfig] = {

    # ── Baseline ───────────────────────────────────────────────────────────
    "full_dirrag": DirRAGConfig(
        variant_name = "full_dirrag",
        description  = "Full DirRAG：所有组件全部开启（Baseline）",
        output_path  = "./results/full_dirrag.jsonl",
    ),

    # ── 一、架构消融 ────────────────────────────────────────────────────────
    "wo_section_routing": _make(
        "wo_section_routing",
        "W/O Section Routing：跳过目录路由，直接在所有 chunk 中检索",
        use_section_routing = False,
    ),

    # ── 二、迭代消融 ────────────────────────────────────────────────────────
    "wo_iteration": _make(
        "wo_iteration",
        "W/O Iteration：单轮检索，无反思扩展（MAX_ITER=1）",
        use_iteration = False,
    ),

    # ── 三、Section 嵌入策略消融 ────────────────────────────────────────────
    "title_only": _make(
        "title_only",
        "Title Only：Section 嵌入仅使用标题，不加 summary",
        title_weight   = 1.0,
        summary_weight = 0.0,
    ),
    "summary_only": _make(
        "summary_only",
        "Summary Only：Section 嵌入仅使用内容摘要，不加 title",
        title_weight   = 0.0,
        summary_weight = 1.0,
    ),

    # ── 四、深度奖励消融（层次信号）────────────────────────────────────────
    "wo_depth_reward": _make(
        "wo_depth_reward",
        "W/O Depth Reward：去掉 Section 评分中的层次深度信号（sec_delta=0）",
        sec_delta = 0.0,
    ),

    # ── 五、混合信号消融（BM25 + Entity 合并为一个变体）────────────────────
    "semantic_only": _make(
        "semantic_only",
        "Semantic Only：Section 和 Chunk 均只保留语义相似度，去掉 BM25 和 Entity 信号",
        sec_beta  = 0.0,
        sec_gamma = 0.0,
        chk_beta  = 0.0,
        chk_gamma = 0.0,
    ),
}


def get_config(variant: str) -> DirRAGConfig:
    """
    按变体名获取配置。

    Parameters
    ----------
    variant : str
        ABLATION_VARIANTS 中的键名，例如 "full_dirrag"、"wo_section_routing"。

    Returns
    -------
    DirRAGConfig

    Raises
    ------
    KeyError : variant 不存在时抛出，并打印所有可用变体名。
    """
    if variant not in ABLATION_VARIANTS:
        available = "\n  ".join(ABLATION_VARIANTS.keys())
        raise KeyError(
            f"未知消融变体 '{variant}'。\n可用变体：\n  {available}"
        )
    return ABLATION_VARIANTS[variant]


def list_variants():
    """打印所有已注册的消融变体及其描述。"""
    print("\n" + "═" * 70)
    print("  DirRAG 消融变体一览")
    print("═" * 70)
    for name, cfg in ABLATION_VARIANTS.items():
        print(f"  [{name}]")
        print(f"    {cfg.description}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    list_variants()