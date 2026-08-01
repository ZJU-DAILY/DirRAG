# DirRAG: Leveraging Native Directory Hierarchies as Structural Priors for Retrieval-Augmented Generation
This repository hosts the official source code, datasets and experimental scripts for the paper **DirRAG: Leveraging Native Directory Hierarchies as Structural Priors for Retrieval-Augmented Generation**.

## Overview
The repository consists of two core modules:
1. **Custom Hierarchical QA Datasets**: Two newly constructed benchmarks built with inherent directory hierarchical structures, tailored for evaluating hierarchical retrieval-augmented generation methods.
2. **DirRAG Implementation**: Full experimental code for baseline comparisons, ablation studies, and scalability verification of the proposed DirRAG framework.

## Repository Structure
```
.
├── DirRAG
│   ├── comparativeAndAblation       # Scripts for ablation experiments & baseline comparison
│   │   ├── ablation_2wikimqa.py
│   │   ├── ablation_cloud_dir.py
│   │   ├── ablation_config.py
│   │   ├── ablation_hotpotqa.py
│   │   ├── ablation_medi.py
│   │   ├── ablation_musique.py
│   │   └── ablation_qasper.py
│   ├── scalability                  # Scalability test scripts
│   │   ├── scalability_2wikimqa.py
│   │   └── scalability_hotpotqa.py
│   └── utils                        # Basic tool modules
│       ├── embedding.py
│       ├── evaluate.py
│       └── llm.py
├── README.md
├── dataset
│   ├── CloudDirWiki
│   │   ├── CloudDirWikiCorpus.json  # Cloud domain corpus
│   │   └── CloudDirWikiQA.json      # Cloud domain QA pairs
│   └── MediDirWiki
│       ├── MediDirWikiCorpus.json   # Medical domain corpus
│       └── MediDirWikiQA.json       # Medical domain QA pairs
└── requirements.txt
```
Total: 8 directories, 18 files.

## Datasets
We release two hierarchical RAG evaluation benchmarks: **MediDirWiki** (medical domain) and **CloudDirWiki** (cloud-native domain).
### File Location
- `dataset/MediDirWiki/MediDirWikiCorpus.json`: Raw medical domain corpus organized by directory hierarchy
- `dataset/MediDirWiki/MediDirWikiQA.json`: Corresponding medical question-answer pairs
- `dataset/CloudDirWiki/CloudDirWikiCorpus.json`: Raw cloud computing domain corpus organized by directory hierarchy
- `dataset/CloudDirWiki/CloudDirWikiQA.json`: Corresponding cloud computing question-answer pairs

### Dataset Statistics
| Metric                     | MediDirWiki       | CloudDirWiki      |
|----------------------------|-------------------|-------------------|
| Domain                     | Medical Wiki      | Cloud‑native Wiki |
| Task Type                  | Single + Multi‑hop| Single‑hop only   |
| Query Count                | 327               | 200               |
| Corpus Size (Characters)   | 1,757,591         | 19,417,801        |
| Average Directory Depth    | 4.46              | 4.72              |
| Minimum / Maximum Depth    | 3 / 5             | 2 / 7             |

## Environment Setup
Install all required dependencies with the provided configuration file:
```bash
pip install -r requirements.txt
```

## Experimental Usage
### 1. Ablation & Comparative Experiments
Before running the codes, please configure your local LLM model in `llm.py`. You can directly execute ablation experiments for the target datasets as follows:
```bash
# Example: ablation experiment on MediDirWiki
# Available options: full_dirrag, wo_section_routing, wo_iteration, title_only, summary_only, wo_depth_reward, semantic_only

python DirRAG/comparativeAndAblation/ablation_medi.py \
    --variant full_dirrag \
    --sample_size 500
```

### 2. Scalability Experiments
```bash
# Example: scalability experiments on 2wikimqa
python DirRAG/scalability/scalability_2wikimqa.py
```


## Notes
- Modify configuration parameters in `ablation_config.py` to adjust experimental settings (embedding models, LLM backends, hyperparameters etc.).
- `utils/` includes unified wrappers for embedding models, LLMs and evaluation metrics.