# DirRAG: Leveraging Native Directory Hierarchies as Structural Priors for Retrieval-Augmented Generation
This repository hosts the official source code, datasets and experimental scripts for the paper **DirRAG: Leveraging Native Directory Hierarchies as Structural Priors for Retrieval-Augmented Generation**.

## Overview
The repository consists of two core modules:
1. **Custom Hierarchical QA Datasets**: Two newly constructed benchmarks built with inherent directory hierarchical structures, tailored for evaluating hierarchical retrieval-augmented generation methods.
2. **DirRAG Implementation**: Full experimental code for baseline comparisons, ablation studies, and scalability verification of the proposed DirRAG framework.

## Repository Structure
```
.
├── Baselines
│   ├── flatrag.py
│   ├── graphrag.py
│   ├── hipporag2.py
│   ├── lightrag.py
│   └── raptor.py
├── Dataset
│   ├── CloudDirWiki
│   │   ├── CloudDirWikiCorpus.json
│   │   └── CloudDirWikiQA.json
│   └── MediDirWiki
│       ├── MediDirWikiCorpus.json
│       └── MediDirWikiQA.json
├── DirRAG
│   ├── utils
│   │   ├── embedding.py
│   │   ├── evaluate.py
│   │   └── llm.py
│   ├── dirrag_ablation.py
│   ├── dirrag_enhance.py
│   ├── dirrag_framework.py
│   └── dirrag_scalability.py
├── README.md
└── requirements.txt
```

## Datasets
We release two hierarchical RAG evaluation benchmarks: **MediDirWiki** (medical domain) and **CloudDirWiki** (cloud-native domain).
### File Location
- `Dataset/MediDirWiki/MediDirWikiCorpus.json`: Raw medical domain corpus organized by directory hierarchy
- `Dataset/MediDirWiki/MediDirWikiQA.json`: Corresponding medical question-answer pairs
- `Dataset/CloudDirWiki/CloudDirWikiCorpus.json`: Raw cloud computing domain corpus organized by directory hierarchy
- `Dataset/CloudDirWiki/CloudDirWikiQA.json`: Corresponding cloud computing question-answer pairs

### Dataset Statistics
| Metric                     | MediDirWiki       | CloudDirWiki      |
|----------------------------|-------------------|-------------------|
| Domain                     | Medical Wiki      | Cloud‑native Wiki |
| Task Type                  | Single + Multi‑hop| Single‑hop only   |
| Query Count                | 327               | 202               |
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
Before running the codes, please configure your local LLM model in `DirRAG/utils/llm.py`.
```bash
# Ablation study (MediDirWiki)
python DirRAG/dirrag_ablation.py

# Enhanced version with supplementary retrieval strategies
python DirRAG/dirrag_enhance.py
```

### 2. Scalability Experiments
```bash
# Scalability test with DirRAG on CloudDirWiki
python DirRAG/dirrag_scalability.py
```

### 3. Baseline Methods
```bash
# FlatRAG (BM25 / Dense)
python Baselines/flatrag.py

# GraphRAG
python Baselines/graphrag.py

# LightRAG
python Baselines/lightrag.py

# HippoRAG2
python Baselines/hipporag2.py

# RAPTOR
python Baselines/raptor.py
```

## Notes
- `DirRAG/utils/` provides unified wrappers for embedding models, LLMs and evaluation metrics.
- `DirRAG/dirrag_framework.py` contains the core DirRAG system implementation.
- Baselines use the same LLM and embedding backends for fair comparison.
