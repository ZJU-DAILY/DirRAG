import json
import argparse
import os
import re
import sys
from collections import Counter


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if not xs:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def compute_em(pred, gold):
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    return 1.0 if pred_norm == gold_norm else 0.0


def compute_f1(pred, gold):
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def evaluate_jsonl(file_path, start_idx=0, max_check_num=-1):
    em_scores = []
    f1_scores = []
    recall_at_k_list = []
    indexing_time_list = []
    retrieval_time_list = []
    e2e_latency_list = []
    total_token_list = []
    retrieval_token_list = []
    indexing_token_list = []
    llm_call_list = []
    mrr_list = []
    sample_cnt = 0
    line_idx = 0               

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                line_idx += 1
                continue

                                
            if line_idx < start_idx:
                line_idx += 1
                continue

                                          
            if max_check_num != -1 and sample_cnt >= max_check_num:
                break

            data = json.loads(line)
            pred = data.get('pred_answer', '')
            gold = data.get('gold_answer', '')
            em = compute_em(pred, gold)
            f1 = compute_f1(pred, gold)
            em_scores.append(em)
            f1_scores.append(f1)
            metrics = data["metrics"]
            recall_at_k_list.append(metrics["recall_at_k"])
            indexing_time_list.append(metrics["indexing_time_s"])
            retrieval_time_list.append(metrics["retrieval_time_s"])
            e2e_latency_list.append(metrics["e2e_latency_s"])
            total_token_list.append(metrics["token_cost"]["total_tokens"])
                                                                                          
            indexing_token_list.append(metrics["indexing_token_cost"]["total_tokens"])
            llm_call_list.append(metrics["llm_call_count"])
            mrr_list.append(metrics["mrr"])
            sample_cnt += 1
            line_idx += 1

                                    
    avg_em = _mean(em_scores)
    avg_f1 = _mean(f1_scores)
    avg_recall_at_k = _mean(recall_at_k_list)
    avg_indexing_time = _mean(indexing_time_list)
    avg_retrieval_time = _mean(retrieval_time_list)
    avg_e2e_latency = _mean(e2e_latency_list)
    avg_total_token = _mean(total_token_list)
    avg_ret_token = _mean(retrieval_token_list)
    avg_idx_token = _mean(indexing_token_list)
    avg_llm_call = _mean(llm_call_list)
    avg_mrr = _mean(mrr_list)

    return (avg_em, avg_f1, avg_recall_at_k, avg_indexing_time, avg_retrieval_time,
            avg_e2e_latency, avg_total_token, avg_ret_token, avg_idx_token,
            avg_llm_call, avg_mrr, len(em_scores), sample_cnt)


def print_report(file_path, res, start_idx, max_check_num):
    (avg_em, avg_f1, avg_recall_at_k, avg_indexing_time, avg_retrieval_time,
     avg_e2e_latency, avg_total_token, avg_ret_token, avg_idx_token,
     avg_llm_call, avg_mrr, total_used, real_valid) = res
    print("=" * 80)
    print(f"File: {file_path}")
    print(f"Start index: {start_idx}")
    print(f"Check samples: {max_check_num}")
    print(f"本次实际读到有效样本数量: {real_valid}")
    print(f"Avg EM:                {avg_em:.4f}")
    print(f"Avg F1:                {avg_f1:.4f}")
    print(f"Avg Recall@k:          {avg_recall_at_k:.4f}")
    print(f"Avg indexing_time_s:   {avg_indexing_time:.4f}")
    print(f"Avg retrieval_time_s:  {avg_retrieval_time:.4f}")
    print(f"Avg e2e_latency_s:     {avg_e2e_latency:.4f}")
    print(f"Avg total_tokens:      {avg_total_token:.2f}")
    print(f"Avg retrieval_tokens:  {avg_ret_token:.2f}")
    print(f"Avg indexing_tokens:   {avg_idx_token:.2f}")
    print(f"Avg llm_call_count:    {avg_llm_call:.2f}")
    print(f"Avg MRR:               {avg_mrr:.4f}")
    print("=" * 80)


def detect_rerun_files(file_path):
    m = re.match(r'^(.*)_rerun_(\d+)(\.[^.]+)$', file_path)
    if not m:
        return None
    base, _, ext = m.groups()
    return [f"{base}_rerun_{i}{ext}" for i in (1, 2, 3)]


METRIC_NAMES = ["EM", "F1", "Recall@k", "indexing_time_s", "retrieval_time_s",
                "e2e_latency_s", "total_tokens", "retrieval_tokens",
                "indexing_tokens", "llm_call_count", "MRR"]
                               
QUALITY_METRICS = {"EM", "F1", "Recall@k", "MRR"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="统计HotpotQA jsonl结果：EM,F1,Recall@k,indexing_time,retrieval_time,e2e_latency,token,llm_call,MRR")
    parser.add_argument('--file', type=str, default="/root/dir-sem-rag/longbench/test/dir_rag_section_chunk_claude/lightrag_cloud_dir_results_rerun_1.jsonl")
    parser.add_argument('--start', type=int, default=0, help="从第几个样本开始统计，下标从0开始，包含该样本")
    parser.add_argument('--check-samples', type=int, default=-1, help="统计多少个样本，-1代表读到文件末尾")
    args = parser.parse_args()

    rerun_files = detect_rerun_files(args.file)

    if rerun_files is None:
                                       
        res = evaluate_jsonl(args.file, start_idx=args.start, max_check_num=args.check_samples)
        print_report(args.file, res, args.start, args.check_samples)
    else:
                                                           
        print(f"检测到 rerun 结果文件: {args.file}，将循环评估 rerun_1/2/3")
        all_res = []
        for fp in rerun_files:
            if not os.path.exists(fp):
                print(f"⚠️ 缺少 rerun 文件，跳过: {fp}")
                continue
            res = evaluate_jsonl(fp, start_idx=args.start, max_check_num=args.check_samples)
            print_report(fp, res, args.start, args.check_samples)
            all_res.append((fp, res))

        if not all_res:
            print("❌ 未找到任何 rerun 结果文件 (rerun_1/2/3)，退出。")
            sys.exit(1)

        if len(all_res) < 3:
            print(f"⚠️ 仅找到 {len(all_res)}/3 个 rerun 文件，以下平均值基于已有文件计算。")

        cols = list(zip(*[r[1][:len(METRIC_NAMES)] for r in all_res]))
        means = [_mean(list(c)) for c in cols]
        stds = [_std(list(c)) for c in cols]
        print("#" * 80)
        print(f"RERUN 平均（共 {len(all_res)} 次）:")
        for fp, _ in all_res:
            print(f"  - {fp}")
        for name, mean, std in zip(METRIC_NAMES, means, stds):
            if name in QUALITY_METRICS:
                print(f"  {name:<18} mean={mean:.4f}  std={std:.4f}")
            elif "tokens" in name or name == "llm_call_count":
                print(f"  {name:<18} mean={mean:.2f}")
            else:
                print(f"  {name:<18} mean={mean:.4f}")
        print("#" * 80)
