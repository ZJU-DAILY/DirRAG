import json
import argparse
import re
import numpy as np
from collections import Counter


def normalize_answer(s):
    """
    标准化答案：统一小写、去除标点、多余空格
    提升评估的合理性（比如 Köln 和 köln 视为相同）
    """
    # 转小写
    s = s.lower()
    # 去除标点符号
    s = re.sub(r'[^\w\s]', '', s)
    # 去除多余空格
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def compute_em(pred, gold):
    """计算精确匹配分数"""
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def compute_f1(pred, gold):
    """计算单词级别的F1分数"""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    
    # 公共单词统计
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    # 无重叠则F1为0
    if num_same == 0:
        return 0.0
    
    # 计算精确率、召回率、F1
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_jsonl(file_path, max_check_num=-1):
    """
    读取jsonl文件，计算平均EM和平均F1
    :param file_path: jsonl文件路径
    :param max_check_num: 最多检查前N个样本，-1代表全部检查
    """
    em_scores = []
    f1_scores = []
    sample_cnt = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 达到上限直接跳出循环
            if max_check_num != -1 and sample_cnt >= max_check_num:
                break
            
            if not line.strip():
                continue
            data = json.loads(line)
            # 提取预测答案和标准答案
            pred = data.get('pred_answer', '')
            gold = data.get('gold_answer', '')
            # pred = data.get('pred', '')
            # gold = data.get('gold', '')
            
            # 计算单条样本的分数
            em = compute_em(pred, gold)
            f1 = compute_f1(pred, gold)
            
            em_scores.append(em)
            f1_scores.append(f1)
            sample_cnt += 1

    # 关键新增逻辑：如果设置了max_check_num且真实有效样本不足，补齐0分样本
    if max_check_num != -1 and sample_cnt < max_check_num:
        pad_num = max_check_num - sample_cnt
        em_scores += [0.0] * pad_num
        f1_scores += [0.0] * pad_num
    
    # 计算平均分
    avg_em = np.mean(em_scores) if em_scores else 0.0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    
    return avg_em, avg_f1, len(em_scores), sample_cnt


if __name__ == '__main__':
    # 写死的控制变量：-1 检查全部，数字N则只检查前N个样本，不足N条补0分
    MAX_CHECK_SAMPLES = 200

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='计算LongBench结果的EM和F1分数')
    parser.add_argument('--file', type=str, 
                       default='/root/dir-sem-rag/longbench/test/ablation_hotpotqa_with_dir/results_hotpotqa/cleaned_results_hotpotqa_with_dir_full_dirrag.jsonl',
                       help='jsonl结果文件路径')
    
    args = parser.parse_args()
    
    # 执行评估，传入写死的上限变量
    avg_em, avg_f1, total_used, real_valid = evaluate_jsonl(args.file, max_check_num=MAX_CHECK_SAMPLES)
    
    # 打印结果
    print("="*50)
    print(f"评估文件：{args.file}")
    print(f"配置评估总样本数：{MAX_CHECK_SAMPLES if MAX_CHECK_SAMPLES != -1 else '全部'}")
    print(f"文件内真实有效样本数：{real_valid}")
    print(f"参与平均分计算总样本（含补0）：{total_used}")
    if MAX_CHECK_SAMPLES != -1 and real_valid < MAX_CHECK_SAMPLES:
        print(f"补充0分样本数量：{MAX_CHECK_SAMPLES - real_valid}")
    print(f"平均 EM 分数：{avg_em:.4f}")
    print(f"平均 F1 分数：{avg_f1:.4f}")
    print("="*50)