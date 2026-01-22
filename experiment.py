#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征选择方法对比实验
基于周志华《机器学习》第11章的方法
对比Relief、LVW、Lasso三种方法
"""

import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import json
from scipy.spatial.distance import pdist, squareform

# 设置随机种子
np.random.seed(42)

def load_dataset():
    """加载Breast Cancer数据集"""
    bc = load_breast_cancer()
    return {
        'X': bc.data,
        'y': bc.target,
        'feature_names': bc.feature_names,
        'n_features': bc.data.shape[1],
        'n_samples': bc.data.shape[0]
    }

def relief_selection(X, y, m=None, k=5):
    """
    Relief算法（西瓜书p.249-251）
    
    参数:
        X: 特征矩阵
        y: 标签
        m: 采样次数（默认使用所有样本）
        k: 最近邻数量
    """
    n_samples, n_features = X.shape
    
    if m is None:
        m = n_samples
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 初始化权重
    W = np.zeros(n_features)
    
    # 随机采样m次
    sample_indices = np.random.choice(n_samples, size=min(m, n_samples), replace=False)
    
    for i in sample_indices:
        x_i = X_scaled[i]
        y_i = y[i]
        
        # 计算到所有样本的距离
        distances = np.sqrt(np.sum((X_scaled - x_i) ** 2, axis=1))
        
        # 找到同类和异类的最近邻
        same_class_mask = (y == y_i) & (np.arange(n_samples) != i)
        diff_class_mask = (y != y_i)
        
        if np.sum(same_class_mask) > 0:
            same_class_distances = distances[same_class_mask]
            hit_indices = np.where(same_class_mask)[0]
            # 找到k个最近邻
            k_hit = min(k, len(hit_indices))
            hit_nearest = hit_indices[np.argsort(same_class_distances)[:k_hit]]
        else:
            hit_nearest = []
        
        if np.sum(diff_class_mask) > 0:
            diff_class_distances = distances[diff_class_mask]
            miss_indices = np.where(diff_class_mask)[0]
            # 找到k个最近邻
            k_miss = min(k, len(miss_indices))
            miss_nearest = miss_indices[np.argsort(diff_class_distances)[:k_miss]]
        else:
            miss_nearest = []
        
        # 更新权重
        for j in range(n_features):
            diff_hit = 0
            diff_miss = 0
            
            if len(hit_nearest) > 0:
                diff_hit = np.mean(np.abs(x_i[j] - X_scaled[hit_nearest, j]))
            
            if len(miss_nearest) > 0:
                diff_miss = np.mean(np.abs(x_i[j] - X_scaled[miss_nearest, j]))
            
            W[j] += diff_miss - diff_hit
    
    # 归一化权重
    if np.max(W) > 0:
        W = W / np.max(np.abs(W))
    
    return W

def lvw_selection(X, y, T=1000, base_estimator=None):
    """
    LVW算法（Las Vegas Wrapper，西瓜书p.251-253）
    
    参数:
        X: 特征矩阵
        y: 标签
        T: 最大迭代次数
        base_estimator: 基础分类器
    """
    if base_estimator is None:
        base_estimator = LogisticRegression(max_iter=2000, random_state=42, solver='liblinear')
    
    n_samples, n_features = X.shape
    
    # 使用5折交叉验证评估
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 初始化：使用所有特征
    A = set(range(n_features))
    best_score = np.mean(cross_val_score(base_estimator, X[:, list(A)], y, cv=cv, scoring='accuracy'))
    best_features = A.copy()
    
    # Las Vegas搜索
    for t in range(T):
        # 随机生成特征子集（随机选择特征数量）
        k = np.random.randint(1, n_features + 1)
        # 随机选择k个特征
        A_new = set(np.random.choice(n_features, size=k, replace=False))
        
        # 评估新特征子集
        try:
            score = np.mean(cross_val_score(base_estimator, X[:, list(A_new)], y, cv=cv, scoring='accuracy'))
            
            # 如果性能更好，或者性能相同但特征更少，则更新
            if score > best_score or (score == best_score and len(A_new) < len(best_features)):
                best_score = score
                best_features = A_new.copy()
        except:
            continue
    
    return list(best_features), best_score

def lasso_selection(X, y, n_features=None):
    """
    Lasso特征选择（西瓜书p.253-258）
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用LassoCV找到最优alpha
    lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
    lasso.fit(X_scaled, y)
    
    # 获取非零系数对应的特征
    selected_features = np.where(np.abs(lasso.coef_) > 1e-5)[0]
    
    # 如果选择的特征数少于n_features，选择系数绝对值最大的特征
    if n_features is not None and len(selected_features) < n_features:
        coef_abs = np.abs(lasso.coef_)
        selected_features = np.argsort(coef_abs)[-n_features:]
    
    X_selected = X[:, selected_features]
    scores = np.abs(lasso.coef_)
    return X_selected, selected_features, scores

def evaluate_features(X, y, method_name, cv=5):
    """评估特征选择后的分类性能"""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42), 
                                scoring='accuracy')
    
    return {
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'n_features': X.shape[1]
    }

def run_experiment():
    """运行完整实验"""
    dataset = load_dataset()
    X = dataset['X']
    y = dataset['y']
    
    print(f"数据集: Breast Cancer")
    print(f"样本数: {dataset['n_samples']}, 特征数: {dataset['n_features']}")
    
    results = {}
    
    # 基准性能（使用所有特征）
    print("\n运行基准测试（全部特征）...")
    start_time = time.time()
    baseline = evaluate_features(X, y, 'Baseline')
    baseline_time = time.time() - start_time
    baseline['time'] = baseline_time
    results['Baseline'] = baseline
    print(f"  准确率: {baseline['mean_accuracy']:.4f} ± {baseline['std_accuracy']:.4f}")
    print(f"  时间: {baseline_time:.4f}秒")
    
    # Relief算法
    print("\n运行Relief算法...")
    start_time = time.time()
    relief_weights = relief_selection(X, y, m=100, k=5)
    # 选择权重最大的10个特征
    n_features_to_select = 10
    relief_selected = np.argsort(relief_weights)[-n_features_to_select:]
    X_relief = X[:, relief_selected]
    relief_eval = evaluate_features(X_relief, y, 'Relief')
    relief_time = time.time() - start_time  # 总时间 = 特征选择 + 评估
    relief_eval['time'] = relief_time
    relief_eval['selected_features'] = relief_selected.tolist()
    relief_eval['weights'] = relief_weights[relief_selected].tolist()
    results['Relief'] = relief_eval
    print(f"  准确率: {relief_eval['mean_accuracy']:.4f} ± {relief_eval['std_accuracy']:.4f}")
    print(f"  选择的特征数: {len(relief_selected)}")
    print(f"  时间: {relief_time:.4f}秒")
    
    # LVW算法
    print("\n运行LVW算法...")
    start_time = time.time()
    lvw_selected, lvw_score = lvw_selection(X, y, T=100)
    X_lvw = X[:, lvw_selected]
    lvw_eval = evaluate_features(X_lvw, y, 'LVW')
    lvw_time = time.time() - start_time  # 总时间 = LVW特征选择 + 最终评估
    lvw_eval['time'] = lvw_time
    lvw_eval['selected_features'] = list(lvw_selected)
    lvw_eval['internal_score'] = lvw_score
    results['LVW'] = lvw_eval
    print(f"  准确率: {lvw_eval['mean_accuracy']:.4f} ± {lvw_eval['std_accuracy']:.4f}")
    print(f"  选择的特征数: {len(lvw_selected)}")
    print(f"  时间: {lvw_time:.4f}秒")
    
    # Lasso算法
    print("\n运行Lasso算法...")
    start_time = time.time()
    X_lasso, lasso_selected, lasso_scores = lasso_selection(X, y)
    lasso_eval = evaluate_features(X_lasso, y, 'Lasso')
    lasso_time = time.time() - start_time  # 总时间 = Lasso特征选择 + 评估
    lasso_eval['time'] = lasso_time
    lasso_eval['selected_features'] = lasso_selected.tolist()
    lasso_eval['coefficients'] = lasso_scores[lasso_selected].tolist()
    results['Lasso'] = lasso_eval
    print(f"  准确率: {lasso_eval['mean_accuracy']:.4f} ± {lasso_eval['std_accuracy']:.4f}")
    print(f"  选择的特征数: {len(lasso_selected)}")
    print(f"  时间: {lasso_time:.4f}秒")
    
    # 特征稳定性分析（运行多次看特征选择的一致性）
    print("\n进行特征稳定性分析...")
    stability_results = {}
    
    # Relief稳定性（运行5次）
    relief_runs = []
    for i in range(5):
        weights = relief_selection(X, y, m=100, k=5)
        selected = np.argsort(weights)[-n_features_to_select:]
        relief_runs.append(set(selected))
    # 计算Jaccard相似度
    relief_intersection = set.intersection(*relief_runs)
    relief_union = set.union(*relief_runs)
    relief_stability = len(relief_intersection) / len(relief_union) if len(relief_union) > 0 else 0
    stability_results['Relief'] = {
        'stability': relief_stability,
        'common_features': list(relief_intersection)
    }
    
    # Lasso稳定性（运行5次）
    lasso_runs = []
    for i in range(5):
        _, selected, _ = lasso_selection(X, y)
        lasso_runs.append(set(selected))
    lasso_intersection = set.intersection(*lasso_runs)
    lasso_union = set.union(*lasso_runs)
    lasso_stability = len(lasso_intersection) / len(lasso_union) if len(lasso_union) > 0 else 0
    stability_results['Lasso'] = {
        'stability': lasso_stability,
        'common_features': list(lasso_intersection)
    }
    
    results['stability'] = stability_results
    
    # 保存结果（转换numpy类型为Python原生类型）
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj
    
    results_serializable = convert_to_serializable(results)
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    # 生成LaTeX表格
    generate_latex_tables(results)
    
    return results

def generate_latex_tables(results):
    """生成LaTeX格式的结果表格"""
    
    # 准确率对比表
    table1 = []
    table1.append("\\begin{table}[H]")
    table1.append("\\centering")
    table1.append("\\caption{不同特征选择方法的分类准确率对比}")
    table1.append("\\label{tab:accuracy}")
    table1.append("\\begin{tabular}{lcc}")
    table1.append("\\toprule")
    table1.append("\\textbf{方法} & \\textbf{准确率} & \\textbf{标准差} \\\\")
    table1.append("\\midrule")
    
    methods = ['Baseline', 'Relief', 'LVW', 'Lasso']
    method_names = {
        'Baseline': '基准（全部特征）',
        'Relief': 'Relief',
        'LVW': 'LVW',
        'Lasso': 'Lasso'
    }
    
    for method in methods:
        if method in results:
            acc = results[method]['mean_accuracy']
            std = results[method]['std_accuracy']
            table1.append(f"{method_names[method]} & ${acc:.4f}$ & ${std:.4f}$ \\\\")
    
    table1.append("\\bottomrule")
    table1.append("\\end{tabular}")
    table1.append("\\end{table}")
    
    # 特征数量和计算时间对比表
    table2 = []
    table2.append("\\begin{table}[H]")
    table2.append("\\centering")
    table2.append("\\caption{特征选择方法的特征数量和计算时间对比}")
    table2.append("\\label{tab:time_features}")
    table2.append("\\begin{tabular}{lcc}")
    table2.append("\\toprule")
    table2.append("\\textbf{方法} & \\textbf{选择特征数} & \\textbf{计算时间（秒）} \\\\")
    table2.append("\\midrule")
    
    for method in ['Baseline', 'Relief', 'LVW', 'Lasso']:
        if method in results:
            n_feat = results[method]['n_features']
            t = results[method].get('time', 0)
            table2.append(f"{method_names[method]} & ${n_feat}$ & ${t:.4f}$ \\\\")
    
    table2.append("\\bottomrule")
    table2.append("\\end{tabular}")
    table2.append("\\end{table}")
    
    # 特征稳定性表
    table3 = []
    table3.append("\\begin{table}[H]")
    table3.append("\\centering")
    table3.append("\\caption{特征选择方法的稳定性分析}")
    table3.append("\\label{tab:stability}")
    table3.append("\\begin{tabular}{lc}")
    table3.append("\\toprule")
    table3.append("\\textbf{方法} & \\textbf{稳定性（Jaccard）} \\\\")
    table3.append("\\midrule")
    
    if 'stability' in results:
        for method in ['Relief', 'Lasso']:
            if method in results['stability']:
                stab = results['stability'][method]['stability']
                table3.append(f"{method_names[method]} & ${stab:.4f}$ \\\\")
    
    table3.append("\\bottomrule")
    table3.append("\\end{tabular}")
    table3.append("\\end{table}")
    
    # 合并所有表格
    all_tables = '\n\n'.join([
        '\n'.join(table1),
        '\n'.join(table2),
        '\n'.join(table3)
    ])
    
    with open('results_table.tex', 'w', encoding='utf-8') as f:
        f.write(all_tables)
    
    print("\n结果表格已保存到 results_table.tex")

if __name__ == '__main__':
    print("=" * 60)
    print("特征选择方法对比实验")
    print("基于周志华《机器学习》第11章")
    print("=" * 60)
    results = run_experiment()
    print("\n" + "=" * 60)
    print("实验完成！结果已保存到 results.json 和 results_table.tex")
    print("=" * 60)
