import os
import warnings

# 設置環境變數以避免 joblib 警告
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# 忽略特定警告
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from collections import Counter
import time
import csv

# pandas, matplotlib, seaborn 會在需要時才載入

# ==================== 載入資料 ====================
print("載入資料中...")

# 獲取腳本所在目錄
script_dir = os.path.dirname(os.path.abspath(__file__))

# 載入特徵資料 (10000張圖, 478個特徵)
data_path = os.path.join(script_dir, "fnormal.txt")
features = []
with open(data_path, 'r') as f:
    for line in f:
        features.append([float(x) for x in line.strip().split()])

features = np.array(features)
print(f"特徵資料維度: {features.shape}")

# 載入標籤資料 (每張圖的類別)
name_path = os.path.join(script_dir, "name.txt")
labels = []
with open(name_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            labels.append(parts[1])  # 取類別名稱

labels = np.array(labels)
print(f"標籤數量: {len(labels)}")
print(f"類別總數: {len(set(labels))}")

# 載入測試集編號 (query.txt中的編號為測試集，從1開始計數)
query_path = os.path.join(script_dir, "query.txt")
test_indices = []
with open(query_path, 'r') as f:
    for line in f:
        test_indices.append(int(line.strip()) - 1)  # 轉為0-based索引

test_indices = np.array(test_indices)
print(f"測試集數量: {len(test_indices)}")

# 建立訓練集和測試集
all_indices = np.arange(len(features))
train_indices = np.array([i for i in all_indices if i not in test_indices])
print(f"訓練集數量: {len(train_indices)}")

X_train = features[train_indices]
y_train = labels[train_indices]
X_test = features[test_indices]
y_test = labels[test_indices]

# ==================== 特徵標準化 ====================
print("\n進行特徵標準化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 特徵分組 (可選) ====================
# 定義各特徵範圍
feature_groups = {
    'ColorStructure': (0, 32),      # 0-31
    'ColorLayout': (32, 44),         # 32-43
    'RegionShape': (44, 80),         # 44-79
    'HomogeneousTexture': (80, 142), # 80-141
    'EdgeHistogram': (142, 222)      # 142-221
}

print("\n特徵分組:")
for name, (start, end) in feature_groups.items():
    print(f"  {name}: 欄位 {start}-{end-1} ({end-start}個特徵)")

# ==================== KNN 分類 ====================
print("\n" + "="*60)
print("開始 KNN 分類...")
print("="*60)

# 測試不同的 K 值
k_values = [1, 3, 5, 7, 9, 11, 15, 19]
best_k = 1
best_accuracy = 0
results = {}
knn_times = {}

for k in k_values:
    print(f"\n測試 K={k}...")
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=1)
    knn.fit(X_train_scaled, y_train)
    
    # 預測
    y_pred = knn.predict(X_test_scaled)
    elapsed_time = time.time() - start_time
    
    # 計算準確率
    accuracy = accuracy_score(y_test, y_pred)
    results[k] = accuracy
    knn_times[k] = elapsed_time
    print(f"  準確率: {accuracy:.4f} ({accuracy*100:.2f}%) - 訓練+預測時間: {elapsed_time:.2f}秒")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print("\n" + "="*60)
print(f"最佳 K 值: {best_k}")
print(f"最佳準確率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print("="*60)

# ==================== 使用最佳 K 值進行詳細評估 ====================
print(f"\n使用 K={best_k} 進行詳細評估...")
start_time = time.time()
knn_best = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', n_jobs=1)
knn_best.fit(X_train_scaled, y_train)
y_pred_best = knn_best.predict(X_test_scaled)
time_knn_best = time.time() - start_time

# 計算訓練準確率
y_pred_train_best = knn_best.predict(X_train_scaled)
train_accuracy_best = accuracy_score(y_train, y_pred_train_best)

# ==================== 評估結果 ====================
print("\n" + "="*60)
print("分類報告")
print("="*60)
print(classification_report(y_test, y_pred_best, zero_division=0))

# 計算每個類別的準確率
print("\n" + "="*60)
print("各類別準確率統計")
print("="*60)

unique_classes = sorted(set(y_test))
class_accuracies = []

for cls in unique_classes:
    cls_mask = (y_test == cls)
    if np.sum(cls_mask) > 0:
        cls_accuracy = accuracy_score(y_test[cls_mask], y_pred_best[cls_mask])
        class_accuracies.append((cls, cls_accuracy, np.sum(cls_mask)))
        print(f"{cls:30s}: {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%) - 測試樣本數: {np.sum(cls_mask)}")

# 計算混淆矩陣
conf_matrix = confusion_matrix(y_test, y_pred_best)
print(f"\n混淆矩陣維度: {conf_matrix.shape}")

# ==================== 錯誤分析 ====================
print("\n" + "="*60)
print("錯誤分析")
print("="*60)

incorrect_mask = (y_test != y_pred_best)
incorrect_indices = test_indices[incorrect_mask]
print(f"錯誤分類數量: {np.sum(incorrect_mask)} / {len(y_test)}")
print(f"錯誤率: {np.sum(incorrect_mask)/len(y_test)*100:.2f}%")

# 顯示前10個錯誤案例
print("\n前10個錯誤分類案例:")
error_count = 0
for i, idx in enumerate(incorrect_indices[:10]):
    original_idx = idx + 1  # 轉回1-based
    print(f"  圖片索引 {original_idx}: 真實={y_test[incorrect_mask][i]:20s}, 預測={y_pred_best[incorrect_mask][i]:20s}")
    error_count += 1

# ==================== SVM 分類 ====================
print("\n" + "="*60)
print("開始 SVM 分類...")
print("="*60)

# 測試不同的 SVM 參數
print("\n訓練 SVM (RBF kernel)...")
start_time = time.time()
svm_rbf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_svm_rbf = svm_rbf.predict(X_test_scaled)
acc_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)
time_svm_rbf = time.time() - start_time

# 計算訓練準確率
y_pred_train_svm_rbf = svm_rbf.predict(X_train_scaled)
train_acc_svm_rbf = accuracy_score(y_train, y_pred_train_svm_rbf)

print(f"SVM (RBF) 準確率: {acc_svm_rbf:.4f} ({acc_svm_rbf*100:.2f}%) - 訓練時間: {time_svm_rbf:.2f}秒")

print("\n訓練 SVM (Linear kernel)...")
start_time = time.time()
svm_linear = SVC(kernel='linear', C=1, random_state=42)
svm_linear.fit(X_train_scaled, y_train)
y_pred_svm_linear = svm_linear.predict(X_test_scaled)
acc_svm_linear = accuracy_score(y_test, y_pred_svm_linear)
time_svm_linear = time.time() - start_time

# 計算訓練準確率
y_pred_train_svm_linear = svm_linear.predict(X_train_scaled)
train_acc_svm_linear = accuracy_score(y_train, y_pred_train_svm_linear)

print(f"SVM (Linear) 準確率: {acc_svm_linear:.4f} ({acc_svm_linear*100:.2f}%) - 訓練時間: {time_svm_linear:.2f}秒")

# ==================== LDA 分類 ====================
print("\n" + "="*60)
print("開始 LDA 分類...")
print("="*60)

start_time = time.time()
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)
y_pred_train_lda = lda.predict(X_train_scaled)  # 訓練集預測
y_pred_lda = lda.predict(X_test_scaled)
acc_lda = accuracy_score(y_test, y_pred_lda)
train_acc_lda = accuracy_score(y_train, y_pred_train_lda)  # 訓練準確率
time_lda = time.time() - start_time
print(f"LDA 準確率: {acc_lda:.4f} ({acc_lda*100:.2f}%) - 訓練時間: {time_lda:.2f}秒")

# ==================== Random Forest 分類 ====================
print("\n" + "="*60)
print("開始 Random Forest 分類...")
print("="*60)

# 測試不同的樹數量
print("\n測試不同的樹數量...")
n_estimators_list = [50, 100, 200]
best_rf_acc = 0
best_rf_n = 50
rf_results = {}

for n in n_estimators_list:
    print(f"\n測試 n_estimators={n}...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=1, max_depth=20)
    rf.fit(X_train_scaled, y_train)
    y_pred_train_rf = rf.predict(X_train_scaled)  # 訓練集預測
    y_pred_rf = rf.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    train_acc_rf = accuracy_score(y_train, y_pred_train_rf)  # 訓練準確率
    time_rf = time.time() - start_time
    rf_results[n] = acc_rf
    print(f"  準確率: {acc_rf:.4f} ({acc_rf*100:.2f}%) - 訓練時間: {time_rf:.2f}秒")
    
    if acc_rf > best_rf_acc:
        best_rf_acc = acc_rf
        best_rf_n = n
        best_rf_model = rf
        y_pred_rf_best = y_pred_rf
        train_acc_rf_best = train_acc_rf  # 保存訓練準確率
        time_rf_best = time_rf

print(f"\n最佳 Random Forest 參數: n_estimators={best_rf_n}")
print(f"最佳 Random Forest 準確率: {best_rf_acc:.4f} ({best_rf_acc*100:.2f}%)")

# ==================== 所有分類器比較 ====================
print("\n" + "="*60)
print("所有分類器效能比較")
print("="*60)

# 記錄所有分類器的詳細資訊（包含訓練準確率）
all_classifiers_info = [
    ('KNN (K={})'.format(best_k), best_accuracy, y_pred_best, time_knn_best, knn_best, train_accuracy_best),
    ('SVM (RBF)', acc_svm_rbf, y_pred_svm_rbf, time_svm_rbf, svm_rbf, train_acc_svm_rbf),
    ('SVM (Linear)', acc_svm_linear, y_pred_svm_linear, time_svm_linear, svm_linear, train_acc_svm_linear),
    ('LDA', acc_lda, y_pred_lda, time_lda, lda, train_acc_lda),
    ('Random Forest (n={})'.format(best_rf_n), best_rf_acc, y_pred_rf_best, time_rf_best, best_rf_model, train_acc_rf_best)
]

all_classifiers = [
    ('KNN (K={})'.format(best_k), best_accuracy, y_pred_best),
    ('SVM (RBF)', acc_svm_rbf, y_pred_svm_rbf),
    ('SVM (Linear)', acc_svm_linear, y_pred_svm_linear),
    ('LDA', acc_lda, y_pred_lda),
    ('Random Forest (n={})'.format(best_rf_n), best_rf_acc, y_pred_rf_best)
]

print("\n分類器準確率排名:")
sorted_classifiers = sorted(all_classifiers, key=lambda x: x[1], reverse=True)
for i, (name, acc, _) in enumerate(sorted_classifiers, 1):
    print(f"{i}. {name:30s}: {acc:.4f} ({acc*100:.2f}%)")

# 找出最佳分類器
best_classifier_name, best_classifier_acc, best_classifier_pred = sorted_classifiers[0]
print(f"\n最佳分類器: {best_classifier_name}")
print(f"最佳準確率: {best_classifier_acc:.4f} ({best_classifier_acc*100:.2f}%)")

# ==================== 計算每個分類器的詳細指標 ====================
print("\n" + "="*60)
print("計算所有分類器的詳細指標...")
print("="*60)

# 提前載入 pandas
import pandas as pd

def calculate_iou(y_true, y_pred, labels):
    """計算每個類別的 IoU (Jaccard Index)"""
    iou_scores = []
    for label in labels:
        y_true_binary = (y_true == label).astype(int)
        y_pred_binary = (y_pred == label).astype(int)
        intersection = np.sum(y_true_binary & y_pred_binary)
        union = np.sum(y_true_binary | y_pred_binary)
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
    return np.array(iou_scores)

def calculate_dice(precision, recall):
    """計算 Dice 係數 (等同於 F1-score)"""
    return 2 * (precision * recall) / (precision + recall + 1e-10)

unique_classes = sorted(set(y_test))

# 儲存所有分類器的詳細結果
all_detailed_results = []

for clf_name, clf_acc, clf_pred, clf_time, clf_model, train_clf_acc in all_classifiers_info:
    print(f"\n處理 {clf_name}...")
    
    # 計算 precision, recall, f1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, clf_pred, labels=unique_classes, zero_division=0
    )
    
    # 計算 IoU
    iou = calculate_iou(y_test, clf_pred, unique_classes)
    
    # 計算 Dice (等同於 F1-score)
    dice = f1
    
    # 儲存每個類別的結果
    for i, cls in enumerate(unique_classes):
        all_detailed_results.append({
            'Classifier': clf_name,
            'Class': cls,
            'Precision': precision[i],
            'Recall': recall[i],
            'F1-Score': f1[i],
            'Dice': dice[i],
            'IoU': iou[i],
            'Support': support[i],
            'Accuracy': clf_acc,
            'Training_Time': clf_time
        })

# 轉換為 DataFrame
df_detailed = pd.DataFrame(all_detailed_results)

print("\n詳細指標計算完成！")

# ==================== 最佳分類器的詳細報告 ====================
if best_classifier_name.startswith('SVM (RBF)'):
    print("\n" + "="*60)
    print(f"{best_classifier_name} 詳細分類報告")
    print("="*60)
    print(classification_report(y_test, y_pred_svm_rbf, zero_division=0))
elif best_classifier_name.startswith('SVM (Linear)'):
    print("\n" + "="*60)
    print(f"{best_classifier_name} 詳細分類報告")
    print("="*60)
    print(classification_report(y_test, y_pred_svm_linear, zero_division=0))
elif best_classifier_name.startswith('LDA'):
    print("\n" + "="*60)
    print(f"{best_classifier_name} 詳細分類報告")
    print("="*60)
    print(classification_report(y_test, y_pred_lda, zero_division=0))
elif best_classifier_name.startswith('Random Forest'):
    print("\n" + "="*60)
    print(f"{best_classifier_name} 詳細分類報告")
    print("="*60)
    print(classification_report(y_test, y_pred_rf_best, zero_division=0))

# ==================== 不同特徵組合的測試 (可選) ====================
print("\n" + "="*60)
print("測試不同特徵組合 (K={})".format(best_k))
print("="*60)

feature_combinations = {
    'All Features': list(range(222)),
    'ColorStructure': list(range(0, 32)),
    'ColorLayout': list(range(32, 44)),
    'RegionShape': list(range(44, 80)),
    'HomogeneousTexture': list(range(80, 142)),
    'EdgeHistogram': list(range(142, 222)),
    'Color Features': list(range(0, 44)),  # ColorStructure + ColorLayout
    'Texture Features': list(range(80, 222)),  # HomogeneousTexture + EdgeHistogram
}

for name, feature_idx in feature_combinations.items():
    if len(feature_idx) == 0:
        continue
    
    X_train_subset = X_train[:, feature_idx]
    X_test_subset = X_test[:, feature_idx]
    
    # 標準化
    scaler_subset = StandardScaler()
    X_train_subset_scaled = scaler_subset.fit_transform(X_train_subset)
    X_test_subset_scaled = scaler_subset.transform(X_test_subset)
    
    # 訓練並預測
    knn_subset = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', n_jobs=1)
    knn_subset.fit(X_train_subset_scaled, y_train)
    y_pred_subset = knn_subset.predict(X_test_subset_scaled)
    
    accuracy_subset = accuracy_score(y_test, y_pred_subset)
    print(f"{name:25s} ({len(feature_idx):3d} 特徵): {accuracy_subset:.4f} ({accuracy_subset*100:.2f}%)")

# ==================== 生成視覺化圖表 ====================
print("\n" + "="*60)
print("生成視覺化圖表...")
print("="*60)

# 載入視覺化套件
import matplotlib.pyplot as plt
import seaborn as sns

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 創建輸出目錄
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# 1. 各分類器準確率比較圖
fig, ax = plt.subplots(figsize=(12, 6))
clf_names = [name for name, _, _ in sorted_classifiers]
clf_accs = [acc * 100 for _, acc, _ in sorted_classifiers]
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
bars = ax.bar(range(len(clf_names)), clf_accs, color=colors, alpha=0.8, edgecolor='black')
ax.set_xlabel('分類器', fontsize=12, fontweight='bold')
ax.set_ylabel('準確率 (%)', fontsize=12, fontweight='bold')
ax.set_title('各分類器準確率比較', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(range(len(clf_names)))
ax.set_xticklabels(clf_names, rotation=15, ha='right')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
# 在柱子上標註數值
for bar, acc in zip(bars, clf_accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'classifier_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
print("[OK] 準確率比較圖已儲存: output/classifier_accuracy_comparison.png")
plt.close()

# 2. 訓練時間比較圖
fig, ax = plt.subplots(figsize=(12, 6))
clf_times = [clf_time for _, _, _, clf_time, _, _ in all_classifiers_info]
bars = ax.bar(range(len(clf_names)), clf_times, color=colors, alpha=0.8, edgecolor='black')
ax.set_xlabel('分類器', fontsize=12, fontweight='bold')
ax.set_ylabel('訓練時間 (秒)', fontsize=12, fontweight='bold')
ax.set_title('各分類器訓練時間比較', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(range(len(clf_names)))
ax.set_xticklabels(clf_names, rotation=15, ha='right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
for bar, t in zip(bars, clf_times):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{t:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'classifier_time_comparison.png'), dpi=300, bbox_inches='tight')
print("[OK] 訓練時間比較圖已儲存: output/classifier_time_comparison.png")
plt.close()

# 3. 各分類器各類別準確率熱力圖
fig, ax = plt.subplots(figsize=(16, 10))
class_acc_matrix = []
for clf_name, _, clf_pred, _, _, _ in all_classifiers_info:
    class_accs = []
    for cls in unique_classes:
        cls_mask = (y_test == cls)
        if np.sum(cls_mask) > 0:
            cls_acc = accuracy_score(y_test[cls_mask], clf_pred[cls_mask])
            class_accs.append(cls_acc * 100)
        else:
            class_accs.append(0)
    class_acc_matrix.append(class_accs)

sns.heatmap(class_acc_matrix, annot=False, fmt='.1f', cmap='RdYlGn', 
            xticklabels=unique_classes, yticklabels=clf_names,
            cbar_kws={'label': '準確率 (%)'}, ax=ax, vmin=0, vmax=100)
ax.set_title('各分類器在各類別的準確率熱力圖', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('類別', fontsize=12, fontweight='bold')
ax.set_ylabel('分類器', fontsize=12, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_accuracy_heatmap.png'), dpi=300, bbox_inches='tight')
print("[OK] 類別準確率熱力圖已儲存: output/class_accuracy_heatmap.png")
plt.close()

# 4. 最佳分類器的混淆矩陣
fig, ax = plt.subplots(figsize=(20, 18))
cm = confusion_matrix(y_test, best_classifier_pred, labels=unique_classes)
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
            xticklabels=unique_classes, yticklabels=unique_classes,
            cbar_kws={'label': '樣本數'}, ax=ax)
ax.set_title(f'{best_classifier_name} 混淆矩陣', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('預測類別', fontsize=12, fontweight='bold')
ax.set_ylabel('真實類別', fontsize=12, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=9)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'best_classifier_confusion_matrix.png'), dpi=300, bbox_inches='tight')
print("[OK] 最佳分類器混淆矩陣已儲存: output/best_classifier_confusion_matrix.png")
plt.close()

# 5. 各分類器平均指標比較 (Precision, Recall, F1, IoU)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['Precision', 'Recall', 'F1-Score', 'IoU']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    avg_values = []
    for clf_name in clf_names:
        clf_data = df_detailed[df_detailed['Classifier'] == clf_name]
        avg_values.append(clf_data[metric].mean() * 100)
    
    bars = ax.bar(range(len(clf_names)), avg_values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title(f'平均 {metric}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric} (%)', fontsize=10)
    ax.set_xticks(range(len(clf_names)))
    ax.set_xticklabels(clf_names, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, avg_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.suptitle('各分類器平均指標比較', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'classifier_metrics_comparison.png'), dpi=300, bbox_inches='tight')
print("[OK] 指標比較圖已儲存: output/classifier_metrics_comparison.png")
plt.close()

# ==================== 儲存 CSV 檔案 ====================
print("\n" + "="*60)
print("儲存 CSV 檔案...")
print("="*60)

# 1. 儲存所有分類器的詳細指標
df_detailed.to_csv(os.path.join(output_dir, 'detailed_metrics_all_classifiers.csv'), index=False, encoding='utf-8-sig')
print("[OK] 詳細指標已儲存: output/detailed_metrics_all_classifiers.csv")

# 2. 儲存各分類器總體性能
overall_performance = []
for clf_name, clf_acc, _, clf_time, _, train_clf_acc in all_classifiers_info:
    clf_data = df_detailed[df_detailed['Classifier'] == clf_name]
    overall_performance.append({
        'Classifier': clf_name,
        'Training_Accuracy': train_clf_acc * 100,
        'Testing_Accuracy': clf_acc * 100,
        'Avg_Precision': clf_data['Precision'].mean() * 100,
        'Avg_Recall': clf_data['Recall'].mean() * 100,
        'Avg_F1-Score': clf_data['F1-Score'].mean() * 100,
        'Avg_Dice': clf_data['Dice'].mean() * 100,
        'Avg_IoU': clf_data['IoU'].mean() * 100,
        'Training_Time_Seconds': clf_time
    })

df_overall = pd.DataFrame(overall_performance)
df_overall = df_overall.sort_values('Testing_Accuracy', ascending=False)
df_overall.to_csv(os.path.join(output_dir, 'overall_performance.csv'), index=False, encoding='utf-8-sig')
print("[OK] 總體性能已儲存: output/overall_performance.csv")

# 3. 儲存最佳分類器的混淆矩陣
cm_df = pd.DataFrame(cm, index=unique_classes, columns=unique_classes)
cm_df.to_csv(os.path.join(output_dir, 'best_classifier_confusion_matrix.csv'), encoding='utf-8-sig')
print("[OK] 混淆矩陣已儲存: output/best_classifier_confusion_matrix.csv")

# 4. 儲存 KNN 不同 K 值的結果
knn_k_results = []
for k in k_values:
    knn_k_results.append({
        'K': k,
        'Accuracy': results[k] * 100,
        'Time_Seconds': knn_times[k]
    })
df_knn_k = pd.DataFrame(knn_k_results)
df_knn_k.to_csv(os.path.join(output_dir, 'knn_k_values_comparison.csv'), index=False, encoding='utf-8-sig')
print("[OK] KNN K值比較已儲存: output/knn_k_values_comparison.csv")

# 5. 儲存 Random Forest 不同樹數量的結果
rf_n_results = []
for n in sorted(rf_results.keys()):
    rf_n_results.append({
        'n_estimators': n,
        'Accuracy': rf_results[n] * 100
    })
df_rf_n = pd.DataFrame(rf_n_results)
df_rf_n.to_csv(os.path.join(output_dir, 'random_forest_n_estimators_comparison.csv'), index=False, encoding='utf-8-sig')
print("[OK] Random Forest樹數量比較已儲存: output/random_forest_n_estimators_comparison.csv")

# 6. 儲存各類別在各分類器的準確率
class_acc_data = []
for clf_name, _, clf_pred, _, _ in all_classifiers_info:
    for cls in unique_classes:
        cls_mask = (y_test == cls)
        if np.sum(cls_mask) > 0:
            cls_acc = accuracy_score(y_test[cls_mask], clf_pred[cls_mask])
            class_acc_data.append({
                'Classifier': clf_name,
                'Class': cls,
                'Accuracy': cls_acc * 100,
                'Test_Samples': np.sum(cls_mask)
            })

df_class_acc = pd.DataFrame(class_acc_data)
df_class_acc_pivot = df_class_acc.pivot(index='Class', columns='Classifier', values='Accuracy')
df_class_acc_pivot.to_csv(os.path.join(output_dir, 'class_accuracy_by_classifier.csv'), encoding='utf-8-sig')
print("[OK] 各類別準確率已儲存: output/class_accuracy_by_classifier.csv")

# ==================== 儲存文字報告 ====================
print("\n" + "="*60)
print("儲存結果...")
print("="*60)

# 儲存預測結果（使用最佳分類器）
output_path = os.path.join(script_dir, "best_classifier_predictions.txt")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(f"使用分類器: {best_classifier_name}\n")
    f.write(f"準確率: {best_classifier_acc:.4f} ({best_classifier_acc*100:.2f}%)\n")
    f.write("="*60 + "\n")
    f.write("圖片索引,真實標籤,預測標籤,是否正確\n")
    for i, idx in enumerate(test_indices):
        original_idx = idx + 1  # 轉回1-based
        is_correct = "O" if y_test[i] == best_classifier_pred[i] else "X"
        f.write(f"{original_idx},{y_test[i]},{best_classifier_pred[i]},{is_correct}\n")

print(f"預測結果已儲存至: {output_path}")

# 儲存評估摘要
summary_path = os.path.join(script_dir, "classification_summary.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("多種分類器評估摘要\n")
    f.write("="*60 + "\n\n")
    f.write(f"資料集資訊:\n")
    f.write(f"  總圖片數: {len(features)}\n")
    f.write(f"  訓練集: {len(train_indices)}\n")
    f.write(f"  測試集: {len(test_indices)}\n")
    f.write(f"  類別數: {len(set(labels))}\n")
    f.write(f"  特徵數: {features.shape[1]}\n\n")
    
    f.write("="*60 + "\n")
    f.write("所有分類器效能比較\n")
    f.write("="*60 + "\n\n")
    
    for i, (name, acc, _) in enumerate(sorted_classifiers, 1):
        f.write(f"{i}. {name:30s}: {acc:.4f} ({acc*100:.2f}%)\n")
    
    f.write(f"\n最佳分類器: {best_classifier_name}\n")
    f.write(f"最佳準確率: {best_classifier_acc:.4f} ({best_classifier_acc*100:.2f}%)\n\n")
    
    f.write("="*60 + "\n")
    f.write("KNN 詳細資訊\n")
    f.write("="*60 + "\n\n")
    f.write(f"不同 K 值的準確率:\n")
    for k, acc in sorted(results.items()):
        f.write(f"  K={k:2d}: {acc:.4f} ({acc*100:.2f}%)\n")
    
    f.write(f"\n最佳 K 值: {best_k}\n")
    f.write(f"準確率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)\n")
    f.write(f"錯誤數: {np.sum(incorrect_mask)}/{len(y_test)}\n")
    f.write(f"錯誤率: {np.sum(incorrect_mask)/len(y_test)*100:.2f}%\n\n")
    
    f.write("="*60 + "\n")
    f.write("SVM 詳細資訊\n")
    f.write("="*60 + "\n\n")
    f.write(f"SVM (RBF) 準確率: {acc_svm_rbf:.4f} ({acc_svm_rbf*100:.2f}%)\n")
    f.write(f"SVM (Linear) 準確率: {acc_svm_linear:.4f} ({acc_svm_linear*100:.2f}%)\n\n")
    
    f.write("="*60 + "\n")
    f.write("LDA 詳細資訊\n")
    f.write("="*60 + "\n\n")
    f.write(f"LDA 準確率: {acc_lda:.4f} ({acc_lda*100:.2f}%)\n\n")
    
    f.write("="*60 + "\n")
    f.write("Random Forest 詳細資訊\n")
    f.write("="*60 + "\n\n")
    f.write(f"不同樹數量的準確率:\n")
    for n, acc in sorted(rf_results.items()):
        f.write(f"  n_estimators={n:3d}: {acc:.4f} ({acc*100:.2f}%)\n")
    f.write(f"\n最佳樹數量: {best_rf_n}\n")
    f.write(f"準確率: {best_rf_acc:.4f} ({best_rf_acc*100:.2f}%)\n")

print(f"評估摘要已儲存至: {summary_path}")

print("\n" + "="*60)
print("完成!")
print("="*60)
