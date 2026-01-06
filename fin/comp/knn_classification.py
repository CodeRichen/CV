import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import Counter
import time

# ==================== 載入資料 ====================
print("載入資料中...")

# 載入特徵資料 (10000張圖, 478個特徵)
data_path = "fnormal.txt"
features = []
with open(data_path, 'r') as f:
    for line in f:
        features.append([float(x) for x in line.strip().split()])

features = np.array(features)
print(f"特徵資料維度: {features.shape}")

# 載入標籤資料 (每張圖的類別)
import os
name_path = os.path.join(os.path.expanduser("~"), "Downloads", "class", "114-1_電腦視覺", "final report - 1", "name.txt")
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
query_path = os.path.join(os.path.expanduser("~"), "Downloads", "class", "114-1_電腦視覺", "final report - 1", "query.txt")
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

for k in k_values:
    print(f"\n測試 K={k}...")
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=1)
    knn.fit(X_train_scaled, y_train)
    
    # 預測
    y_pred = knn.predict(X_test_scaled)
    
    # 計算準確率
    accuracy = accuracy_score(y_test, y_pred)
    results[k] = accuracy
    print(f"  準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print("\n" + "="*60)
print(f"最佳 K 值: {best_k}")
print(f"最佳準確率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print("="*60)

# ==================== 使用最佳 K 值進行詳細評估 ====================
print(f"\n使用 K={best_k} 進行詳細評估...")
knn_best = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', n_jobs=1)
knn_best.fit(X_train_scaled, y_train)
y_pred_best = knn_best.predict(X_test_scaled)

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
print(f"SVM (RBF) 準確率: {acc_svm_rbf:.4f} ({acc_svm_rbf*100:.2f}%) - 訓練時間: {time_svm_rbf:.2f}秒")

print("\n訓練 SVM (Linear kernel)...")
start_time = time.time()
svm_linear = SVC(kernel='linear', C=1, random_state=42)
svm_linear.fit(X_train_scaled, y_train)
y_pred_svm_linear = svm_linear.predict(X_test_scaled)
acc_svm_linear = accuracy_score(y_test, y_pred_svm_linear)
time_svm_linear = time.time() - start_time
print(f"SVM (Linear) 準確率: {acc_svm_linear:.4f} ({acc_svm_linear*100:.2f}%) - 訓練時間: {time_svm_linear:.2f}秒")

# ==================== LDA 分類 ====================
print("\n" + "="*60)
print("開始 LDA 分類...")
print("="*60)

start_time = time.time()
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)
y_pred_lda = lda.predict(X_test_scaled)
acc_lda = accuracy_score(y_test, y_pred_lda)
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
    y_pred_rf = rf.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    time_rf = time.time() - start_time
    rf_results[n] = acc_rf
    print(f"  準確率: {acc_rf:.4f} ({acc_rf*100:.2f}%) - 訓練時間: {time_rf:.2f}秒")
    
    if acc_rf > best_rf_acc:
        best_rf_acc = acc_rf
        best_rf_n = n
        best_rf_model = rf
        y_pred_rf_best = y_pred_rf

print(f"\n最佳 Random Forest 參數: n_estimators={best_rf_n}")
print(f"最佳 Random Forest 準確率: {best_rf_acc:.4f} ({best_rf_acc*100:.2f}%)")

# ==================== 所有分類器比較 ====================
print("\n" + "="*60)
print("所有分類器效能比較")
print("="*60)

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

# ==================== 儲存結果 ====================
print("\n" + "="*60)
print("儲存結果...")
print("="*60)

# 儲存預測結果（使用最佳分類器）
output_path = "best_classifier_predictions.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(f"使用分類器: {best_classifier_name}\n")
    f.write(f"準確率: {best_classifier_acc:.4f} ({best_classifier_acc*100:.2f}%)\n")
    f.write("="*60 + "\n")
    f.write("圖片索引,真實標籤,預測標籤,是否正確\n")
    for i, idx in enumerate(test_indices):
        original_idx = idx + 1  # 轉回1-based
        is_correct = "✓" if y_test[i] == best_classifier_pred[i] else "✗"
        f.write(f"{original_idx},{y_test[i]},{best_classifier_pred[i]},{is_correct}\n")

print(f"預測結果已儲存至: {output_path}")

# 儲存評估摘要
summary_path = "classification_summary.txt"
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
