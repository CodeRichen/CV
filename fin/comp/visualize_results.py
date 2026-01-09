"""
Classification Results Visualization Program
Reads data.txt and generates various charts for analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib
import warnings

# Ignore font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Set default font - use Arial for English
matplotlib.rc('font', family='Arial')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

print("Using font: Arial")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')
data_file = os.path.join(output_dir, 'data.txt')

# Read data
print("Reading data file...")
df = pd.read_csv(data_file, sep='\t')

# Extract method names
methods = df['Method'].values

# Extract numerical data
train_acc = df['Training_Accuracy'].values * 100
test_acc = df['Testing_Accuracy'].values * 100
overfit_gap = df['Overfit_Gap'].values * 100
precision = df['Precision'].values * 100
recall = df['Recall'].values * 100
f1_score = df['F1-Score'].values * 100
iou = df['IoU'].values * 100
std_acc = df['Std_Accuracy'].values * 100

# Set color scheme
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
method_colors = dict(zip(methods, colors))

print("Generating charts...")

# ============================================================================
# Chart 1: Training Accuracy vs Testing Accuracy Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width/2, train_acc, width, label='Training Accuracy', 
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, test_acc, width, label='Testing Accuracy', 
               color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 1,
            f'{height1:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 1,
            f'{height2:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Classification Method', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Training vs Testing Accuracy Comparison by Classification Method', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha='right')
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim([0, 110])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart1_train_vs_test_accuracy.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Chart 1 saved: chart1_train_vs_test_accuracy.png")
plt.close()

# ============================================================================
# Chart 2: Overfitting Gap Analysis
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(methods, overfit_gap, color=colors, alpha=0.7, 
              edgecolor='black', linewidth=1.5)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.2f}%', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

# Add warning lines (overfitting threshold)
ax.axhline(y=20, color='red', linestyle='--', linewidth=2, 
           label='Severe Overfitting Line (20%)', alpha=0.7)
ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, 
           label='Mild Overfitting Line (10%)', alpha=0.7)

ax.set_xlabel('Classification Method', fontsize=13, fontweight='bold')
ax.set_ylabel('Overfitting Gap (Training - Testing Accuracy) (%)', fontsize=13, fontweight='bold')
ax.set_title('Overfitting Analysis by Classification Method', fontsize=15, fontweight='bold', pad=20)
ax.set_xticklabels(methods, rotation=15, ha='right')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add description text
textstr = 'Note:\nLarger overfitting gap indicates the model performs\nwell on training data but poorly generalizes to test data'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart2_overfitting_analysis.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Chart 2 saved: chart2_overfitting_analysis.png")
plt.close()

# ============================================================================
# Chart 3: Comprehensive Performance Metrics Comparison (Radar Chart)
# ============================================================================
fig = plt.figure(figsize=(14, 10))

# Prepare radar chart data (using normalized values)
categories = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'IoU']
n_categories = len(categories)

# Calculate angles
angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
angles += angles[:1]

# Create subplots
for idx, method in enumerate(methods):
    ax = fig.add_subplot(2, 3, idx+1, projection='polar')
    
    # Prepare data
    values = [test_acc[idx], precision[idx], recall[idx], f1_score[idx], iou[idx]]
    values += values[:1]
    
    # Draw radar chart
    ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx], label=method)
    ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title(method, fontsize=12, fontweight='bold', pad=20)
    ax.grid(True)

# Add overall title
fig.suptitle('Comprehensive Performance Metrics Radar Chart by Method', fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart3_performance_radar.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Chart 3 saved: chart3_performance_radar.png")
plt.close()

# ============================================================================
# Chart 4: Performance Metrics Heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data matrix
metrics_data = np.array([
    test_acc,
    precision,
    recall,
    f1_score,
    iou,
    100 - overfit_gap  # Generalization ability (100 - overfitting gap)
])

metrics_names = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'IoU', 'Generalization']

# Draw heatmap
im = ax.imshow(metrics_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Set axis labels
ax.set_xticks(np.arange(len(methods)))
ax.set_yticks(np.arange(len(metrics_names)))
ax.set_xticklabels(methods, rotation=15, ha='right')
ax.set_yticklabels(metrics_names)

# Add value labels
for i in range(len(metrics_names)):
    for j in range(len(methods)):
        text = ax.text(j, i, f'{metrics_data[i, j]:.1f}%',
                      ha="center", va="center", color="black", 
                      fontsize=11, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score (%)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

ax.set_title('Performance Metrics Heatmap by Classification Method', fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart4_metrics_heatmap.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Chart 4 saved: chart4_metrics_heatmap.png")
plt.close()

# ============================================================================
# Chart 5: Stability Analysis (Standard Deviation Comparison)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(methods, std_acc, color=colors, alpha=0.7, 
              edgecolor='black', linewidth=1.5)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.2f}%', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

ax.set_xlabel('Classification Method', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy Standard Deviation (%)', fontsize=13, fontweight='bold')
ax.set_title('Stability Analysis (Standard Deviation of Class Accuracy)', fontsize=15, fontweight='bold', pad=20)
ax.set_xticklabels(methods, rotation=15, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Add description text
textstr = 'Note:\nSmaller standard deviation indicates more balanced\nclassification ability across different classes.\nLarger values indicate strong performance on some\nclasses but weak on others.'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart5_stability_analysis.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Chart 5 saved: chart5_stability_analysis.png")
plt.close()

# ============================================================================
# Chart 6: Best and Worst Classification Classes Analysis
# ============================================================================
fig, axes = plt.subplots(5, 2, figsize=(16, 20))

for idx, method in enumerate(methods):
    # Parse best and worst classes
    best_classes = []
    worst_classes = []
    
    for i in range(1, 4):
        best_col = f'Best_Class_{i}'
        worst_col = f'Worst_Class_{i}'
        
        best_str = df.loc[idx, best_col]
        worst_str = df.loc[idx, worst_col]
        
        # Parse format "ClassName (accuracy)"
        best_name = best_str.split('(')[0].strip()
        best_acc = float(best_str.split('(')[1].strip(')')) * 100
        best_classes.append((best_name, best_acc))
        
        worst_name = worst_str.split('(')[0].strip()
        worst_acc = float(worst_str.split('(')[1].strip(')')) * 100
        worst_classes.append((worst_name, worst_acc))
    
    # Left side: Best 3 classes
    ax_left = axes[idx, 0]
    best_names = [c[0] for c in best_classes]
    best_accs = [c[1] for c in best_classes]
    
    bars = ax_left.barh(best_names, best_accs, color='green', alpha=0.7, 
                        edgecolor='black', linewidth=1.5)
    
    for i, (bar, acc) in enumerate(zip(bars, best_accs)):
        width = bar.get_width()
        ax_left.text(width + 1, bar.get_y() + bar.get_height()/2.,
                    f'{acc:.2f}%', ha='left', va='center', 
                    fontsize=10, fontweight='bold')
    
    ax_left.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax_left.set_title(f'{method} - Best 3 Classes', fontsize=12, fontweight='bold')
    ax_left.set_xlim([0, 110])
    ax_left.grid(True, alpha=0.3, axis='x')
    
    # Right side: Worst 3 classes
    ax_right = axes[idx, 1]
    worst_names = [c[0] for c in worst_classes]
    worst_accs = [c[1] for c in worst_classes]
    
    bars = ax_right.barh(worst_names, worst_accs, color='red', alpha=0.7, 
                         edgecolor='black', linewidth=1.5)
    
    for i, (bar, acc) in enumerate(zip(bars, worst_accs)):
        width = bar.get_width()
        ax_right.text(width + 1, bar.get_y() + bar.get_height()/2.,
                     f'{acc:.2f}%', ha='left', va='center', 
                     fontsize=10, fontweight='bold')
    
    ax_right.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax_right.set_title(f'{method} - Worst 3 Classes', fontsize=12, fontweight='bold')
    ax_right.set_xlim([0, 110])
    ax_right.grid(True, alpha=0.3, axis='x')

fig.suptitle('Best and Worst Classification Classes by Method', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart6_best_worst_classes.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Chart 6 saved: chart6_best_worst_classes.png")
plt.close()

# ============================================================================
# Chart 7: Overall Ranking Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Calculate overall score (weighted average)
weights = {
    'test_acc': 0.30,
    'precision': 0.15,
    'recall': 0.15,
    'f1_score': 0.15,
    'iou': 0.10,
    'generalization': 0.15  # 100 - overfit_gap
}

generalization_score = 100 - overfit_gap
overall_scores = (
    test_acc * weights['test_acc'] +
    precision * weights['precision'] +
    recall * weights['recall'] +
    f1_score * weights['f1_score'] +
    iou * weights['iou'] +
    generalization_score * weights['generalization']
)

# Sort
sorted_indices = np.argsort(overall_scores)[::-1]
sorted_methods = methods[sorted_indices]
sorted_scores = overall_scores[sorted_indices]
sorted_colors = [colors[i] for i in sorted_indices]

# Draw bar chart
bars = ax.barh(sorted_methods, sorted_scores, color=sorted_colors, 
               alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels and rankings
for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
    width = bar.get_width()
    # Score
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
            f'{score:.2f}', ha='left', va='center', 
            fontsize=11, fontweight='bold')
    # Rank
    ax.text(1, bar.get_y() + bar.get_height()/2.,
            f'#{i+1}', ha='left', va='center', 
            fontsize=12, fontweight='bold', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

ax.set_xlabel('Overall Score', fontsize=13, fontweight='bold')
ax.set_title('Overall Ranking by Classification Method (Weighted Average Score)', fontsize=15, fontweight='bold', pad=20)
ax.set_xlim([0, 100])
ax.grid(True, alpha=0.3, axis='x')

# Add weight configuration
weight_text = 'Weight Configuration:\n'
weight_text += f"• Test Accuracy: {weights['test_acc']*100:.0f}%\n"
weight_text += f"• Precision: {weights['precision']*100:.0f}%\n"
weight_text += f"• Recall: {weights['recall']*100:.0f}%\n"
weight_text += f"• F1-Score: {weights['f1_score']*100:.0f}%\n"
weight_text += f"• IoU: {weights['iou']*100:.0f}%\n"
weight_text += f"• Generalization: {weights['generalization']*100:.0f}%"

props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax.text(0.98, 0.03, weight_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart7_overall_ranking.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Chart 7 saved: chart7_overall_ranking.png")
plt.close()

# ============================================================================
# Generate chart description document
# ============================================================================
summary_file = os.path.join(output_dir, 'charts_summary.txt')

with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("Chart Description Documentation\n")
    f.write("="*80 + "\n\n")
    
    f.write("Chart 1: Training vs Testing Accuracy Comparison (chart1_train_vs_test_accuracy.png)\n")
    f.write("-" * 80 + "\n")
    f.write("Description: Compares accuracy performance on training and testing sets for each method\n")
    f.write("Key Observations:\n")
    f.write("  • SVM achieves extremely high training accuracy of 99.93%\n")
    f.write("  • KNN and LDA show smaller gaps between training/testing accuracy, better generalization\n")
    f.write("  • Random Forest has high training but relatively lower testing accuracy\n\n")
    
    f.write("Chart 2: Overfitting Gap Analysis (chart2_overfitting_analysis.png)\n")
    f.write("-" * 80 + "\n")
    f.write("Description: Shows the degree of overfitting for each method (Training - Testing Accuracy)\n")
    f.write("Key Observations:\n")
    f.write(f"  • Random Forest has the most severe overfitting: {overfit_gap[4]:.2f}%\n")
    f.write(f"  • KNN has the mildest overfitting: {overfit_gap[0]:.2f}%\n")
    f.write(f"  • LDA also shows good generalization ability: {overfit_gap[3]:.2f}%\n")
    f.write("  • Gap over 20% indicates severe overfitting, requires model parameter adjustment\n\n")
    
    f.write("Chart 3: Comprehensive Performance Metrics Radar Chart (chart3_performance_radar.png)\n")
    f.write("-" * 80 + "\n")
    f.write("Description: Displays performance across multiple metrics in radar chart format\n")
    f.write("Key Observations:\n")
    f.write("  • Larger radar area indicates better overall performance\n")
    f.write("  • SVM (RBF) has the best performance on most metrics\n")
    f.write("  • IoU metrics are generally low, indicating room for improvement in classification precision\n\n")
    
    f.write("Chart 4: Performance Metrics Heatmap (chart4_metrics_heatmap.png)\n")
    f.write("-" * 80 + "\n")
    f.write("Description: Uses color intensity to show scores across different metrics\n")
    f.write("Key Observations:\n")
    f.write("  • Green indicates high scores, red indicates low scores\n")
    f.write("  • Generalization ability = 100 - overfitting gap\n")
    f.write("  • Quickly identifies strengths and weaknesses of each method\n\n")
    
    f.write("Chart 5: Stability Analysis (chart5_stability_analysis.png)\n")
    f.write("-" * 80 + "\n")
    f.write("Description: Shows the balance of classification ability across different classes\n")
    f.write("Key Observations:\n")
    f.write(f"  • SVM (RBF) has lowest std dev: {std_acc[1]:.2f}%, most balanced classification\n")
    f.write(f"  • KNN has highest std dev: {std_acc[0]:.2f}%, large variation across classes\n")
    f.write("  • Smaller std dev indicates more consistent handling of all classes\n\n")
    
    f.write("Chart 6: Best and Worst Classification Classes Analysis (chart6_best_worst_classes.png)\n")
    f.write("-" * 80 + "\n")
    f.write("Description: Lists the top 3 best and worst performing classes for each method\n")
    f.write("Key Observations:\n")
    f.write("  • AncestorDinoArt and BWimage are easily classified by most methods\n")
    f.write("  • Sculpt and Desert are difficult for most methods to classify\n")
    f.write("  • Allows targeted feature engineering for poorly performing classes\n\n")
    
    f.write("圖表7: 綜合排名比較 (chart7_overall_ranking.png)\n")
    f.write("-" * 80 + "\n")
    f.write("說明：根據加權平均計算各方法的綜合得分並排名\n")
    f.write("重點觀察：\n")
    for i, idx in enumerate(sorted_indices):
        f.write(f"  #{i+1}: {methods[idx]} (得分: {overall_scores[idx]:.2f})\n")
    f.write("\n權重配置：測試準確率30%、各項指標各15%、泛化能力15%\n\n")
    
    f.write("="*80 + "\n")
    f.write("總結與建議\n")
    f.write("="*80 + "\n")
    f.write(f"1. 最佳整體性能: {methods[sorted_indices[0]]} (綜合得分: {overall_scores[sorted_indices[0]]:.2f})\n")
    f.write(f"   - 測試準確率: {test_acc[sorted_indices[0]]:.2f}%\n")
    f.write(f"   - 泛化能力: {generalization_score[sorted_indices[0]]:.2f}\n\n")
    
    f.write(f"2. 最佳泛化能力: {methods[np.argmin(overfit_gap)]} (過擬合差距: {np.min(overfit_gap):.2f}%)\n")
    f.write(f"   - 訓練準確率: {train_acc[np.argmin(overfit_gap)]:.2f}%\n")
    f.write(f"   - 測試準確率: {test_acc[np.argmin(overfit_gap)]:.2f}%\n\n")
    
    f.write(f"3. 最高測試準確率: {methods[np.argmax(test_acc)]} ({np.max(test_acc):.2f}%)\n\n")
    
    f.write(f"4. 最均衡分類能力: {methods[np.argmin(std_acc)]} (標準差: {np.min(std_acc):.2f}%)\n\n")
    
    f.write("5. 改進建議:\n")
    f.write("   - 針對過擬合嚴重的模型（Random Forest, SVM），考慮:\n")
    f.write("     * 增加正則化強度\n")
    f.write("     * 減少模型複雜度\n")
    f.write("     * 使用交叉驗證調整超參數\n")
    f.write("   - 針對表現差的類別（Sculpt, Desert等），考慮:\n")
    f.write("     * 增加該類別的訓練樣本\n")
    f.write("     * 進行特徵工程和特徵選擇\n")
    f.write("     * 使用數據增強技術\n")
    f.write("   - 考慮集成學習方法結合多個模型的優勢\n\n")

print(f"\n✓ Chart description document saved: charts_summary.txt")

print("\n" + "="*80)
print("All charts generated successfully!")
print("="*80)
print(f"Total of 7 charts saved to: {output_dir}")
print("\nChart List:")
print("  1. chart1_train_vs_test_accuracy.png - Training vs Testing Accuracy Comparison")
print("  2. chart2_overfitting_analysis.png - Overfitting Gap Analysis")
print("  3. chart3_performance_radar.png - Comprehensive Performance Metrics Radar Chart")
print("  4. chart4_metrics_heatmap.png - Performance Metrics Heatmap")
print("  5. chart5_stability_analysis.png - Stability Analysis")
print("  6. chart6_best_worst_classes.png - Best and Worst Classification Classes")
print("  7. chart7_overall_ranking.png - Overall Ranking Comparison")
print("\nFor detailed descriptions, see: charts_summary.txt")
print("="*80)
