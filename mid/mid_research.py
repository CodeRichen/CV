import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
from time import time
import cv2

class ImageRetrievalSystem:
    def __init__(self, data_dir, is_normalized=True):
        """
        初始化圖像檢索系統
        data_dir: 資料目錄路徑
        is_normalized: True 使用正規化資料，False 使用非正規化資料
        """
        self.data_dir = data_dir
        self.is_normalized = is_normalized
        self.features = []
        self.labels = []
        self.image_paths = []
        self.feature_ranges = {
            'ColorStructure': (0, 32),
            'ColorLayout': (32, 44),
            'RegionShape': (44, 80),
            'HomogeneousTexture': (336, 398),
            'EdgeHistogram': (398, 478)
        }
        
    def load_data(self):
        """載入所有特徵檔案"""
        print(f"載入{'正規化' if self.is_normalized else '非正規化'}資料...")
        
        txt_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        
        image_counter = 1  # 從1開始計數
        
        for txt_file in txt_files:
            class_name = txt_file.replace('.txt', '')
            file_path = os.path.join(self.data_dir, txt_file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 478:
                    continue
                
                features = np.array([float(x) for x in parts[:478]])
                # 使用編號作為圖片檔名，格式: 05位數.jpg
                image_filename = f"{image_counter:05d}.jpg"
                
                self.features.append(features)
                self.labels.append(class_name)
                self.image_paths.append(image_filename)
                
                image_counter += 1
        
        self.features = np.array(self.features)
        print(f"載入完成: {len(self.features)} 張圖像")
        
    def detailed_retrieval_analysis(self, start_k=1, max_k=None, method='cosine', 
                                   show_top_n=10, output_file=None):
        """
        詳細分析：每個 k 值使用第 k 張圖片作為查詢，找出最相似的前 10 張
        
        參數:
            start_k: 起始的 k 值（對應第幾張圖片）
            max_k: 最大的 k 值
            method: 相似度計算方法
            show_top_n: 顯示前 n 個最相似的圖像
            output_file: 輸出檔案路徑
        """
        
        if max_k is None or max_k >= len(self.features):
            max_k = len(self.features) - 1
        
        # 預先計算所有圖像的相似度矩陣
        start_time = time()
        if method == 'cosine':
            similarity_matrix = cosine_similarity(self.features)
        elif method == 'euclidean':
            distance_matrix = euclidean_distances(self.features)
            similarity_matrix = 1 / (1 + distance_matrix)
        elif method == 'pcc':
            similarity_matrix = np.corrcoef(self.features)
            similarity_matrix = np.nan_to_num(similarity_matrix, 0)
        
        accuracies = []
        
        # 準備輸出檔案
        if output_file:
            f_out = open(output_file, 'w', encoding='utf-8')
            f_out.write(f"方法: {method}\n")
            f_out.write(f"資料類型: {'正規化' if self.is_normalized else '非正規化'}\n")
            f_out.write("="*50 + "\n\n")
        
        # 從 k=start_k 開始
        for k in range(start_k, max_k + 1):
            # 檢查索引是否有效
            if k >= len(self.features):
                break
            
            query_idx = k  # 使用第 k 張圖片作為查詢
            query_label = self.labels[query_idx]
            
            # 輸出到檔案
            if output_file:
                f_out.write(f"k={k}\n")
            
            # 取得第 k 張圖片與所有圖片的相似度
            scores = similarity_matrix[query_idx].copy()
            scores[query_idx] = -np.inf  # 排除自己
            
            # 找出最相似的前 show_top_n 張
            top_n_indices = np.argsort(scores)[-show_top_n:][::-1]
            
            # 寫入檔案
            for idx in top_n_indices:
                if output_file:
                    f_out.write(f"第{idx}個\n")
            
            # 計算準確率（與當前查詢圖像（第 k 張）的同類別比例）
            correct = sum(1 for idx in top_n_indices if self.labels[idx] == query_label)
            accuracy = correct / show_top_n
            
            accuracies.append(accuracy)
            
            if output_file:
                f_out.write(f"準確率為{accuracy:.6f}\n")
                f_out.write("-"*33 + "\n")
        
        avg_accuracy = np.mean(accuracies)
        
        if output_file:
            f_out.write(f"\n平均準確率為{avg_accuracy:.6f}\n")
            f_out.close()
        
        return accuracies, avg_accuracy


# 主程式
if __name__ == "__main__":
    # 設定路徑
    normalized_dir = r"output_txt\normalized"
    non_normalized_dir = r"output_txt\non_normalized"
    
    # 設定輸出目錄
    output_dir = "retrieval_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 定義計算參數
    methods = ['pcc', 'cosine', 'euclidean']
    data_types = [
        ('normalized', normalized_dir, True),
        ('non_normalized', non_normalized_dir, False)
    ]
    
    print("\n開始圖像檢索分析...")
    
    results_summary = []
    
    # 遍歷所有組合
    for data_name, data_dir, is_normalized in data_types:
        # 載入系統
        system = ImageRetrievalSystem(data_dir, is_normalized=is_normalized)
        system.load_data()
        
        for method in methods:
            # 設定輸出檔案名稱
            output_file = os.path.join(output_dir, f"{data_name}_{method}.txt")
            
            # 執行分析
            accuracies, avg_accuracy = system.detailed_retrieval_analysis(
                start_k=1,
                max_k=10000,
                method=method,
                show_top_n=10,
                output_file=output_file
            )
            
            results_summary.append({
                'data_type': data_name,
                'method': method,
                'avg_accuracy': avg_accuracy
            })
    
    # 輸出總結報告
    print("\n" + "="*70)
    print("平均準確率總結")
    print("="*70)
    print(f"{'資料類型':<20} {'方法':<15} {'平均準確率':<15}")
    print("-"*70)
    
    for result in results_summary:
        print(f"{result['data_type']:<20} {result['method']:<15} {result['avg_accuracy']:.6f}")
    
    print("="*70)
    print(f"\n所有結果已儲存至目錄: {output_dir}")
    print("程式結束！")