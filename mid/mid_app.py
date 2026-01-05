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
        

    def query_and_display_images(self, query_position, image_base_dir, method='cosine', show_top_n=10):
        """
        查詢指定位置的圖片，並顯示相似度最高的前 N 張圖片
        
        參數:
            query_position: 圖片順位 (1-based, 例如 500)
            image_base_dir: 圖片基礎目錄 (如: "C:\\...\\ALL_PIC")
            method: 相似度計算方法
            show_top_n: 顯示前 n 個最相似的圖像（包含自己）
        """
        
        query_idx = query_position - 1  # 轉換為 0-based 索引
        
        if query_idx < 0 or query_idx >= len(self.features):
            print(f"錯誤：圖片順位 {query_position} 超出範圍 (1-{len(self.features)})")
            return
        
        query_label = self.labels[query_idx]
        query_filename = self.image_paths[query_idx]
        
        print(f"\n查詢圖像編號: {query_position}")
        print(f"類別: {query_label}")
        print(f"檔案名稱: {query_filename}")
        print(f"相似度計算方法: {method}")
        
        # 檢查圖片目錄
        if not os.path.exists(image_base_dir):
            print(f"錯誤：目錄不存在！{image_base_dir}")
            return
        
        # 計算相似度矩陣
        print("計算相似度中...")
        start_time = time()
        if method == 'cosine':
            similarity_matrix = cosine_similarity(self.features)
        elif method == 'euclidean':
            distance_matrix = euclidean_distances(self.features)
            similarity_matrix = 1 / (1 + distance_matrix)
        elif method == 'pcc':
            similarity_matrix = np.corrcoef(self.features)
            similarity_matrix = np.nan_to_num(similarity_matrix, 0)
        print(f"完成！耗時: {time() - start_time:.2f} 秒")
        
        # 取得相似度分數（包含自己）
        scores = similarity_matrix[query_idx].copy()
        
        # 找出最相似的前 show_top_n 張（包含自己）
        top_n_indices = np.argsort(scores)[-show_top_n:][::-1]
        
        print(f"\n最相似的前 {show_top_n} 張圖片（包含查詢圖片）:")
        print("-"*70)
        
        # 顯示結果
        for rank, idx in enumerate(top_n_indices, 1):
            is_query = " 查詢圖片" if idx == query_idx else ""
            print(f"{rank:2d}. 編號:{self.image_paths[idx]:<12} | 類別:{self.labels[idx]:<30} | 相似度:{scores[idx]:.6f}{is_query}")
        
        # 計算準確率
        correct = sum(1 for idx in top_n_indices if self.labels[idx] == query_label)
        accuracy = correct / show_top_n
        print("-"*70)
        print(f"準確率: {accuracy:.4f} ({correct}/{show_top_n} 張圖片與查詢圖片同類別)")
        print("-"*70)
        
        # 使用 matplotlib 顯示圖片
        print("\n正在載入圖片...")
        
        # 計算子圖佈局
        n_cols = min(5, show_top_n)
        n_rows = (show_top_n + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3.5))
        if show_top_n == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        loaded_count = 0
        for rank, idx in enumerate(top_n_indices):
            img_filename = self.image_paths[idx]
            full_path = os.path.join(image_base_dir, img_filename)
            
            ax = axes[rank]
            
            if os.path.exists(full_path):
                img = cv2.imread(full_path)
                if img is not None:
                    # OpenCV 讀取的是 BGR，轉換為 RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    
                    is_query = "\nSame No1" if idx == query_idx else ""
                    title = f"Rank {rank+1}: {img_filename}{is_query}\n{self.labels[idx]}\nSimilarity: {scores[idx]:.4f}"
                    ax.set_title(title, fontsize=9, fontweight='bold' if idx == query_idx else 'normal', 
                                color='red' if idx == query_idx else 'black')
                    ax.axis('off')
                    loaded_count += 1
                else:
                    ax.text(0.5, 0.5, f'無法讀取\n{img_filename}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'檔案不存在\n{img_filename}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        # 隱藏多餘的子圖
        for idx in range(show_top_n, len(axes)):
            axes[idx].axis('off')
        
        # 在圖片視窗標題顯示準確率
        fig.suptitle(f'Query number: {query_position} ({query_label}) | Method: {method.upper()} | Similarity: {accuracy:.4f} ({correct}/{show_top_n})', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print(f"成功載入 {loaded_count}/{show_top_n} 張圖片")
        print("圖片視窗已顯示，關閉視窗後可繼續查詢")
        
        return top_n_indices, scores[top_n_indices]
    


# 主程式
if __name__ == "__main__":
    # 設定路徑
    normalized_dir = r"output_txt\normalized"
    image_base_dir = r"ALL_PIC"
    
    # 載入系統
    system = ImageRetrievalSystem(normalized_dir, is_normalized=True)
    system.load_data()
    
    print("\n" + "="*70)
    print("圖像檢索系統")
    print("="*70)
    print(f"總共載入 {len(system.features)} 張圖像")
    print(f"類別數: {len(set(system.labels))}")
    print("="*70)
    
    # 無窮迴圈讓使用者持續查詢
    while True:
        print("\n請輸入要查詢的圖片編號 (例如: 9771)，或輸入 Q 退出:")
        user_input = input(">>> ").strip()
        
        # 檢查是否要退出
        if user_input.upper() == 'Q':
            print("\n感謝使用！再見！")
            break
        
        # 檢查輸入是否為數字
        try:
            query_position = int(user_input)
        except ValueError:
            print("錯誤：請輸入有效的數字或 Q 退出")
            continue
        
        # 檢查範圍
        if query_position < 1 or query_position > len(system.features):
            print(f"錯誤：圖片編號必須在 1 到 {len(system.features)} 之間")
            continue
        
        # 讓用戶選擇相似度計算方法
        print("\n請選擇相似度計算方法:")
        print("1. Cosine (餘弦相似度)")
        print("2. Euclidean (歐式距離)")
        print("3. PCC (皮爾森相關係數)")
        method_input = input("請輸入選項 (1/2/3): ").strip()
        
        # 根據選擇設定方法
        if method_input == '1':
            method = 'cosine'
        elif method_input == '2':
            method = 'euclidean'
        elif method_input == '3':
            method = 'pcc'
        else:
            print("錯誤：無效的選項，使用預設方法 (Cosine)")
            method = 'cosine'
        
        # 執行查詢並顯示結果
        print("\n" + "="*70)
        system.query_and_display_images(
            query_position=query_position,
            image_base_dir=image_base_dir,
            method=method,
            show_top_n=10
        )
        print("="*70)
    
    print("\n程式結束！")