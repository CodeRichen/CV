# ==================== 導入必要的庫 ====================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from PIL import Image
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ==================== 啟用 Eager Execution ====================
# 解決特徵提取時可能出現的 numpy() 不可用問題
tf.config.run_functions_eagerly(True)

# ==================== Google Colab 環境設定 ====================
from google.colab import drive
drive.mount('/content/drive')

# ==================== 建立輸出目錄 ====================
output_dir = '/content/block_outputs'

# 安全的路徑拼接函數，避免 Path 物件問題
def join_path(*parts):
    return os.path.join(*[str(p) for p in parts])

# 清空舊資料夾，確保乾淨輸出
if os.path.exists(output_dir):
    import shutil
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# ==================== 參數設定 ====================
image_size = (224, 224)
batch_size = 8
num_classes = 5


# ==================== 自行構建 VGG16 特徵提取器（不含分類頭） ====================
def build_custom_vgg16_features(input_shape):
    """
    使用 Keras 提供的基本層自行構建 VGG16 特徵提取器
    與完整 VGG16 的差異：移除全連接層（FC1, FC2, Softmax），改用 GlobalAveragePooling
    
    VGG16 卷積架構：
    - Block 1: 2 個 Conv(64, 3×3, ReLU) + MaxPool(2×2)
    - Block 2: 2 個 Conv(128, 3×3, ReLU) + MaxPool(2×2)
    - Block 3: 3 個 Conv(256, 3×3, ReLU) + MaxPool(2×2)
    - Block 4: 3 個 Conv(512, 3×3, ReLU) + MaxPool(2×2)
    - Block 5: 3 個 Conv(512, 3×3, ReLU) + MaxPool(2×2)
    - GlobalAveragePooling: 將 (7, 7, 512) 壓縮為 (512,) 特徵向量
    
    輸出：512 維特徵向量，用於 SVM 分類
    """
    
    model = models.Sequential()
    
    # ==================== Block 1 ====================
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    
    # ==================== Block 2 ====================
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    
    # ==================== Block 3 ====================
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    
    # ==================== Block 4 ====================
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    
    # ==================== Block 5 ====================
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
    
    # ==================== 特徵池化層（取代全連接層） ====================
    # 使用 GlobalAveragePooling2D 將 (7, 7, 512) 壓縮為 (512,)
    # 優點：無參數、降低過擬合、輸出固定維度特徵
    model.add(layers.GlobalAveragePooling2D(name='global_avg_pool'))
    
    return model


# ==================== 建立中間層特徵提取模型（用於可視化） ====================
def build_feature_extraction_model(base_model):
    """
    建立能夠輸出每個 Block 池化層特徵的子模型
    用於保存特徵圖影像（可視化 CNN 學到的特徵）
    
    參數:
        base_model: VGG16 特徵提取器模型
    
    返回:
        [(block_name, sub_model), ...] 列表
        每個 sub_model 輸出對應 block 的池化層結果
    """
    
    # 要提取的池化層名稱
    pool_layer_names = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']
    
    block_outputs = []
    
    # 判斷模型類型並獲取正確的 input
    if isinstance(base_model, models.Sequential):
        # Sequential 模型使用第一層的 input
        model_input = base_model.layers[0].input
    else:
        # Functional API 模型（官方 VGG16）直接使用 model.input
        model_input = base_model.input
    
    for block_name in pool_layer_names:
        # 尋找對應的層
        layer_found = False
        for i, layer in enumerate(base_model.layers):
            if layer.name == block_name:
                # 使用子模型提取從輸入到該層的特徵
                block_model = models.Model(
                    inputs=model_input,
                    outputs=layer.output
                )
                block_outputs.append((block_name.replace('_pool', ''), block_model))
                layer_found = True
                break
        
        if not layer_found:
            print(f"警告: 找不到層 {block_name}")
    
    return block_outputs


# ==================== 保存特徵圖為影像檔案 ====================
def save_feature_maps_all(features, block_name, image_name, output_dir):
    """
    將單張影像的所有特徵通道（filters）分別保存為灰階影像
    
    參數:
        features: 特徵張量，形狀 (batch_size, height, width, channels)
        block_name: Block 名稱（如 'block1'）
        image_name: 影像名稱（如 'test_image_0'）
        output_dir: 輸出根目錄
    
    輸出目錄結構: output_dir/blockX/image_name/filter_XXX.png
    """
    features = features[0]  # 取第一個樣本
    num_filters = features.shape[2]

    # 建立目錄
    save_dir = join_path(output_dir, block_name, image_name)
    os.makedirs(save_dir, exist_ok=True)

    # 保存每個 filter 的激活圖
    for i in range(num_filters):
        feature_map = features[:, :, i]
        # 正規化到 0-255 以便保存灰階圖
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        feature_map = (feature_map * 255).astype(np.uint8)
        img = Image.fromarray(feature_map, mode='L')
        img.save(join_path(save_dir, f'filter_{i:03d}.png'))

    print(f"已保存: {save_dir} (共 {num_filters} 個 filters)")


# ==================== 從資料集提取特徵的輔助函數 ====================
def extract_features(model, dataset):
    """
    使用 VGG16 特徵提取器從整個資料集中提取特徵
    
    參數:
        model: VGG16 特徵提取器（輸出 512 維向量）
        dataset: tf.data.Dataset 物件
    
    返回:
        features: numpy array, 形狀 (樣本數, 512)
        labels: numpy array, 形狀 (樣本數,)
    """
    features_list = []
    labels_list = []
    
    for images, labels in dataset:
        # 批次預測特徵
        batch_features = model.predict(images, verbose=0)
        features_list.append(batch_features)
        labels_list.append(labels.numpy())
    
    # 合併所有批次
    features = np.vstack(features_list)
    labels = np.hstack(labels_list)
    
    return features, labels


# ==================== 載入訓練和驗證資料 ====================
print("正在載入訓練和驗證資料...")

# 使用 label_mode='int' 以便直接獲取類別索引（SVM 需要整數標籤，不是 one-hot）
tdata = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/My Drive/CNN/train/',
    validation_split=0.1,
    subset="training",
    seed=7414,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='int'  # SVM 使用整數標籤
)

vdata = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/My Drive/CNN/train/',
    validation_split=0.1,
    subset="validation",
    seed=7414,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='int'
)

train_class_names = tdata.class_names
print("訓練集類別:", train_class_names)

# 優化資料管道
tdata = tdata.prefetch(buffer_size=batch_size)
vdata = vdata.prefetch(buffer_size=batch_size)

input_shape = (image_size[0], image_size[1], 3)


# ==================== 建立自定義 VGG16 特徵提取器並訓練 SVM ====================
print("\n" + "="*60)
print("建立自定義 VGG16 特徵提取器...")
print("="*60)

custom_vgg16_features = build_custom_vgg16_features(input_shape)
print("\n自定義 VGG16 特徵提取器結構:")
custom_vgg16_features.summary()

# 使用 VGG16 提取訓練集特徵
print("\n提取訓練集特徵...")
X_train_custom, y_train = extract_features(custom_vgg16_features, tdata)
print(f"訓練特徵形狀: {X_train_custom.shape}, 標籤形狀: {y_train.shape}")

# 使用 VGG16 提取驗證集特徵
print("\n提取驗證集特徵...")
X_val_custom, y_val = extract_features(custom_vgg16_features, vdata)
print(f"驗證特徵形狀: {X_val_custom.shape}, 標籤形狀: {y_val.shape}")

# 訓練 SVM 分類器
print("\n訓練 SVM 分類器（Custom VGG16 特徵）...")
print("SVM 參數: kernel='rbf', C=10, gamma='scale'")
svm_custom = SVC(kernel='rbf', C=10, gamma='scale', verbose=True, max_iter=1000)
svm_custom.fit(X_train_custom, y_train)

# 評估 SVM 性能
y_train_pred_custom = svm_custom.predict(X_train_custom)
y_val_pred_custom = svm_custom.predict(X_val_custom)

train_acc_custom = accuracy_score(y_train, y_train_pred_custom)
val_acc_custom = accuracy_score(y_val, y_val_pred_custom)

print(f"\n自定義 VGG16 + SVM 結果:")
print(f"訓練準確度: {train_acc_custom:.4f}")
print(f"驗證準確度: {val_acc_custom:.4f}")

# 保存模型
print("\n保存自定義模型...")
custom_vgg16_features.save('/content/drive/My Drive/CNN/custom_vgg16_features.keras')
with open('/content/drive/My Drive/CNN/svm_custom.pkl', 'wb') as f:
    pickle.dump(svm_custom, f)


# ==================== 建立官方 VGG16 特徵提取器並訓練 SVM ====================
print("\n" + "="*60)
print("建立官方 VGG16 特徵提取器（用於比較）...")
print("="*60)

official_vgg16_features = tf.keras.applications.VGG16(
    include_top=False,  # 不包含頂部的全連接層
    weights='imagenet',  # ✅ 使用 ImageNet 預訓練權重（1400萬張圖片訓練）
    input_shape=input_shape,
    pooling='avg'  # 使用全局平均池化，輸出 (512,) 特徵向量
)

print("\n官方 VGG16 特徵提取器結構:")
official_vgg16_features.summary()

# 使用官方 VGG16 提取訓練集特徵
print("\n提取訓練集特徵（Official VGG16）...")
X_train_official, _ = extract_features(official_vgg16_features, tdata)
print(f"訓練特徵形狀: {X_train_official.shape}")

# 使用官方 VGG16 提取驗證集特徵
print("\n提取驗證集特徵（Official VGG16）...")
X_val_official, _ = extract_features(official_vgg16_features, vdata)
print(f"驗證特徵形狀: {X_val_official.shape}")

# 訓練 SVM 分類器
print("\n訓練 SVM 分類器（Official VGG16 特徵）...")
print("SVM 參數: kernel='rbf', C=10, gamma='scale'")
svm_official = SVC(kernel='rbf', C=10, gamma='scale', verbose=True, max_iter=1000)
svm_official.fit(X_train_official, y_train)

# 評估 SVM 性能
y_train_pred_official = svm_official.predict(X_train_official)
y_val_pred_official = svm_official.predict(X_val_official)

train_acc_official = accuracy_score(y_train, y_train_pred_official)
val_acc_official = accuracy_score(y_val, y_val_pred_official)

print(f"\n官方 VGG16 + SVM 結果:")
print(f"訓練準確度: {train_acc_official:.4f}")
print(f"驗證準確度: {val_acc_official:.4f}")

# 保存模型
print("\n保存官方模型...")
official_vgg16_features.save('/content/drive/My Drive/CNN/official_vgg16_features.keras')
with open('/content/drive/My Drive/CNN/svm_official.pkl', 'wb') as f:
    pickle.dump(svm_official, f)


# ==================== 提取中間層特徵並保存為影像（可視化） ====================
print("\n" + "="*60)
print("提取中間層特徵並保存為影像...")
print("="*60)

# 從驗證集取得一個 batch 進行特徵可視化
test_batch = next(iter(vdata))
test_images = test_batch[0]

# 建立自定義模型的中間層特徵提取器
feature_models_custom = build_feature_extraction_model(custom_vgg16_features)

print("\n自定義 VGG16 特徵提取器的中間層特徵:")
for block_name, block_model in feature_models_custom:
    features = block_model.predict(test_images)
    print(f"{block_name} 輸出形狀: {features.shape}")
    
    # 保存前 2 張測試圖片的特徵圖
    for i, image in enumerate(test_images[:2]):
        image_name = f"test_image_{i}"
        save_feature_maps_all(features[i:i+1], block_name, image_name, output_dir)


# 建立官方模型的中間層特徵提取器
feature_models_official = build_feature_extraction_model(official_vgg16_features)

print("\n官方 VGG16 特徵提取器的中間層特徵:")
for block_name, block_model in feature_models_official:
    features = block_model.predict(test_images)
    print(f"{block_name} 輸出形狀: {features.shape}")
    
    # 保存前 2 張測試圖片的特徵圖
    for i, image in enumerate(test_images[:2]):
        image_name = f"test_image_{i}_official"
        save_feature_maps_all(features[i:i+1], block_name, image_name, output_dir)


# ==================== 比較兩個 VGG16+SVM 模型的結果 ====================
print("\n" + "="*60)
print("比較 VGG16+SVM 模型結果...")
print("="*60)

# 整理最終指標
final_custom = {
    'train_acc': train_acc_custom,
    'val_acc': val_acc_custom,
}
final_official = {
    'train_acc': train_acc_official,
    'val_acc': val_acc_official,
}

# 繪製準確度對比條狀圖
metrics = ['Train Acc', 'Val Acc']
custom_vals = [final_custom['train_acc'], final_custom['val_acc']]
official_vals = [final_official['train_acc'], final_official['val_acc']]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, custom_vals, width, label='Custom VGG16 + SVM', color='#2E86AB')
plt.bar(x + width/2, official_vals, width, label='Official VGG16 + SVM', color='#A23B72')
plt.xticks(x, metrics, rotation=0)
plt.title('VGG16 + SVM Performance Comparison')
plt.ylabel('Accuracy')
plt.ylim([0, 1.1])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 在條狀圖上顯示數值
for i, v in enumerate(custom_vals):
    plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
for i, v in enumerate(official_vals):
    plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)

bar_path = join_path(output_dir, 'svm_comparison_bar.png')
plt.tight_layout()
plt.savefig(bar_path, dpi=120, bbox_inches='tight')
plt.show()

# 輸出詳細的分類報告（每類別的 precision, recall, f1-score）
print("\n自定義 VGG16 + SVM 詳細報告:")
print(classification_report(y_val, y_val_pred_custom, target_names=train_class_names))

print("\n官方 VGG16 + SVM 詳細報告:")
print(classification_report(y_val, y_val_pred_official, target_names=train_class_names))

# 保存文字報告
report_path = join_path(output_dir, 'svm_comparison_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('VGG16 + SVM Model Comparison Report\n')
    f.write('====================================\n\n')
    
    f.write('Custom VGG16 + SVM (from scratch)\n')
    f.write(f"  Train Accuracy: {final_custom['train_acc']:.4f}\n")
    f.write(f"  Val Accuracy  : {final_custom['val_acc']:.4f}\n\n")
    
    f.write('Official VGG16 + SVM (from scratch)\n')
    f.write(f"  Train Accuracy: {final_official['train_acc']:.4f}\n")
    f.write(f"  Val Accuracy  : {final_official['val_acc']:.4f}\n\n")
    
    f.write('Architecture:\n')
    f.write('- Feature Extractor: VGG16 (without top layers, using GlobalAveragePooling)\n')
    f.write('- Classifier: SVM with RBF kernel (C=10, gamma=scale)\n')
    f.write('- Feature dimension: 512\n\n')
    
    f.write('Classification Report (Custom VGG16 + SVM):\n')
    f.write(classification_report(y_val, y_val_pred_custom, target_names=train_class_names))
    f.write('\n\n')
    
    f.write('Classification Report (Official VGG16 + SVM):\n')
    f.write(classification_report(y_val, y_val_pred_official, target_names=train_class_names))

print(f"\n✓ 所有輸出已保存至: {output_dir}")
print("✓ 自定義 VGG16 特徵提取器已保存")
print("✓ 官方 VGG16 特徵提取器已保存")
print("✓ SVM 模型已保存 (custom & official)")
print("✓ 特徵圖已保存")
print(f"✓ 對比條狀圖已保存: {bar_path}")
print(f"✓ 報告已保存: {report_path}")


# ==================== 打包並下載到個人電腦 ====================
import shutil
from google.colab import files

print("\n" + "="*60)
print("打包所有輸出檔案為 ZIP...")
print("="*60)

zip_filename = "/content/VGG16_SVM_Outputs"
shutil.make_archive(zip_filename, 'zip', output_dir)
print(f"ZIP 檔案已建立: {zip_filename}.zip")

files.download(f"{zip_filename}.zip")
print("✓ 瀏覽器應該會提示下載。如果沒有，請檢查 Colab 檔案面板。")
