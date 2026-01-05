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

# ==================== 啟用 Eager Execution ====================
# 解決 numpy() 不可用的問題
tf.config.run_functions_eagerly(True)

# ==================== Google Colab 環境設定 ====================
from google.colab import drive
drive.mount('/content/drive')

# ==================== 建立輸出目錄 (改為本機 /content/) ====================
output_dir = '/content/block_outputs'

# Safe path join to handle str/Path inputs and avoid pathlib issues
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
epochs = 10
num_classes = 5


# ==================== 自行構建 VGG16 模型 ====================
def build_custom_vgg16(input_shape, num_classes):
    """
    使用 Keras 提供的基本層自行構建 VGG16 模型
    
    VGG16 架構：
    - Block 1: 2 個 Conv(64) + MaxPool
    - Block 2: 2 個 Conv(128) + MaxPool
    - Block 3: 3 個 Conv(256) + MaxPool
    - Block 4: 3 個 Conv(512) + MaxPool
    - Block 5: 3 個 Conv(512) + MaxPool
    - 全連接層 + 分類層
    """
    
    model = models.Sequential()
    
    # ==================== Block 1 ====================
    # 輸入層
    model.add(layers.Input(shape=input_shape))
    
    # Block 1: 2 個卷積層 (64 個篩選器) + 最大池化
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    
    # ==================== Block 2 ====================
    # Block 2: 2 個卷積層 (128 個篩選器) + 最大池化
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    
    # ==================== Block 3 ====================
    # Block 3: 3 個卷積層 (256 個篩選器) + 最大池化
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    
    # ==================== Block 4 ====================
    # Block 4: 3 個卷積層 (512 個篩選器) + 最大池化
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    
    # ==================== Block 5 ====================
    # Block 5: 3 個卷積層 (512 個篩選器) + 最大池化
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
    
    # ==================== 分類頭部 ====================
    # 攤平層：將高維特徵圖轉換為 1D 向量
    model.add(layers.Flatten(name='flatten'))
    
    # 全連接層
    model.add(layers.Dense(4096, activation='relu', name='fc1'))
    model.add(layers.Dense(4096, activation='relu', name='fc2'))
    
    # 輸出層（使用 softmax 激活用於多分類）
    model.add(layers.Dense(num_classes, activation='softmax', name='predictions'))
    
    return model


# ==================== 建立特徵提取模型（用於保存中間層輸出） ====================
def build_feature_extraction_model(base_model):
    """
    建立能夠輸出每個 Block 特徵的模型
    使用層的名稱來提取特徵
    """
    
    # 找出每個 block 的池化層索引
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
                # 使用子模型提取特徵
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


# ==================== 保存特徵圖為影像 ====================
def save_feature_maps_all(features, block_name, image_name, output_dir):
    """
    將單張影像的所有特徵圖（每個 filter 一張）保存到對應資料夾。
    features: 特徵張量，形狀 (batch_size, height, width, channels)
    目錄結構：output_dir/blockX/image_name/filter_XXX.png
    """
    features = features[0]  # 取第一個樣本
    num_filters = features.shape[2]

    # 建立目錄：/output_dir/blockN/image_name/
    save_dir = join_path(output_dir, block_name, image_name)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_filters):
        feature_map = features[:, :, i]
        # 正規化到 0-255 以便保存灰階圖
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        feature_map = (feature_map * 255).astype(np.uint8)
        img = Image.fromarray(feature_map, mode='L')
        img.save(join_path(save_dir, f'filter_{i:03d}.png'))

    print(f"已保存: {save_dir} (filters={num_filters})")


# ==================== 保存訓練歷史為 CSV ====================
def save_history_csv(history, file_path):
    """Save training history (all metrics) to CSV with an epoch column."""
    keys = list(history.history.keys())
    num_epochs = len(history.history[keys[0]]) if keys else 0
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch'] + keys)
        for i in range(num_epochs):
            writer.writerow([i + 1] + [history.history[k][i] for k in keys])
    print(f"已保存歷史: {file_path}")


# ==================== 從目錄載入訓練和驗證資料 ====================
print("正在載入訓練和驗證資料...")

tdata = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/My Drive/CNN/train/',
    validation_split=0.1,
    subset="training",
    seed=7414,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

vdata = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/My Drive/CNN/train/',
    validation_split=0.1,
    subset="validation",
    seed=7414,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

train_class_names = tdata.class_names
print("訓練集類別:", train_class_names)

# 優化資料管道
tdata = tdata.prefetch(buffer_size=batch_size)
vdata = vdata.prefetch(buffer_size=batch_size)

input_shape = (image_size[0], image_size[1], 3)


# ==================== 建立和訓練自定義 VGG16 模型 ====================
print("\n" + "="*60)
print("建立自定義 VGG16 模型...")
print("="*60)

custom_vgg16 = build_custom_vgg16(input_shape, num_classes)
print("\n自定義 VGG16 模型結構:")
custom_vgg16.summary()

# 編譯模型 - 為自定義模型創建優化器實例
optimizer_custom = tf.keras.optimizers.Adam(learning_rate=1e-4)
custom_vgg16.compile(
    optimizer=optimizer_custom,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# 訓練自定義 VGG16
print("\n開始訓練自定義 VGG16...")
history_custom = custom_vgg16.fit(
    tdata,
    epochs=epochs,
    validation_data=vdata,
    batch_size=batch_size
)

print("\n訓練完成！保存模型...")
custom_vgg16.save('/content/drive/My Drive/CNN/custom_vgg16_model.keras')


# ==================== 建立官方 VGG16 模型用於比較 ====================
print("\n" + "="*60)
print("建立官方 VGG16 模型（用於比較）...")
print("="*60)

official_vgg16 = tf.keras.applications.VGG16(
    include_top=True,
    weights=None,  # 不載入預訓練權重，從零開始訓練
    input_shape=input_shape,
    classes=num_classes,
)

print("\n官方 VGG16 模型結構:")
official_vgg16.summary()

# 編譯模型 - 為官方模型創建獨立的優化器實例
optimizer_official = tf.keras.optimizers.Adam(learning_rate=1e-4)
official_vgg16.compile(
    optimizer=optimizer_official,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# 訓練官方 VGG16
print("\n開始訓練官方 VGG16...")
history_official = official_vgg16.fit(
    tdata,
    epochs=epochs,
    validation_data=vdata,
    batch_size=batch_size
)

print("\n訓練完成！保存模型...")
official_vgg16.save('/content/drive/My Drive/CNN/official_vgg16_model.keras')


# ==================== 提取測試圖片並生成特徵圖 ====================
print("\n" + "="*60)
print("提取中間層特徵並保存為影像...")
print("="*60)

# 從驗證集取得一個 batch 進行特徵提取
test_batch = next(iter(vdata))
test_images = test_batch[0]

# 建立自定義模型的特徵提取器
feature_models_custom = build_feature_extraction_model(custom_vgg16)

print("\n自定義 VGG16 模型的特徵提取:")
for block_name, block_model in feature_models_custom:
    features = block_model.predict(test_images)
    print(f"{block_name} 輸出形狀: {features.shape}")
    
    # 保存每個 block 的全部特徵圖（每個 filter 一張）
    for i, image in enumerate(test_images[:2]):  # 只保存前 2 張測試圖片
        image_name = f"test_image_{i}"
        save_feature_maps_all(features[i:i+1], block_name, image_name, output_dir)


# 建立官方模型的特徵提取器
feature_models_official = build_feature_extraction_model(official_vgg16)

print("\n官方 VGG16 模型的特徵提取:")
for block_name, block_model in feature_models_official:
    features = block_model.predict(test_images)
    print(f"{block_name} 輸出形狀: {features.shape}")
    
    # 保存每個 block 的全部特徵圖（每個 filter 一張）
    for i, image in enumerate(test_images[:2]):  # 只保存前 2 張測試圖片
        image_name = f"test_image_{i}_official"
        save_feature_maps_all(features[i:i+1], block_name, image_name, output_dir)


# ==================== 比較兩個模型的訓練結果 ====================
print("\n" + "="*60)
print("Comparing model training results...")
print("="*60)

# 保存歷史到 CSV，便於報告與數據分析
save_history_csv(history_custom, join_path(output_dir, 'history_custom.csv'))
save_history_csv(history_official, join_path(output_dir, 'history_official.csv'))

# 取最終與最佳指標
final_custom = {
    'train_acc': history_custom.history['accuracy'][-1],
    'val_acc': history_custom.history['val_accuracy'][-1],
    'train_loss': history_custom.history['loss'][-1],
    'val_loss': history_custom.history['val_loss'][-1],
    'best_val_acc': max(history_custom.history['val_accuracy']),
}
final_official = {
    'train_acc': history_official.history['accuracy'][-1],
    'val_acc': history_official.history['val_accuracy'][-1],
    'train_loss': history_official.history['loss'][-1],
    'val_loss': history_official.history['val_loss'][-1],
    'best_val_acc': max(history_official.history['val_accuracy']),
}

# 曲線圖（英文標籤）
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history_custom.history['accuracy'], label='Custom VGG16 Train')
plt.plot(history_custom.history['val_accuracy'], label='Custom VGG16 Val')
plt.plot(history_official.history['accuracy'], label='Official VGG16 Train')
plt.plot(history_official.history['val_accuracy'], label='Official VGG16 Val')
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_custom.history['loss'], label='Custom VGG16 Train')
plt.plot(history_custom.history['val_loss'], label='Custom VGG16 Val')
plt.plot(history_official.history['loss'], label='Official VGG16 Train')
plt.plot(history_official.history['val_loss'], label='Official VGG16 Val')
plt.title('Loss Comparison')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
curve_path = join_path(output_dir, 'model_comparison.png')
plt.savefig(curve_path, dpi=120, bbox_inches='tight')
plt.show()

# 終值對比條狀圖（英文標籤）
metrics = ['Train Acc', 'Val Acc', 'Train Loss', 'Val Loss']
custom_vals = [
    final_custom['train_acc'],
    final_custom['val_acc'],
    final_custom['train_loss'],
    final_custom['val_loss'],
]
official_vals = [
    final_official['train_acc'],
    final_official['val_acc'],
    final_official['train_loss'],
    final_official['val_loss'],
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, custom_vals, width, label='Custom VGG16')
plt.bar(x + width/2, official_vals, width, label='Official VGG16')
plt.xticks(x, metrics, rotation=15)
plt.title('Final Metrics (Last Epoch)')
plt.ylabel('Value')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
bar_path = join_path(output_dir, 'final_metrics_bar.png')
plt.tight_layout()
plt.savefig(bar_path, dpi=120, bbox_inches='tight')
plt.show()

# 輸出
report_path = join_path(output_dir, 'comparison_report.txt')
with open(report_path, 'w') as f:
    f.write('Model Comparison Report\n')
    f.write('======================\n\n')
    f.write('Custom VGG16 (from scratch)\n')
    f.write(f"  Final Train Acc : {final_custom['train_acc']:.4f}\n")
    f.write(f"  Final Val   Acc : {final_custom['val_acc']:.4f}\n")
    f.write(f"  Best  Val   Acc : {final_custom['best_val_acc']:.4f}\n")
    f.write(f"  Final Train Loss: {final_custom['train_loss']:.4f}\n")
    f.write(f"  Final Val   Loss: {final_custom['val_loss']:.4f}\n\n")

    f.write('Official VGG16 (from scratch)\n')
    f.write(f"  Final Train Acc : {final_official['train_acc']:.4f}\n")
    f.write(f"  Final Val   Acc : {final_official['val_acc']:.4f}\n")
    f.write(f"  Best  Val   Acc : {final_official['best_val_acc']:.4f}\n")
    f.write(f"  Final Train Loss: {final_official['train_loss']:.4f}\n")
    f.write(f"  Final Val   Loss: {final_official['val_loss']:.4f}\n\n")

    f.write('Notes:\n')
    f.write('- All metrics are from the last epoch unless marked as "Best".\n')
    f.write('- For pretrained weights, set weights="imagenet" in the official model and apply preprocess_input.\n')

print(f"\n✓ All outputs saved to: {output_dir}")
print("✓ Custom VGG16 model saved")
print("✓ Official VGG16 model saved")
print("✓ Feature maps saved")
print(f"✓ Curves saved: {curve_path}")
print(f"✓ Final metrics bar saved: {bar_path}")
print(f"✓ Histories saved: history_custom.csv, history_official.csv")
print(f"✓ Report saved: {report_path}")


# ==================== 打包並下載到個人電腦 ====================
import shutil
from google.colab import files

print("\n" + "="*60)
print("Packing all outputs into ZIP...")
print("="*60)

zip_filename = "/content/VGG16_Outputs"
shutil.make_archive(zip_filename, 'zip', output_dir)
print(f"ZIP created: {zip_filename}.zip")

files.download(f"{zip_filename}.zip")
print("✓ Browser should prompt download. If not, check Colab file panel.")
