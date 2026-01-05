#導入需要的程式庫
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization as bn
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K
import tensorflow as tf
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import keras
from random import shuffle

# ==================== 掛載 Google Drive ====================
from google.colab import drive
drive.mount('/content/drive')

# ==================== 載入訓練資料 ====================
#load train data
total_patch = np.load("/content/drive/MyDrive/Unet/trainNpy/total_mask.npy")
total_mask = np.load("/content/drive/MyDrive/Unet/trainNpy/total_patch.npy")
total_patch.shape

# ==================== 定義 Patch 相關函數 ====================
patch_ratio = []

for i in range(16 + 1):
  patch_ratio.append(32 * i)

print(patch_ratio)

def patch_sampling(img, mask, patch_ratio, pos_neg_ratio, threshold):

  temp_mask = mask

  #temp_mask[temp_mask == 1] = 0
  #temp_mask[temp_mask == 2] = 1

  positive_patch = []
  positive_mask = []

  negative_patch = []
  negative_mask = []

  negative_set = []

  print("Temp_Mask shape",temp_mask.shape)
  for i in range(temp_mask.shape[2]):
    for x_bin in range(2, len(patch_ratio)):

        for y_bin in range(2, len(patch_ratio)):

          img_patch = img[patch_ratio[x_bin-2] : patch_ratio[x_bin], patch_ratio[y_bin - 2] : patch_ratio[y_bin], i]
          mask_patch = temp_mask[patch_ratio[x_bin-2] : patch_ratio[x_bin], patch_ratio[y_bin - 2] : patch_ratio[y_bin], i]
          _, count = np.unique(mask_patch, return_counts = True)

          #Mask圖上有肝臟
          if len(count) == 2:
            mask_percentage = count[1] / sum(count) * 100

            #mask有肝臟的比例要超過門檻，避免欲切出的mask太小
            if threshold < mask_percentage :
              positive_patch.append(img_patch)
              positive_mask.append(mask_patch)

          #Mask圖上沒有肝臟
          elif len(count) ==1:

            temp_list = []
            temp_list.append(img_patch)
            temp_list.append(mask_patch)

            negative_set.append(temp_list)

  shuffle(negative_set)

  #根據pos_neg_ratio要讓負樣本是正樣本的幾倍
  negative_set_to_use = negative_set[:len(positive_patch) * pos_neg_ratio]
  for negative_set in negative_set_to_use:
    negative_patch.append(negative_set[0])
    negative_mask.append(negative_set[1])

  negative_set_to_use = []

  return positive_patch, positive_mask, negative_patch, negative_mask

def slice_to_patch(slice, patch_ratio):
  #slice[slice == 1] = 0
  #slice[slice == 2] = 1

  patch_list = []

  for x_bin in range(2, len(patch_ratio)):
    for y_bin in range(2, len(patch_ratio)):
      patch = slice[patch_ratio[x_bin-2] : patch_ratio[x_bin], patch_ratio[y_bin - 2] : patch_ratio[y_bin]]
      patch = patch.reshape(patch.shape + (1,))
      patch_list.append(patch)

  return np.array(patch_list)

def patch_to_slice(patch, patch_ratio, input_shape, conf_threshold):

  slice = np.zeros((512, 512, 1))
  row_idx = 0
  col_idx = 0

  for i in range(len(patch)):

    slice[patch_ratio[row_idx]:patch_ratio[row_idx + 2], patch_ratio[col_idx]:patch_ratio[col_idx + 2]][patch[i] > conf_threshold] = 1

    col_idx += 1

    if i != 0 and (i+1) % 15 == 0:
      row_idx += 1
      col_idx = 0

  return slice

# ==================== 定義損失函數 ====================
#loss function
@keras.saving.register_keras_serializable()
def weighted_binary_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1.-10e-8)  #把值壓縮到(min,max)之間，小於min的值變成min，大於max的值變成max
    loss = - (y_true * K.log(y_pred) * 0.90 + (1 - y_true) * K.log(1 - y_pred) * 0.10)

    return K.mean(loss)

#dice loss:通常用於計算兩個樣本的相似度
smooth = 1.               #用於防止分母為0
@keras.saving.register_keras_serializable()
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)   #攤平成一維
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)  #計算兩者之間的重疊處
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# ==================== 定義 U-Net 模型 ====================
#Unet架構
def u_net(input_shape, dropout_rate, l2_lambda):

  # Encoder
  input = Input(shape = input_shape, name = "input")
  conv1_1 = Conv2D(32, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv1_1")(input)
  conv1_1 = bn(name = "conv1_1_bn")(conv1_1)
  conv1_2 = Conv2D(32, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv1_2")(conv1_1)
  conv1_2 = bn(name = "conv1_2_bn")(conv1_2)
  pool1 = MaxPooling2D(name = "pool1")(conv1_2)
  drop1 = Dropout(dropout_rate)(pool1)

  conv2_1 = Conv2D(64, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv2_1")(drop1)
  conv2_1 = bn(name = "conv2_1_bn")(conv2_1)
  conv2_2 = Conv2D(64, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv2_2")(conv2_1)
  conv2_2 = bn(name = "conv2_2_bn")(conv2_2)
  pool2 = MaxPooling2D(name = "pool2")(conv2_2)
  drop2 = Dropout(dropout_rate)(pool2)

  conv3_1 = Conv2D(128, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv3_1")(drop2)
  conv3_1 = bn(name = "conv3_1_bn")(conv3_1)
  conv3_2 = Conv2D(128, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv3_2")(conv3_1)
  conv3_2 = bn(name = "conv3_2_bn")(conv3_2)
  pool3 = MaxPooling2D(name = "pool3")(conv3_2)
  drop3 = Dropout(dropout_rate)(pool3)

  conv4_1 = Conv2D(256, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv4_1")(drop3)
  conv4_1 = bn(name = "conv4_1_bn")(conv4_1)
  conv4_2 = Conv2D(256, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv4_2")(conv4_1)
  conv4_2 = bn(name = "conv4_2_bn")(conv4_2)
  pool4 = MaxPooling2D(name = "pool4")(conv4_2)
  drop4 = Dropout(dropout_rate)(pool4)

  conv5_1 = Conv2D(512, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv5_1")(drop4)
  conv5_1 = bn(name = "conv5_1_bn")(conv5_1)
  conv5_2 = Conv2D(512, (3, 3), padding = "same", activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), name = "conv5_2")(conv5_1)
  conv5_2 = bn(name = "conv5_2_bn")(conv5_2)

  # Decoder
  upconv6 = Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same')(conv5_2)
  upconv6 = Dropout(dropout_rate)(upconv6)
  concat6 = concatenate([conv4_2, upconv6], name = "concat6")
  conv6_1 = Conv2D(256, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv6_1")(concat6)
  conv6_1 = bn(name = "conv6_1_bn")(conv6_1)
  conv6_2 = Conv2D(256, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv6_2")(conv6_1)
  conv6_2 = bn(name = "conv6_2_bn")(conv6_2)

  upconv7 = Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same')(conv6_2)
  upconv7 = Dropout(dropout_rate)(upconv7)
  concat7 = concatenate([conv3_2, upconv7], name = "concat7")
  conv7_1 = Conv2D(128, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv7_1")(concat7)
  conv7_1 = bn(name = "conv7_1_bn")(conv7_1)
  conv7_2 = Conv2D(128, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv7_2")(conv7_1)
  conv7_2 = bn(name = "conv7_2_bn")(conv7_2)

  upconv8 = Conv2DTranspose(64,(2, 2), strides=(2, 2), padding='same')(conv7_2)
  upconv8 = Dropout(dropout_rate)(upconv8)
  concat8 = concatenate([conv2_2, upconv8], name = "concat8")
  conv8_1 = Conv2D(64, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv8_1")(concat8)
  conv8_1 = bn(name = "conv8_1_bn")(conv8_1)
  conv8_2 = Conv2D(64, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv8_2")(conv8_1)
  conv8_2 = bn(name = "conv8_2_bn")(conv8_2)

  upconv9 = Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(conv8_2)
  upconv9 = Dropout(dropout_rate)(upconv9)
  concat9 = concatenate([conv1_2, upconv9], name = "concat9")
  conv9_1 = Conv2D(32, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv9_1")(concat9)
  conv9_1 = bn(name = "conv9_1_bn")(conv9_1)
  conv9_2 = Conv2D(32, (3, 3), padding = "same", kernel_regularizer=regularizers.l2(l2_lambda), name = "conv9_2")(conv9_1)
  conv9_2 = bn(name = "conv9_2_bn")(conv9_2)
  dropout = Dropout(dropout_rate)(conv9_2)

  conv10 = Conv2D(1, (1, 1), padding = "same", activation = 'sigmoid', name = "conv10")(dropout)

  model = Model(input, conv10)
  return model

# ==================== 參數設定與建立模型 ====================
#參數設定

input_shape = [64, 64, 1]
dropout_rate = 0.2
l2_lambda = 0.0002

#建立模型
model = u_net(input_shape, dropout_rate, l2_lambda)
model.summary()

# ==================== 訓練模型 ====================
#訓練模型
adam = Adam(learning_rate = 0.0001)
model.compile(optimizer = adam, loss = weighted_binary_crossentropy, metrics = [dice_coef])

model.fit(total_patch, total_mask, batch_size = 128, epochs = 10)

# ==================== 儲存模型 ====================
model_json = model.to_json()
with open("./model_json_final.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("./model_weights_final.h5")
print("Saved model to disk")
