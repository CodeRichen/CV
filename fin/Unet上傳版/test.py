# ==================== 載入模型 ====================
json_file = open("/content/drive/MyDrive/Unet/model/model_json_final.json", 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/content/drive/MyDrive/Unet/model/model_weights_final.weights.h5")
print("Loaded model from disk")

# ==================== 載入測試資料 ====================
img_ex = nib.load('/content/drive/MyDrive/Unet/model/volume-3.nii').get_fdata()
mask_ex = nib.load("/content/drive/MyDrive/Unet/model/segmentation-3.nii").get_fdata()

print(img_ex.shape)
print(mask_ex.shape)

# ==================== 預測與評估 ====================
mask_ex[mask_ex == 2] = 1

Dice_coef = 0
samples = 0
for i in range(mask_ex.shape[2]):
  _, count = np.unique(mask_ex[:, :, i], return_counts=True)

  #if len(count) > 1 and count[1] > 300:

  patch_ex = slice_to_patch(img_ex[:, :, i], patch_ratio)
  prediction = loaded_model.predict(patch_ex)

  prediction_mask = patch_to_slice(prediction, patch_ratio, input_shape, conf_threshold = 0.97)

  one_sample_loss = dice_coef(mask_ex[:, :, i], prediction_mask)
  Dice_coef += one_sample_loss
  samples += 1

  print("Slice to Patch Shape:",patch_ex.shape)
  print("Prediction Shape:",prediction_mask.shape)
  print("Dice_coef:",one_sample_loss)

  fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize = ((15, 15)))

  ax1.imshow(np.rot90(img_ex[:, :, i], 3), cmap = 'gray')
  ax1.set_title("Image", fontsize = "x-large")
  ax1.grid(False)
  ax2.imshow(np.rot90(mask_ex[:, :, i], 3), cmap = 'gray')
  ax2.set_title("Mask (True)", fontsize = "x-large")
  ax2.grid(False)
  ax3.imshow(np.rot90(prediction_mask.reshape((512, 512)), 3), cmap = 'gray')
  ax3.set_title("Mask (Pred)", fontsize = "x-large")
  ax3.grid(False)
  plt.show()


print(samples)
print("Dice_coef:", Dice_coef/samples)
