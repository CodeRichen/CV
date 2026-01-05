import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from PIL import Image
import cv2

from google.colab import drive
drive.mount('/content/drive')

image_size = (224,224)
path='/content/drive/MyDrive/CNN/train/'
testpath='/content/drive/MyDrive/CNN/test/'

label_dir = [x for x in os.listdir(path) if os.path.isdir(path+x)]
label = []

def read_image(testpath):
    test = []
    for img_name in os.listdir(testpath):
        img2 = np.zeros((224,224,3))
        label.append(img_name.split('_')[0])
        img_path = os.path.join(testpath,img_name)
        img = cv2.imread(img_path)
        img = np.array(img)
        img = cv2.resize(img,image_size)
        test.append(img)

    return test

if __name__ == '__main__':
    test_data  = np.array(read_image(testpath))

    #print("Test Data Shape:",test_data.shape)

    Accuracy = []
    for cur_model_number in range(1):
        Acc = 0

        model = keras.models.load_model(
            '/content/drive/My Drive/CNN/model.keras',)

        pred = model.predict(test_data)
        #print("Prediction Shape:",pred.shape)

        for index in range(len(pred)):
            #print("index:",index)
            pred_Index = np.argmax(pred[index])
            print("預測類別:",label_dir[pred_Index])
            #print("label_dir:",len(label_dir))
            print("答案:",label[index]+"\n")
        # 如果答案正確正確率就+1
            if label_dir[pred_Index]==label[index]:
                Acc += 1

        Accuracy.append(Acc)
    print("Average:",np.sum(np.array(Accuracy))/(len(pred)*1))
