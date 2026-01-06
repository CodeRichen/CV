

import cv2
import numpy as np

#Generate Data
# Data A
a = np.random.randint(95,100, (20, 2)).astype(np.float32)
#Data B
b = np.random.randint(90,95, (20, 2)).astype(np.float32)
#Merge A and B
data = np.vstack((a,b))
data = np.array(data,dtype='float32')
print('------------training data---------------')
print(data)



#Set class labels
#aLabel for A
aLabel=np.zeros((20,1))
#bLabel for B
bLabel=np.ones((20,1))
#Merge labels
label = np.vstack((aLabel, bLabel))
label = np.array(label,dtype='int32')
print('-----------training Label----------------')
print(label)


kernels = [cv2.ml.SVM_LINEAR, cv2.ml.SVM_INTER, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_RBF]
kernels1 = ['LINEAR','INTER','SIGMOID','RBF']


# create SVM model 
svm = cv2.ml.SVM_create() 
svm.setKernel(kernels[0]) 
# Training
result = svm.train(data,cv2.ml.ROW_SAMPLE,label)



#Generate Testing Data
test = np.vstack([[99,95],[90,90],[98,98]]) #0-Class A, 1-Class B
test = np.array(test,dtype='float32')

print('-----------testing data----------------')
print(test)
print('-----------testing result----------------')

#Perform testing
(p1,p2) = svm.predict(test)


print(p2)

