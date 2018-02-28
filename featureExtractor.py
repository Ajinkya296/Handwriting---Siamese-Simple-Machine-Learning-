import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

sift = cv2.xfeatures2d.SIFT_create()
train_path = 'Images/'
training_names = os.listdir(train_path)

image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    image_paths.append(os.path.join(train_path, training_name))
    #print(training_name);
#print(image_paths)

des_list = []
for image_path in image_paths:
    im = cv2.imread(image_path,0)
    kpts, des = sift.detectAndCompute(im,None)
    des_list.append((image_path, des))

descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

print(len(descriptors))
k = 20
voc, variance = kmeans(descriptors, k, 1)
print("kmeans done")

im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

print(im_features)

feat = np.array(im_features)
np.savetxt('features.csv',im_features, delimiter=',')

'''
plt.imshow(img2),plt.show()
'''
