import os
import numpy as np
from PIL import Image, ImageOps
# Meow Woof
CAT = 0
DOG = 1

def get_next_batch(batch_size=128,image_size = 64):
    path = '../../ML Testers/catvsdog/train/train'
    allFiles = os.listdir(path)
    imgFiles = [i for i in allFiles if i.endswith('.jpg')]
    idx = np.random.permutation(len(imgFiles))
    idx = idx[0:batch_size]
    images = []
    labels = []
    for item in idx:
        images.append(np.array(ImageOps.fit(Image.open(path+'/'+imgFiles[item]),(image_size,image_size)))/255)
        if imgFiles[item].split('.')[0] == 'dog':
            labels.append(DOG)
        else:
            labels.append(CAT)
    images = np.reshape(images,[-1,image_size,image_size,3])
    labels = np.array(labels)
    labels = np.reshape(labels,[-1,1])
    return [images.astype(np.float32),labels]


#get_next_batch(batch_size=128,image_size = 64)