import argparse
import cv2
import sys
import numpy as np
import pandas as pd 
from multiprocessing import Pool
from tqdm import *

if __name__ == '__main__':
    insightface_dict = {
        'model-r100-ii.csv':               0.75,
        'model-r34-amf.csv':               0.1,
        'model-r50-am-lfw.csv':            0.1,
        'model-y1-test2.csv':              0.05
    }

    vggface2_dict = {
        'resnet50_128_pytorch.csv':        0.125,
        'resnet50_256_pytorch.csv':        0.125,
        'resnet50_ft_pytorch.csv':         0.125,
        'resnet50_scratch_pytorch.csv':    0.125,
        'senet50_128_pytorch.csv':         0.125,
        'senet50_256_pytorch.csv':         0.125,
        'senet50_ft_pytorch.csv':          0.125,
        'senet50_scratch_pytorch.csv':     0.125
    }

    facenet_dict = {
        'facenet_20180402-114759.csv':     0.5,
        'facenet_20180408-102900.csv':     0.5
    }

    test_df = pd.read_csv('../../datasets/test_refined.csv')
    train_df = pd.read_csv('../../datasets/train.csv')
    pseudo_test_df = pd.read_csv('../../datasets/pseudo_test.csv')

    insightface_dist = np.zeros(test_df.shape[0])
    for model, weight in insightface_dict.items():
        insightface_dist += weight*np.array(pd.read_csv(model).dist.values)

    # vggface2_dist = np.zeros(test_df.shape[0])
    # for model, weight in vggface2_dict.items():
    #     vggface2_dist += weight*np.array(pd.read_csv(model).dist.values)

    # facenet_dist = np.zeros(test_df.shape[0])
    # for model, weight in facenet_dict.items():
    #     facenet_dist += weight*np.array(pd.read_csv(model).dist.values)

    dist = insightface_dist
    test_df['dist'] = np.array(dist)

    dist_arr = []
    for idx,row in tqdm(pseudo_test_df.iterrows(),total = pseudo_test_df.shape[0]):
        tmp_df = test_df.loc[test_df['image'] == row['image']]
        assert tmp_df.shape[0] == 1
        dist_arr.append(tmp_df.dist.values[0])
    pseudo_test_df['dist'] = np.array(dist_arr)
    pseudo_test_df = pseudo_test_df.sort_values(by=['dist'], ascending=False)
    pseudo_test_df.to_csv('unknown.csv', index = False)

    pseudo_train_df = pseudo_test_df[(pseudo_test_df['dist'] < 0.9) & (pseudo_test_df['prob'] > 0.65)]
    print(pseudo_train_df.shape)

    unknown_df = pseudo_test_df[(pseudo_test_df['dist'] > 1.35) & (pseudo_test_df['prob'] < 0.1)]
    unknown_df = unknown_df.sort_values(by=['dist'], ascending=False)
    print('NO. Unknown images: %d'%unknown_df.shape[0])

    with open('../../datasets/unknown/test.txt') as f:
        noface = f.readlines()
    noface = [x.strip() for x in noface]
    print('No Image without face: %d'%len(noface))

    unknown_images = unknown_df.image.values.tolist()
    for x in noface:
        if x not in unknown_images:
            print(x)
            unknown_df = unknown_df.append({'image': x}, ignore_index=True)
    print('NO. Unknown images after add non face image: %d'%unknown_df.shape[0])

    unknown_df.to_csv('../../datasets/unknown_predicted.csv', index = False)

    images = []
    labels = []
    for img in test_df.image.values.tolist():
        images.append(img)
        if img in unknown_images or img in noface:
            labels.append('1000 -1 -1 -1 -1')
        else:
            labels.append('-1 -1 -1 -1 -1')
    sub = pd.DataFrame()
    sub['image'] = np.array(images)
    sub['label'] = np.array(labels)
    sub.to_csv('submission_test_unknown.csv', index=False)

    unknown_df = unknown_df.assign(label=1000)
    pseudo_train_df = pd.concat([pseudo_train_df, unknown_df], ignore_index=True)
    pseudo_train_df = pseudo_train_df.sample(frac = 1)
    pseudo_train_df.to_csv('../../datasets/pseudo_train.csv', index = False, columns = ['image', 'label'])