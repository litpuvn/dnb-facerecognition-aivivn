import argparse
import cv2
import sys
import numpy as np
import pandas as pd 
from multiprocessing import Pool
from tqdm import *

class Pair:
  def __init__(self, img1_path, img2_path, pretrain_model):
    self.img1_path = img1_path
    self.img2_path = img2_path
    self.pretrain_model = pretrain_model

def my_process(pair):
    emb1 = np.load('../vgg_face2/%s/embedding/test/%s'%(pair.pretrain_model, pair.img1_path.replace('.png', '.npy')))
    emb2 = np.load('../vgg_face2/%s/embedding/train/%s'%(pair.pretrain_model, pair.img2_path.replace('.png', '.npy')))
    dist = np.sum(np.square(emb1-emb2))
    return dist

if __name__ == '__main__':
    test_df = pd.read_csv('../../datasets/test_refined.csv')
    train_df = pd.read_csv('../../datasets/train.csv')
    train_images = []
    train_labels = []
    for idx,row in train_df.iterrows():
        train_images.append(row['image'])
        train_labels.append(row['label'])

    for pretrain_model in ['resnet50_128_pytorch','resnet50_256_pytorch','resnet50_ft_pytorch','resnet50_scratch_pytorch','senet50_128_pytorch','senet50_256_pytorch','senet50_ft_pytorch','senet50_scratch_pytorch']:
        print(pretrain_model)

        test_label_predict = []
        test_distance = []
        train_match = []

        for idx, row in tqdm(test_df.iterrows(), total = test_df.shape[0]):
            PairList = []
            for img_path in train_images:
                PairList.append(Pair(row['image'], img_path, pretrain_model))
            p = Pool(16)
            result = p.map(func=my_process, iterable = PairList)
            p.close()
            best_idx = np.argmin(result)

            test_label_predict.append(train_labels[best_idx])
            test_distance.append(result[best_idx])
            train_match.append(train_images[best_idx])

        new_df = pd.DataFrame()
        new_df['image'] = test_df['image']
        new_df['dist'] = np.array(test_distance)
        new_df['label_predict'] = np.array(test_label_predict)
        new_df['match'] = np.array(train_match)
        new_df.to_csv('%s.csv'%pretrain_model, index = False)
