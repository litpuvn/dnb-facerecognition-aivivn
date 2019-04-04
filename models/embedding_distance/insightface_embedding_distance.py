import argparse
import cv2
import sys
import numpy as np
import pandas as pd 
from multiprocessing import Pool
from tqdm import *

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--model', default='', help='path to load model.')
args = parser.parse_args()

class Pair:
  def __init__(self, img1_path, img2_path):
    self.img1_path = img1_path
    self.img2_path = img2_path

def my_process(pair):
    emb1 = np.load('../insightface/embedding/%s/test/%s'%(args.model.split(',')[0].split('/')[-2],pair.img1_path.replace('.png', '.npy')))
    emb2 = np.load('../insightface/embedding/%s/train/%s'%(args.model.split(',')[0].split('/')[-2],pair.img2_path.replace('.png', '.npy')))
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

    test_label_predict = []
    test_distance = []
    train_match = []

    for idx, row in tqdm(test_df.iterrows(), total = test_df.shape[0]):
        PairList = []
        for img_path in train_images:
            PairList.append(Pair(row['image'], img_path))
        p = Pool(16)
        result = p.map(func=my_process, iterable = PairList)
        p.close()
        best_idx = np.argmin(result)

        test_label_predict.append(train_labels[best_idx])
        test_distance.append(result[best_idx])
        train_match.append(train_images[best_idx])

    test_df['dist'] = np.array(test_distance)
    test_df['label_predict'] = np.array(test_label_predict)
    test_df['match'] = np.array(train_match)
    test_df.to_csv('%s.csv'%args.model.split(',')[0].split('/')[-2], index = False)
