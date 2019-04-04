import pandas as pd
import numpy as np
import os
from tqdm import *
from shutil import copyfile
from multiprocessing import Pool, cpu_count

def my_process1(file_name):
    emb_path = '../../models/vgg_face2/resnet50_scratch_pytorch/embedding/test/%s'%file_name.replace('.png', '.npy')
    emb = np.load(emb_path)
    return emb

def my_process2(file_name):
    emb_path = '../../models/vgg_face2/resnet50_scratch_pytorch/embedding/test/%s'%file_name.replace('.png', '_flip.npy')
    emb = np.load(emb_path)
    return emb

if __name__ == '__main__':
    pseudo_train_df = pd.read_csv('../../datasets/pseudo_train.csv')

    p = Pool(16)
    pseudo_train_data = p.map(func=my_process1, iterable = pseudo_train_df.image.values.tolist())
    p.close()
    pseudo_train_data = np.array(pseudo_train_data)
    print(pseudo_train_data.shape)
    np.save('pseudo_train_data.npy', pseudo_train_data)
    pseudo_train_data = []

    p = Pool(16)
    pseudo_train_flip_data = p.map(func=my_process2, iterable = pseudo_train_df.image.values.tolist())
    p.close()
    pseudo_train_flip_data = np.array(pseudo_train_flip_data)
    print(pseudo_train_flip_data.shape)
    np.save('pseudo_train_flip_data.npy', pseudo_train_flip_data)
    pseudo_train_flip_data = []