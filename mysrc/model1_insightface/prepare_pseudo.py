import pandas as pd
import numpy as np
import os
from tqdm import *
from shutil import copyfile
from multiprocessing import Pool, cpu_count

def my_process(file_name):
    emb_path = '../../models/insightface/embedding/model-r100-ii/test/%s'%file_name.replace('.png', '_augmentation.npy')
    emb = np.load(emb_path).reshape(100,512)
    return emb

if __name__ == '__main__':
    pseudo_train_df = pd.read_csv('../../datasets/pseudo_train.csv')

    p = Pool(16)
    pseudo_train_data = p.map(func=my_process, iterable = pseudo_train_df.image.values.tolist())
    p.close()
    pseudo_train_data = np.array(pseudo_train_data)
    print(pseudo_train_data.shape)
    np.save('pseudo_train_data.npy', pseudo_train_data)