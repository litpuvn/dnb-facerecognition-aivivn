import sys
import cv2
import numpy as np
import os
from tqdm import *
import imp
import torch
import PIL
import torchvision
from PIL import Image
import pandas as pd
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
from multiprocessing import Pool

def rotate_channels(img):
    return PIL.Image.merge("RGB", (list(img.split()))[::-1])

def my_process_train(image_file):
    img_path = '../datasets/aligned/train/224x224/%s'%image_file
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(img)
    img = to_tensor(rotate_channels(img))*255 - torch.Tensor([91.4953, 103.8827, 131.0912]).view((3,1,1))
    return img.numpy()

def my_process_test(image_file):
    img_path = '../datasets/aligned/test/224x224/%s'%image_file
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(img)
    img = to_tensor(rotate_channels(img))*255 - torch.Tensor([91.4953, 103.8827, 131.0912]).view((3,1,1))
    return img.numpy()

def my_process_train_flip(image_file):
    img_path = '../datasets/aligned/train/224x224/%s'%image_file
    img = cv2.imread(img_path)
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(img)
    img = to_tensor(rotate_channels(img))*255 - torch.Tensor([91.4953, 103.8827, 131.0912]).view((3,1,1))
    return img.numpy()

def my_process_test_flip(image_file):
    img_path = '../datasets/aligned/test/224x224/%s'%image_file
    img = cv2.imread(img_path)
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(img)
    img = to_tensor(rotate_channels(img))*255 - torch.Tensor([91.4953, 103.8827, 131.0912]).view((3,1,1))
    return img.numpy()

to_tensor = torchvision.transforms.ToTensor()

if __name__ == "__main__":
    batch_size = 1024
    for pretrain_model in ['resnet50_128_pytorch','resnet50_256_pytorch','resnet50_ft_pytorch','resnet50_scratch_pytorch','senet50_128_pytorch','senet50_256_pytorch','senet50_ft_pytorch','senet50_scratch_pytorch']:
        print(pretrain_model)
        MainModel = imp.load_source('MainModel', '../models/vgg_face2/%s/%s.py'%(pretrain_model,pretrain_model))
        model = torch.load('../models/vgg_face2/%s/%s.pth'%(pretrain_model,pretrain_model))
        model = model.cuda()
        for mset in ['train','test']:
            output_dir = '../models/vgg_face2/%s/embedding/%s'%(pretrain_model,mset)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            df = pd.read_csv('../datasets/%s_refined.csv'%mset)
            images = df.image.values.tolist()
            
            for start in tqdm(range(0, len(images), batch_size)):
                end = min(start + batch_size, len(images))
                batch = images[start:end]
                p = Pool(8)
                if mset == 'train':
                    imgs = np.array(p.map(func=my_process_train, iterable = batch))
                else:
                    imgs = np.array(p.map(func=my_process_test, iterable = batch))
                p.close()

                for idx,img in enumerate(imgs):
                    img = torch.from_numpy(np.expand_dims(img, 0)).cuda()
                    features = np.squeeze(model(img).data.cpu().numpy())
                    embed = preprocessing.normalize(np.expand_dims(features,0)).flatten()
                    np.save(output_dir + '/' + batch[idx].replace('.png','.npy'), embed)

                p = Pool(8)
                if mset == 'train':
                    imgs = np.array(p.map(func=my_process_train_flip, iterable = batch))
                else:
                    imgs = np.array(p.map(func=my_process_test_flip, iterable = batch))
                p.close()

                for idx,img in enumerate(imgs):
                    img = torch.from_numpy(np.expand_dims(img, 0)).cuda()
                    features = np.squeeze(model(img).data.cpu().numpy())
                    embed = preprocessing.normalize(np.expand_dims(features,0)).flatten()
                    np.save(output_dir + '/' + batch[idx].replace('.png','_flip.npy'), embed)
