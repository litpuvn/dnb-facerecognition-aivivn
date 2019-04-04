import face_model
import argparse
import cv2
import sys
import numpy as np
import os
from tqdm import *
import imgaug as ia
from imgaug import augmenters as iaa

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

print(args.model)
model = face_model.FaceModel(args)

sometimes = lambda aug: iaa.Sometimes(0.8, aug)
seq = iaa.Sequential([
	iaa.Fliplr(0.5),
	sometimes(
		iaa.OneOf([
			iaa.Grayscale(alpha=(0.0, 1.0)),
			iaa.AddToHueAndSaturation((-20, 20)),
			iaa.Add((-20, 20), per_channel=0.5),
			iaa.Multiply((0.5, 1.5), per_channel=0.5),
			iaa.GaussianBlur((0, 2.0)),
			iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
			iaa.Sharpen(alpha=(0, 0.5), lightness=(0.7, 1.3)),
			iaa.Emboss(alpha=(0, 0.5), strength=(0, 1.5))
		])
	)
])

for mset in ['train', 'test']:
    output_dir = '../models/insightface/embedding/%s/%s'%(args.model.split(',')[0].split('/')[-2],mset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for rdir, sdir, files in os.walk('../datasets/aligned/%s/112x112/'%mset):
        for file in tqdm(files):
            if '.png' not in file:
                continue
            fn, fe = os.path.splitext(file)
            img_path = os.path.join(rdir, file)
            img_org = cv2.imread(img_path)
            img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

            img = np.transpose(img_org, (2,0,1))
            emb = model.get_feature(img)
            np.save(output_dir + '/%s.npy'%fn, emb)

            if mset == 'test':
                flip_img = cv2.flip(img_org, 1)
                flip_img = np.transpose(flip_img, (2,0,1))
                emb = model.get_feature(flip_img)
                np.save(output_dir + '/%s_flip.npy'%fn, emb)

            if 'model-y1-test2' == args.model.split(',')[0].split('/')[-2]:
                augmentation_arr = np.array([],dtype=np.float32).reshape(0,128)
                for i in range(100):
                    img_aug = seq.augment_image(img_org)
                    img_aug = np.transpose(img_aug, (2,0,1))
                    emb = model.get_feature(img_aug)
                    augmentation_arr = np.vstack((augmentation_arr, emb.reshape(1,128)))
                np.save(output_dir + '/%s_augmentation.npy'%fn, augmentation_arr)
            else:
                augmentation_arr = np.array([],dtype=np.float32).reshape(0,512)
                for i in range(100):
                    img_aug = seq.augment_image(img_org)
                    img_aug = np.transpose(img_aug, (2,0,1))
                    emb = model.get_feature(img_aug)
                    augmentation_arr = np.vstack((augmentation_arr, emb.reshape(1,512)))
                np.save(output_dir + '/%s_augmentation.npy'%fn, augmentation_arr)
