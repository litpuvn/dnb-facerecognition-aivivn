import sys
import dlib
import cv2
import numpy as np
import os
from tqdm import *
import imgaug as ia
from imgaug import augmenters as iaa

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if len(sys.argv) != 3:
    print(
        "Call this program like this:\n"
        "   ./face_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat\n"
        "You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
    exit()

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

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
    output_dir = '../models/dlib-19.17/embedding/%s'%mset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for rdir, sdir, files in os.walk('../datasets/aligned/%s/150x150/'%mset):
        for file in tqdm(files):
            if '.png' not in file:
                continue
            fn, fe = os.path.splitext(file)
            img_path = os.path.join(rdir, file)
            img_org = cv2.imread(img_path)
            img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
            
            emb = facerec.compute_face_descriptor(img_org)
            np.save(output_dir + '/%s.npy'%fn, np.array(emb))

            if mset == 'test':
                flip_img = cv2.flip(img_org, 1)
                emb = facerec.compute_face_descriptor(flip_img)
                np.save(output_dir + '/%s_flip.npy'%fn, np.array(emb))

            augmentation_arr = np.array([],dtype=np.float32).reshape(0,128)
            for i in range(100):
                img_aug = seq.augment_image(img_org)
                emb = np.array(facerec.compute_face_descriptor(img_aug))
                augmentation_arr = np.vstack((augmentation_arr, emb.reshape(1,128)))
            np.save(output_dir + '/%s_augmentation.npy'%fn, augmentation_arr)