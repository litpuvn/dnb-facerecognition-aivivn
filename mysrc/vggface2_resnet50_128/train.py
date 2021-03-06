import pandas as pd
import numpy as np

import keras
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import *
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, Activation, BatchNormalization, Dense, Dropout, Flatten, add, Lambda
import tensorflow as tf
from keras import backend as K
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from clr_callback import CyclicLR
import random
import threading
from random import randint
import os

BATCH_SIZE = 32
EPOCHS = 300
NUMBER_OF_FOLDS = 5
NUMBER_OF_PARTS = 4
INPUT_DIM = 128
NUMBER_OF_CLASSES = 1000

def get_y_true(df):
    y_true = []
    for index, row in df.iterrows():
        y_true.append(to_categorical(row['label'], num_classes=NUMBER_OF_CLASSES))
    return np.array(y_true)

def Model():
    model = Sequential()
    model.add(Dense(1024, input_dim=INPUT_DIM, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(NUMBER_OF_CLASSES, init='uniform'))
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    test_df = pd.read_csv('../../datasets/test_refined.csv')
    train_df = pd.read_csv('../../datasets/train_refined.csv')

    xtrain = np.load('train_data.npy')
    xtrain_flip = np.load('train_flip_data.npy')
    ytrain = get_y_true(train_df)
    xtest = np.load('test_data.npy')
    xtest_flip = np.load('test_flip_data.npy')

    if not os.path.exists('weights'):
        os.makedirs('weights')

    ptest = np.zeros((xtest.shape[0], NUMBER_OF_CLASSES), dtype = np.float64)
    training_log = open('training_log.txt', 'w')
    loss_average = 0.0
    acc_average = 0.0
    for part in random.sample(range(30), NUMBER_OF_PARTS):
        for fold in range(NUMBER_OF_FOLDS):
            v_df = train_df.loc[train_df['rt%d'%part] == fold]
            vidxs = v_df.index.values.tolist()
            t_df = train_df.loc[~train_df.index.isin(v_df.index)]
            tidxs = t_df.index.values.tolist()
            print('**************Part %d    Fold %d**************'%(part, fold))

            xtrain_fold = np.vstack((xtrain[tidxs,:],xtrain_flip[tidxs,:]))
            ytrain_fold = np.vstack((ytrain[tidxs,:],ytrain[tidxs,:]))
            xtrain_fold, ytrain_fold = shuffle(xtrain_fold, ytrain_fold)

            xvalid_fold = xtrain[vidxs,:]
            yvalid_fold = ytrain[vidxs,:]

            train_steps = np.ceil(float(2*len(tidxs)) / float(BATCH_SIZE))

            WEIGHTS_BEST = 'weights/best_weight_part%d_fold%d.hdf5'%(part, fold)

            clr = CyclicLR(base_lr=1e-7, max_lr=2e-4, step_size=4*train_steps, mode='exp_range',gamma=0.99994)
            early_stopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='max')
            save_checkpoint = ModelCheckpoint(WEIGHTS_BEST, monitor = 'val_acc', verbose = 1, save_weights_only = True, save_best_only=True, mode='max')
            callbacks = [save_checkpoint, early_stopping, clr]

            model = Model()
            model.summary()
            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=2e-4), metrics=['accuracy'])

            model.fit(xtrain_fold, ytrain_fold, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(xvalid_fold, yvalid_fold), callbacks=callbacks, shuffle=True)

            model.load_weights(WEIGHTS_BEST)

            ptest += model.predict(xtest, batch_size=BATCH_SIZE, verbose=1)
            ptest += model.predict(xtest_flip, batch_size=BATCH_SIZE, verbose=1)

            score = model.evaluate(x=xvalid_fold, y=yvalid_fold, batch_size=BATCH_SIZE, verbose=1)
            loss_average += score[0]
            acc_average += score[1]
            training_log.write('PART:%d FOLD:%d LOSS:%f ACC:%f\n'%(part,fold,score[0], score[1]))

            K.clear_session()

    ptest /= float(2*NUMBER_OF_PARTS*NUMBER_OF_FOLDS)
    np.save('ptest.npy', ptest)

    loss_average /= float(NUMBER_OF_PARTS*NUMBER_OF_FOLDS)
    acc_average /= float(NUMBER_OF_PARTS*NUMBER_OF_FOLDS)
    training_log.write('AVERAGE LOSS:%f ACC:%f\n'%(loss_average, acc_average))
    training_log.close()
