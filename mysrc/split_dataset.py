import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from random import randint

NUMBER_OF_FOLDS = 5
NUMBER_OF_PARTS = 30

if __name__ == '__main__':
    test_df = pd.read_csv('../datasets/sample_submission.csv', usecols=['image'])
    test_df.to_csv('../datasets/test_refined.csv', index = False)
    print(test_df.head(5))

    train_df = pd.read_csv('../datasets/train.csv')
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    indexs = list(range(train_df.shape[0]))
    for i in range(NUMBER_OF_PARTS):
        info = {}
        rt = randint(1, 99)
        kf = KFold(n_splits=NUMBER_OF_FOLDS, random_state= rt, shuffle=True)
        for fold, (_, valid_index) in enumerate(kf.split(indexs)):
            for vi in valid_index:
                info[vi] = fold
        myarr = []
        for idx in range(train_df.shape[0]):
            myarr.append(info[idx])
        train_df['rt%d'%i] = np.array(myarr)
    train_df.to_csv('../datasets/train_refined.csv', index = False)
    print(train_df.head(5))