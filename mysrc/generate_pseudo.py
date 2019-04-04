import pandas as pd
import numpy as np
from tqdm import *
from random import randint

NUMBER_OF_CLASSES = 1000

insightface_dict = {
    'model1_insightface':           0.75,
    'model2_insightface':           0.1,
    'model3_insightface':           0.1,
    'model4_insightface':           0.05
}
vggface2_dict = {
    'vggface2_resnet50_128':        0.125,
    'vggface2_resnet50_256':        0.125,
    'vggface2_resnet50_ft':         0.125,
    'vggface2_resnet50_scratch':    0.125,
    'vggface2_senet50_128':         0.125,
    'vggface2_senet50_256':         0.125,
    'vggface2_senet50_ft':          0.125,
    'vggface2_senet50_scratch':     0.125
}
facenet_dict = {
    'model6_facenet':               0.5,
    'model7_facenet':               0.5
}

if __name__ == '__main__':
    test_df = pd.read_csv('../datasets/test_refined.csv')
    test_size = test_df.shape[0]

    insightface_p_test = np.zeros((test_size, NUMBER_OF_CLASSES), dtype = np.float64)
    for model, weight in insightface_dict.items():
        insightface_p_test += weight*np.load('%s/ptest.npy'%model)

    vggface2_p_test = np.zeros((test_size, NUMBER_OF_CLASSES), dtype = np.float64)
    for model, weight in vggface2_dict.items():
        vggface2_p_test += weight*np.load('%s/ptest.npy'%model)

    facenet_p_test = np.zeros((test_size, NUMBER_OF_CLASSES), dtype = np.float64)
    for model, weight in facenet_dict.items():
        facenet_p_test += weight*np.load('%s/ptest.npy'%model)
    
    p_test = 0.7*insightface_p_test + 0.2*vggface2_p_test + 0.1*facenet_p_test

    best_idx = np.argmax(p_test, axis=1)
    probs = []
    for i, v in enumerate(best_idx):
        probs.append(p_test[i,v])

    test_df['label'] = np.array(best_idx)
    test_df['prob'] = np.array(probs)

    test_df = test_df.sort_values(by=['prob'], ascending=False)
    test_df.to_csv('../datasets/pseudo_test.csv', index = False)