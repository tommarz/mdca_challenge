import pandas as pd
import numpy as np
# import tensorflow as tf
import os
import sys
import sklearn
from sklearn.svm import OneClassSVM


TRAIN = 5000
LABELED_USERS = 10


def load_user_data(dir_path='FraudedRawData'):
    df = pd.DataFrame()
    for file in os.listdir(dir_path):
        df = pd.concat([df, pd.read_csv(dir_path + '/' + file, sep=' ', header=None)], axis=1)
    return df.T.reset_index(drop=True)


if __name__ == '__main__':
    df = load_user_data()
    train_df = df.iloc[:LABELED_USERS, :TRAIN]
    test_df = df.iloc[:LABELED_USERS, TRAIN:]
    submission_df = df.iloc[LABELED_USERS:, TRAIN:]
    print(df.shape)
    print(train_df.shape)
    print(test_df.shape)