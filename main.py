import pandas as pd
import numpy as np
# import tensorflow as tf
import os
import sys
import sklearn
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from preprocessing import *

TRAIN = 5000
SEG_SIZE = 100
TRAIN_SEG = round(TRAIN / SEG_SIZE)
LABELED_USERS = 10


def load_user_data(dir_path='FraudedRawData'):
    df = pd.DataFrame()
    for file in os.listdir(dir_path):
        tmp = pd.read_csv(dir_path + '/' + file, sep=' ', header=None)
        df = pd.concat([df, tmp], axis=1)
    df = df.T.reset_index(drop=True)
    seg_df = pd.DataFrame()
    for i in range(150):
        seg_df = pd.concat([seg_df, df.iloc[:, i:i + 100].apply(lambda x: ",".join(x.astype(str)).split(','), axis=1)],
                           axis=1, names=[str(i)])
    seg_df = seg_df.T.reset_index(drop=True).T
    return df, seg_df


def create_svm_model(X_train, y_train, X_test, y_test, report_grade=90):
    svm = OneClassSVM(nu=0.5, gamma='auto')
    svm.fit(X_train)
    preds = [1 if p == -1 else 0 for p in svm.predict(X_test)]
    y_true = y_test.values.flatten().tolist()
    print(f'Acc: {accuracy_score(y_true, preds)}')
    print(f'Precision: {precision_score(y_true, preds)}')
    print(f'Recall: {recall_score(y_true, preds)}')
    print(f'F1 Score: {f1_score(y_true, preds)}')
    conf_mat = confusion_matrix(y_true, preds)
    tp, tn = conf_mat[1, 1], conf_mat[0, 0]
    classification_score = tn + 9*tp
    final_grade = 0.7 * min(100, 95 * 3*classification_score/4575) + 0.3 * report_grade
    print(f'Final Grade: {final_grade}')
    return preds


if __name__ == '__main__':
    df, seg_df = load_user_data()
    y = pd.read_csv('challengeToFill.csv').drop(columns=['Unnamed: 0']).T.reset_index(drop=True).T
    y_train = y.iloc[:LABELED_USERS, :TRAIN_SEG]
    y_test = y.iloc[:LABELED_USERS, TRAIN_SEG:]
    train_df = df.iloc[:LABELED_USERS, :TRAIN]
    seg_train_df = seg_df.iloc[:LABELED_USERS, :TRAIN_SEG]
    test_df = df.iloc[:LABELED_USERS, TRAIN:]
    seg_test_df = seg_df.iloc[:LABELED_USERS, TRAIN_SEG:]
    submission_df = seg_df.iloc[LABELED_USERS:, TRAIN_SEG:]
    sentences = seg_train_df.to_numpy().flatten().tolist()
    embedding_model = create_embeddings(sentences)
    print(embedding_model.wv)
    # vec_df = seg_df.applymap(lambda x: embedding_model.wv[x].mean(axis=0))
    # f = lambda x: embedding_model.wv[x].mean(axis=0)
    # vfunc = np.vectorize(f)
    vec_train_df = np.vstack(
        seg_train_df.applymap(embedding_model.wv.get_sentence_vector).to_numpy().reshape(-1, 1).tolist())
    vec_test_df = np.vstack(
        seg_test_df.applymap(embedding_model.wv.get_sentence_vector).to_numpy().reshape(-1, 1).tolist())
    create_svm_model(X_train=vec_train_df, y_train=y_train, X_test=vec_test_df, y_test=y_test)
    print(vec_train_df.shape)
    print(vec_test_df.shape)
    print(df.shape)
    print(train_df.shape)
    print(test_df.shape)
