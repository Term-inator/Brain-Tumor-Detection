# ====================================================
# main
# ====================================================
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from utils import get_score, seed_torch
from train import train_loop, set_params
from logger import init_logger, Logger


class Params:
    n_fold = 4
    trn_fold = [0, 1, 2]
    output_dir = '../output/'
    data_path = '../input/'

    debug = False
    train = True

    def __init__(self, type, seed, epochs):
        Params.type = type
        if type == 'tumor' or type == 'T1SS' or type == 'T2SS':
            Params.target_cols = ['label']
        elif type == 'T1':
            Params.target_cols = ['T1']
        elif type == '2label' or type == 'randT1' or type == 'randTumor':
            Params.target_cols = ['label', 'T1']

        if type == 'tumor' or type == 'T1' or type == '2label':
            Params.data_path += 'RealTrain/'
        elif type == 'randT1':
            Params.data_path += 'RealTrainRandomT1/'
        elif type == 'randTumor':
            Params.data_path += 'RealTrainRandomTumor/'
        elif type == 'T1SS':
            Params.data_path += 'T1TrainSameSize/'
        elif type == 'T2SS':
            Params.data_path += 'T2TrainSameSize/'
        Params.target_size = len(Params.target_cols)
        Params.seed = seed
        Params.epochs = epochs
        Params.output_dir += f'{type}_seed{seed}-ep{epochs}/'
        # ====================================================
        # Directory settings
        # ====================================================
        if os.path.exists(Params.output_dir):
            shutil.rmtree(Params.output_dir)
        os.makedirs(Params.output_dir)

        print(Params.target_cols, Params.target_size)
        print(Params.data_path)


if Params.debug:
    Params.epochs = 1


def main():
    train = pd.read_csv(Params.data_path + 'data.csv')
    # print(train.head())
    init_logger(Params.output_dir + 'train.log')

    seed_torch(seed=Params.seed)

    # ====================================================
    # Split Train Test
    # ====================================================
    folds = train.copy()
    if Params.type != 'T1SS' and Params.type != 'T2SS':
        Fold = GroupKFold(n_splits=Params.n_fold)
        groups = folds['filename'].values
        for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[Params.target_cols], groups)):
            folds.loc[val_index, 'fold'] = int(n)
        folds['fold'] = folds['fold'].astype(int)
        # print(folds.groupby('fold').size())
        # print(folds)

    tst_idx = folds[folds['fold'] == Params.n_fold - 1].index

    test_fold = folds.loc[tst_idx].reset_index(drop=True)
    _test_fold = test_fold.copy(deep=True)

    train_folds = folds[folds['fold'].isin([i for i in range(Params.n_fold - 1)])]

    # print(train_folds.groupby('fold').size())
    # print(train_folds)

    def get_test_result(test_scores):
        Logger().info(f'Scores: {np.round(np.mean(test_scores, axis=0), decimals=4)}')

    def get_result(result_df):
        preds = result_df[[f'pred_{c}' for c in Params.target_cols]].values
        labels = result_df[Params.target_cols].values
        score, scores = get_score(labels, preds)
        Logger().info(f'Score: {score:<.4f}  Scores: {np.round(scores, decimals=4)}')

    set_params(Params)
    all_test_scores = []

    if Params.train:
        # train
        oof_df = pd.DataFrame()
        for fold in range(Params.n_fold - 1):
            if fold in Params.trn_fold:
                _oof_df, test_scores = train_loop(train_folds, fold, _test_fold)
                oof_df = pd.concat([oof_df, _oof_df])
                all_test_scores.append(test_scores)
                Logger().info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # test result
        Logger().info(f"\n========== TEST ==========")
        get_test_result(np.array(all_test_scores))
        # CV result
        Logger().info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(Params.output_dir + 'result.csv', index=False)


seed_list = [37, 41, 42, 43, 47]

# tumor T1SS T2SS T1 2label randT1 randTumor
if __name__ == '__main__':
    for seed in seed_list:
        for epochs in range(10, 61, 10):
            Params('tumor', seed, epochs)
            main()

