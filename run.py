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
from logger import init_logger, close_logger, Logger


target_cols_map = {
    'tumor': ['label'],
    'T1SS': ['label'],
    'T2SS': ['label'],
    'T1': ['T1'],
    '2label': ['label', 'T1'],
    'randT1': ['label', 'T1'],
    'randTumor': ['label', 'T1']
}

data_path_map = {
    'tumor': 'RealTrain/',
    'T1SS': 'T1TrainSameSize/',
    'T2SS': 'T2TrainSameSize/',
    'T1': 'RealTrain/',
    '2label': 'RealTrain/',
    'randT1': 'RealTrainRandomT1/',
    'randTumor': 'RealTrainRandomTumor/'
}


class Params:
    n_fold = 4
    trn_fold = [0, 1, 2]

    debug = False
    train = True

    type = None
    target_cols = None
    data_path = None
    output_dir = None
    seed = None

    def __init__(self, type, seed, epochs):
        Params.type = type
        output_base_path = '../output/'
        data_base_path = '../input/'

        Params.target_cols = target_cols_map[type]
        Params.data_path = data_base_path + data_path_map[type]

        Params.target_size = len(Params.target_cols)
        Params.seed = seed
        Params.epochs = epochs
        Params.output_dir = output_base_path + f'{type}_seed{seed}-ep{epochs}/'
        # ====================================================
        # Directory settings
        # ====================================================
        if os.path.exists(Params.output_dir):
            shutil.rmtree(Params.output_dir)
        os.makedirs(Params.output_dir)


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

    close_logger()


seed_list = [31, 37, 41, 42, 43, 47, 53]
seeds = [53]
type_list = ['tumor', 'T1SS', 'T2SS', 'T1', '2label', 'randT1', 'randTumor']
types = ['T2SS']

if __name__ == '__main__':
    for seed in seeds:
        for type in types:
            for epochs in range(10, 61, 10):
                Params(type, seed, epochs)
                print(f'target_cols: {Params.target_cols}')
                print(f'data_path: {Params.data_path}, output_dir: {Params.output_dir}')
                print(f'seed: {seed}, epochs: {epochs}')
                main()

