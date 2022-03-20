# ====================================================
# main
# ====================================================
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from utils import get_score, LOGGER, seed_torch
from train import train_loop, set_params


class Params:
    n_fold = 4
    trn_fold = [0, 1, 2]

    debug = False
    train = True

    # target_cols=['label', 'T1']
    target_cols = ['label']
    # target_cols=['T1']
    target_size = len(target_cols)
    output_dir = './'
    data_path = '../input/RealTrain/'
    seed = 42
    epochs = 0


if Params.debug:
    Params.epochs = 1

# ====================================================
# Directory settings
# ====================================================
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


train = pd.read_csv(Params.data_path + 'data.csv')
# print(train.head())

seed_torch(seed=Params.seed)


# ====================================================
# Split Train Test
# ====================================================
folds = train.copy()
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


def main():
    """
    Prepare: 1.train  2.folds
    """

    def get_test_result(test_scores):
        LOGGER.info(f'Scores: {np.mean(test_scores):<.4f}')

    def get_result(result_df):
        preds = result_df[[f'pred_{c}' for c in Params.target_cols]].values
        labels = result_df[Params.target_cols].values
        score, scores = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}  Scores: {np.round(scores, decimals=4)}')

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
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # test result
        LOGGER.info(f"\n========== TEST ==========")
        get_test_result(np.array(all_test_scores))
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR + 'result.csv', index=False)


if __name__ == '__main__':
    main()
