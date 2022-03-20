# ====================================================
# main
# ====================================================
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from utils import get_score, LOGGER, seed_torch
from train import train_loop


# ====================================================
# Directory settings
# ====================================================
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_PATH = '../input/RealTrain/'

class CFG:
    debug = False
    print_freq = 10
    num_workers = 0
    model_name = 'resnext50_32x4d'
    size = 200
    scheduler = 'CosineAnnealingLR'  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs = 1
    # factor=0.2 # ReduceLROnPlateau
    # patience=4 # ReduceLROnPlateau
    # eps=1e-6 # ReduceLROnPlateau
    T_max = 6  # CosineAnnealingLR
    # T_0=6 # CosineAnnealingWarmRestarts
    lr = 1e-4
    min_lr = 1e-6
    # batch_size=32
    batch_size = 2
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 43
    target_size = 1
    # target_cols=['label', 'T1']
    target_cols = ['label']
    # target_cols=['T1']
    n_fold = 4
    trn_fold = [0, 1, 2]
    train = True


if CFG.debug:
    CFG.epochs = 1




train = pd.read_csv(TRAIN_PATH + 'data.csv')
# print(train.head())

seed_torch(seed=CFG.seed)

folds = train.copy()
Fold = GroupKFold(n_splits=CFG.n_fold)
groups = folds['filename'].values
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_cols], groups)):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
# print(folds.groupby('fold').size())
# print(folds)

# ====================================================
# Test
# ====================================================
tst_idx = folds[folds['fold'] == CFG.n_fold - 1].index

test_fold = folds.loc[tst_idx].reset_index(drop=True)
_test_fold = test_fold.copy(deep=True)

train_folds = folds[folds['fold'].isin([i for i in range(CFG.n_fold - 1)])]
# print(train_folds.groupby('fold').size())
# print(train_folds)


def main():
    """
    Prepare: 1.train  2.folds
    """

    def get_result(result_df):
        preds = result_df[[f'pred_{c}' for c in CFG.target_cols]].values
        labels = result_df[CFG.target_cols].values
        score, scores = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}  Scores: {np.round(scores, decimals=4)}')

    if CFG.train:
        # train
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold - 1):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train_folds, fold, _test_fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                # display(_oof_df)
                get_result(_oof_df)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR + 'oof_df.csv', index=False)


if __name__ == '__main__':
    main()
