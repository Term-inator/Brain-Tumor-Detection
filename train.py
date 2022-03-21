import time
import warnings

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from albumentations import Compose, Normalize, Resize, RandomResizedCrop, HorizontalFlip
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from utils import get_score, AverageMeter, timeSince
from logger import Logger

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Params:
    model_name = 'resnext50_32x4d'
    scheduler = 'CosineAnnealingLR'  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    # factor=0.2 # ReduceLROnPlateau
    # patience=4 # ReduceLROnPlateau
    # eps=1e-6 # ReduceLROnPlateau
    T_max = 6  # CosineAnnealingLR
    # T_0=6 # CosineAnnealingWarmRestarts
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000

    size = 200
    batch_size = 2
    print_freq = 10
    num_workers = 0

    # target_cols=['label', 'T1']
    target_cols = ['label']
    # target_cols=['T1']
    target_size = len(target_cols)
    output_dir = './'
    data_path = '../input/RealTrain/'
    seed = 42
    epochs = 1


def set_params(params):
    Params.target_cols = params.target_cols
    Params.target_size = params.target_size
    Params.output_dir = params.output_dir
    Params.data_path = params.data_path
    Params.seed = params.seed
    Params.epochs = params.epochs


# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['filename'].values
        self.labels = df[Params.target_cols].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        label = torch.tensor(self.labels[idx]).float()
        image = cv2.imread(Params.data_path + 'train/' + file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label


# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):
    if data == 'train':
        return Compose([
            Resize(Params.size, Params.size),
            # RandomResizedCrop(Params.size, Params.size, scale=(0.85, 1.0)),
            HorizontalFlip(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(Params.size, Params.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


# ====================================================
# MODEL
# ====================================================
class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        if pretrained:
            self.model.load_state_dict(torch.load('../models/resnext50_32x4d_a1h-0146ab0a.pth'))
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, Params.target_size)

    def forward(self, x):
        x = self.model(x)
        return x


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    scaler = GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with autocast():
            y_preds = model(images)
            loss = criterion(y_preds, labels)
        # record loss
        losses.update(loss.item(), batch_size)
        if Params.gradient_accumulation_steps > 1:
            loss = loss / Params.gradient_accumulation_steps
        scaler.scale(loss).backward()
        # loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Params.max_grad_norm)
        if (step + 1) % Params.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % Params.print_freq == 0 or step == (len(train_loader) - 1):
            Logger().info('Epoch: [{0}][{1}/{2}] '
                          'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                          'Elapsed {remain:s} '
                          'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                          'Grad: {grad_norm:.4f}  '
                          'LR: {lr:.6f}  '.format(
                                epoch + 1, step, len(train_loader), batch_time=batch_time,
                                data_time=data_time, loss=losses,
                                remain=timeSince(start, float(step + 1) / len(train_loader)),
                                grad_norm=grad_norm,
                                lr=scheduler.get_lr()[0]
                            ))
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.sigmoid().to('cpu').numpy())
        if Params.gradient_accumulation_steps > 1:
            loss = loss / Params.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % Params.print_freq == 0 or step == (len(valid_loader) - 1):
            Logger().info('EVAL: [{0}/{1}] '
                          'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                          'Elapsed {remain:s} '
                          'Loss: {loss.val:.4f}({loss.avg:.4f}) '.format(
                                step, len(valid_loader), batch_time=batch_time,
                                data_time=data_time, loss=losses,
                                remain=timeSince(start, float(step + 1) / len(valid_loader))
                            ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def test_fn(test_loader, model, criterion, device):
    avg_test_loss, preds = valid_fn(test_loader, model, criterion, device)
    return avg_test_loss, preds


# ====================================================
# Train loop
# ====================================================
def train_loop(_train_folds, fold, test_fold):
    Logger().info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = _train_folds[_train_folds['fold'] != fold].index
    val_idx = _train_folds[_train_folds['fold'] == fold].index

    train_folds = _train_folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = _train_folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds[Params.target_cols].values

    train_dataset = TrainDataset(train_folds,
                                 transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds,
                                 transform=get_transforms(data='valid'))
    test_dataset = TrainDataset(test_fold,
                                transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset,
                              batch_size=Params.batch_size,
                              shuffle=True,
                              num_workers=Params.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=Params.batch_size * 2,
                              shuffle=False,
                              num_workers=Params.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=Params.batch_size,
                             shuffle=False,
                             num_workers=Params.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if Params.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=Params.factor, patience=Params.patience,
                                          verbose=True,
                                          eps=Params.eps)
        elif Params.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=Params.T_max, eta_min=Params.min_lr, last_epoch=-1)
        elif Params.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=Params.T_0, T_mult=1, eta_min=Params.min_lr,
                                                    last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomResNext(Params.model_name, pretrained=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=Params.lr, weight_decay=Params.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss()
    best_score = 0.
    best_loss = np.inf

    for epoch in range(Params.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score, scores = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        Logger().info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        Logger().info(f'Epoch {epoch + 1} - Score: {score:.4f}  Scores: {np.round(scores, decimals=4)}')

        """
        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth')
        """

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            Logger().info(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'preds': preds},
                       Params.output_dir + f'{Params.model_name}_fold{fold}_best.pth')

    check_point = torch.load(Params.output_dir + f'{Params.model_name}_fold{fold}_best.pth')
    for c in [f'pred_{c}' for c in Params.target_cols]:
        valid_folds[c] = np.nan
    valid_folds[[f'pred_{c}' for c in Params.target_cols]] = check_point['preds']

    # test
    avg_test_loss, preds = test_fn(test_loader, model, criterion, device)

    test_labels = test_fold[Params.target_cols].values
    test_score, test_scores = get_score(test_labels, preds)

    Logger().info(f"========== fold: {fold} test ==========")
    Logger().info(f'avg_val_loss: {avg_test_loss:.4f}')
    Logger().info(f'Score: {test_score:.4f}  Scores: {np.round(test_scores, decimals=4)}')

    return valid_folds, test_scores
