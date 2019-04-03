import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils import data
from earlystop import EarlyStopping
from seed import set_random_seed
import trainer
from load_data import load_data
from model import CLCNN
from preprocess import PrepreprocessData, make_char2idx


def main():
    # hyper parameter
    batch_size = 128
    char_len = 150
    MAX_epoch = 50
    encode_dim = 250
    feat_num = 256
    device = torch.device('cuda:0')
    seed = 2434
    data_path = Path('../data/train.csv')
    patience = 7
    n_fold = 4
    momentum = 0.9
    lr = 0.01
    ##############################
    set_random_seed(seed)

    X_data, y_data = load_data(data_path)

    train = [(x, y) for x, y in zip(X_data, y_data)]

    kf = KFold(n_splits=n_fold)
    for i, (train_idx, valid_idx) in enumerate(kf.split(train)):
        print(f'Fold : {i+1}')
        train_fold = [train[i] for i in train_idx]
        valid_fold = [train[i] for i in valid_idx]

        char2idx, ignore_idx = make_char2idx([text for text, _ in train])

        target_arr = np.array([y for _, y in train_fold])
        weight_dict = {i: np.sum(target_arr == i) for i in range(2)}
        weghit = 1/torch.Tensor([weight_dict[i] for i in target_arr])
        sampler = data.WeightedRandomSampler(weghit, len(weghit))

        train_data = PrepreprocessData(
            train_fold, char_len, char2idx, ignore_idx)
        valid_data = PrepreprocessData(
            valid_fold, char_len, char2idx, ignore_idx)
        train_loader = data.DataLoader(dataset=train_data,
                                       batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        valid_loader = data.DataLoader(dataset=valid_data,
                                       batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = CLCNN(encode_dim, char_len, len(char2idx), ignore_idx, feat_num).to(device)

        loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum)
        early_stopping = EarlyStopping(char2idx, patience=patience, verbose=True)

        for epoch in range(MAX_epoch):
            start_time = time.time()
            train_loss = trainer.train(
                model, train_loader, loss_func, device, optimizer)

            valid_loss = trainer.valid(model, valid_loader, loss_func, device)

            elapsed_time = time.time() - start_time
            print(
                f'Epoch {epoch+1}/{MAX_epoch} \t loss={train_loss:.4f} \t val_loss={valid_loss:.4f} \t time={elapsed_time:.2f}')

            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print('stop')
                break
        print(f'{i}-fold best result valid loss : {early_stopping.best_score:2f}')


if __name__ == "__main__":
    main()
