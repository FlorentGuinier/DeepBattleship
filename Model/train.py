import os
import sys
import argparse
import logging
from tqdm import tqdm
from boat_prediction_model import *
from boat_prediction_dataset import *

import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

dir_checkpoint = 'Checkpoints/'

def eval_net(net, states, val_idx, batch_size, device):
    net.eval()
    tot = 0
    numBatches = 0
    with torch.no_grad():
        for i in range(0, len(val_idx) - batch_size + 1, batch_size):
            imgs, targets = MakeBatch(states, val_idx[i:i+batch_size])
            tot += F.binary_cross_entropy_with_logits(net(imgs), targets).item()
            numBatches += 1
    net.train()
    return tot / numBatches

def train_net(net,
              device,
              epochs,
              batch_size,
              lr,
              val_percent,
              save_cp):

    statesArray, boardIndexArray = LoadGameStates()
    numBoards = int(boardIndexArray[-1]) + 1
    firstValBoard = numBoards - max(int(numBoards * val_percent), 1)

    # the whole dataset stays on the device as uint8, batches are sliced and converted there
    states = torch.from_numpy(statesArray).to(device)
    boardIndex = torch.from_numpy(boardIndexArray).to(device)
    # split by board: snapshots of one game must never span train and val
    train_idx = (boardIndex < firstValBoard).nonzero().squeeze(1)
    val_idx = (boardIndex >= firstValBoard).nonzero().squeeze(1)
    n_train = len(train_idx)
    n_val = len(val_idx)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_cp}
            Device:          {device.type}
        ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n_train, device=device)

        numItemSinceVal = 0
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for i in range(0, n_train, batch_size):
                imgs, targets = MakeBatch(states, train_idx[perm[i:i+batch_size]])
                target_preds = net(imgs)
                loss = criterion(target_preds, targets)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                numItemSinceVal += imgs.shape[0]
                if numItemSinceVal > 300000:### Num item before validation
                    numItemSinceVal = 0
                    val_score = eval_net(net, states, val_idx, batch_size, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    logging.info('Validation BCE with logit loss: {}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)
                    writer.add_images('sources', imgs, global_step)
                    writer.add_images('targets', targets, global_step)
                    writer.add_images('target_preds', target_preds, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train boat prediction net')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=64, help='Batch size')
    parser.add_argument('-l', '--learningrate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('-f', '--load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', type=float, default=1.0, help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = BoatPredictionUNet()
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.learningrate,
                  device=device,
                  val_percent=args.validation / 100,
                  save_cp=True)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
