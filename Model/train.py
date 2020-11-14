import os
import sys
import argparse
import logging
from tqdm import tqdm
from boat_prediction_model import *
from boat_prediction_dataset import *

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

dir_img = '../Data/GameStates/'
dir_checkpoint = 'Checkpoints/'

def eval_net(net, loader, device):
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, targets = batch['X'], batch['Y']
            imgs = imgs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                targets_pred = net(imgs)

            tot += F.binary_cross_entropy_with_logits(targets_pred, targets).item()
            pbar.update()

    net.train()
    return tot / n_val

def train_net(net,
              device,
              epochs,
              batch_size,
              lr,
              val_percent,
              save_cp):

    dataset = BoatPredictionDataset()
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

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

        numItemSinceVal = 0
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['X']
                targets = batch['Y']
                imgs = imgs.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.float32)
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
                numItemSinceVal += batch_size
                if numItemSinceVal > 300000:### Num item before validation (low as it require a GPU upload killing perf)
                    numItemSinceVal = 0
                    val_score = eval_net(net, val_loader, device)
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