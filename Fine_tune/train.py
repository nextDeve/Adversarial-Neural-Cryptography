from Fine_tune.dataset import BitsDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import itertools
import pandas as pd
import math
from Fine_tune.validate import validate
import traceback


def train(config, k):
    dataset_train = BitsDataset(config, True, config['batch_size'], k)
    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    device = config['device']
    alice = torch.load('.{}/Alice.pth'.format(config['model_path']),map_location=device)
    bob = torch.load('.{}/Bob.pth'.format(config['model_path']),map_location=device)
    eve = torch.load('.{}/Eve.pth'.format(config['model_path']),map_location=device)
    optimizer_e = torch.optim.Adam(eve.parameters(), lr=config['lr'])
    optimizer_ab = torch.optim.Adam(list(alice.parameters()) + list(bob.parameters()), lr=config['lr'])
    criterion = nn.L1Loss()
    step = 0
    try:
        loss_history, accuracy_history = [], []
        for epoch in itertools.count(0):
            # validate model_code
            with torch.no_grad():
                acc_e, acc_b = validate(alice, bob, eve, config, k)
            accuracy_history.append([acc_e, acc_b])
            alice.train(), bob.train(), eve.train()
            loader = tqdm(train_loader)
            for plainE, keyE, plainAB, keyAB, diffPlainAB in loader:
                plainE, keyE = plainE.to(config['device']), keyE.to(config['device'])
                plainAB, keyAB = plainAB.to(config['device']), keyAB.to(config['device'])
                diffPlainAB = diffPlainAB.to(config['device'])
                # train model_code
                # Eve
                optimizer_e.zero_grad()
                cipher = alice(plainE, keyE).detach()
                out_e = eve(cipher)
                loss_e = criterion(plainE, out_e)
                loss_e.backward()
                optimizer_e.step()
                loss_e_temp = loss_e.item()

                # Alice & Bob
                optimizer_ab.zero_grad()

                cipher = alice(plainAB, keyAB)
                diffCipher = alice(diffPlainAB, keyAB)

                out_e = eve(cipher)
                out_b = bob(cipher, keyAB)
                out_diff = bob(diffCipher, keyAB)

                loss_diff = criterion(diffCipher, -cipher)
                loss_e = criterion(plainAB, out_e)
                loss_b = criterion(plainAB, out_b)
                loss_b_diff = criterion(diffPlainAB, out_diff)

                loss_all = (loss_b + loss_b_diff) / 2. + (1. - loss_e).pow(2) + loss_diff * 0.001

                loss_all.backward()
                optimizer_ab.step()

                loss_b = loss_b.item()
                loss_all = loss_all.item()
                loss_b_diff = loss_b_diff.item()
                loss_diff = loss_diff.item()

                loss_history.append([step, loss_e_temp, loss_all, loss_b, loss_b_diff, loss_diff])

                step += 1
                max_loss = max(loss_all, loss_b, loss_e_temp)
                if max_loss > 1e8 or math.isnan(max_loss) or epoch >= config['max_epoch']:
                    torch.save(alice, './model_dict/Alice.pth')
                    torch.save(bob, './model_dict/Bob.pth')
                    torch.save(eve, './model_dict/Eve.pth')
                    loss_df = pd.DataFrame(loss_history,
                                           columns=['step', 'loss_e_temp', 'loss_all', 'loss_b', 'loss_b_diff',
                                                    'loss_diff'])
                    accuracy_df = pd.DataFrame(accuracy_history, columns=['acc_e', 'acc_b'])
                    loss_df.to_csv('./loss_data/loss.csv', index=False)
                    accuracy_df.to_csv('./loss_data/acc_loss.csv', index=False)
                    raise Exception("Loss exploded or Exceeding the maximum number of epoch")
                loader.set_description("AB %.02f B %.02f E %.02f DiffB %.2f Diff %.2f step %d" % (
                    loss_all, loss_b, loss_e_temp, loss_b_diff, loss_diff, step))
    except:
        traceback.print_exc()
