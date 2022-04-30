import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, json

img_path = './img'
os.makedirs(img_path, exist_ok=True)

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 12,
        }

loss = pd.read_csv('./loss_data/loss.csv')
plt.figure(figsize=(15, 8), dpi=500)
plt.plot(np.arange(len(loss)), loss['loss_e_temp'], label='Loss_Eve', linewidth=1)
plt.plot(np.arange(len(loss)), loss['loss_all'], label='Loss_All', linewidth=1)
plt.plot(np.arange(len(loss)), loss['loss_b'], label='Loss_Bob', linewidth=1)
plt.plot(np.arange(len(loss)), loss['loss_diff'], label='Loss_Diff', linewidth=1)
plt.xlabel('Steps', font)
plt.ylabel('Loss', font)
plt.legend(prop=font, loc='upper right')
plt.savefig('{}/loss_Fine_tune.png'.format(img_path))

acc = pd.read_csv('./loss_data/acc_loss.csv')
loss = loss[loss['step'] % 250 == 0]
plt.figure(figsize=(15, 8), dpi=300)
plt.plot(np.arange(len(acc)), acc['acc_e'], label='Acc_Eve')
plt.plot(np.arange(len(acc)), acc['acc_b'], label='Acc_Bob')
plt.plot(np.arange(len(acc)), np.ones(len(acc)) * 0.5, label='Random')
plt.xlabel('Epoch', font)
plt.ylabel('Accuracy', font)
plt.legend(prop=font, loc='upper right')
plt.savefig('{}/acc_Fine_tune.png'.format(img_path))

