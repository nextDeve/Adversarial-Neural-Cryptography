import numpy as np
import torch
from torch.utils.data import Dataset


class BitsDataset(Dataset):
    def __init__(self, config, is_train, batch_size):
        self.batch_size = batch_size
        self.train = is_train
        self.plain = config['plain']
        self.key = config['key']
        self.steps = config['steps'][0 if self.train else 1]

    def __len__(self):
        return self.batch_size * self.steps

    def __getitem__(self, idx):
        plainE = self.rand(self.plain)
        keyE = self.rand(self.key)
        plainAB = self.rand(self.plain)
        keyAB = self.rand(self.key)
        diffPlainAB = self.effectRand(plainAB)

        if self.train:
            return plainE, keyE, plainAB, keyAB, diffPlainAB
        else:
            return plainE, keyE

    def rand(self, size):
        # 随机生成长度为size取值为[-1,1]的序列
        x = 2. * np.random.randint(0, 2, size=size) - 1.
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        return x

    def effectRand(self, plainAB):
        # 置反字典
        change_dic = {1: -1, -1: 1}
        # 选取plainAB中随机一个index
        random_plainAB_ids = np.arange(0, self.plain)
        random_plainAB_id = np.random.choice(random_plainAB_ids, 1)
        # tenser 转numpy 对象
        diff_plainAB = plainAB.numpy().copy()
        # plainAB 中随机一位置反
        diff_plainAB[random_plainAB_id[0]] = change_dic[int(diff_plainAB[random_plainAB_id[0]])]
        return torch.from_numpy(diff_plainAB)
