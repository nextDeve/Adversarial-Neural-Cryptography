import torch
import numpy as np
import json
import distance
from tqdm import tqdm
import pandas as pd


def predict(encoder, decoder, p, k):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # 加密
        c = encoder(p, k)
        # 解密
        p_bob = decoder(c, k)
    c = list(c.numpy())
    p_bob = list(p_bob.numpy())
    return c, p_bob


def change_encoding(code):
    res = []
    for item in code:
        if item < 0:
            res.append(-1)
        else:
            res.append(1)
    return res


def hamming_distance(x: int, y: int) -> int:
    return bin(x ^ y).count('1')


if __name__ == '__main__':
    with open('../config.json') as f:
        config = json.load(f)
    # 加载模型
    alice = torch.load('.{}/Alice.pth'.format(config['model_path'])).cpu()
    bob = torch.load('.{}/Bob.pth'.format(config['model_path'])).cpu()
    pks = np.load('../model_code/pk.npz', allow_pickle=True)
    ps, ks = pks['p'], pks['k']
    alice.eval()
    bob.eval()
    distances = []
    with torch.no_grad():
        for i in tqdm(range(len(ps))):
            p, k = ps[i], ks[i]
            # 转换p格式
            p = p.astype(np.float32)
            p = torch.from_numpy(p)
            # 转换k格式
            k = k.astype(np.float32)
            k = torch.from_numpy(k)
            c_p = alice(p, k)
            c_p = c_p.numpy()
            str_o = []
            try:
                for n in range(config['cipher']):
                    # 密文一共24位，每一位密文原文都是小数，只取小数点后一位计算距离
                    str_o.append(int(str(c_p[n])[3:4]) if c_p[n] < 0 else int(str(c_p[n])[2:3]))
                for j in range(config['plain']):
                    newP = p
                    newP[j] = -1 if p[j] == 1 else 1
                    c_n = alice(newP, k)
                    c_n = c_n.numpy()
                    d = 0
                    for m in range(config['cipher']):
                        d += hamming_distance(str_o[m], int(str(c_n[n])[3:4]) if c_n[n] < 0 else int(str(c_n[n])[2:3]))
                    distances.append(d)
            except:
                pass
    pd.DataFrame({'distance': distances}).to_csv('distance.csv', index=False)
