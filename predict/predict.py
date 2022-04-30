import torch
import numpy as np
import json


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


if __name__ == '__main__':
    with open('../config.json') as f:
        config = json.load(f)
    # 加载模型
    alice = torch.load('.{}/Alice.pth'.format(config['model_path'])).cpu()
    bob = torch.load('.{}/Bob.pth'.format(config['model_path'])).cpu()
    # 生成明文和key
    # p = 2. * np.random.randint(0, 2, size=24) - 1.
    # k = 2. * np.random.randint(0, 2, size=24) - 1.
    p = np.array([-1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1])
    k = np.array([-1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1])
    # 转换p格式
    p = p.astype(np.float32)
    p = torch.from_numpy(p)
    # 转换k格式
    k = k.astype(np.float32)
    k = torch.from_numpy(k)
    # 加密和解密
    c, p_bob = predict(alice, bob, p, k)
    # 改变编码
    # c = change_encoding(c)
    p_bob = change_encoding(p_bob)

    p_o = list(p.numpy().astype(np.int))
    correct = 0

    for i in range(config['plain']):
        if p_o[i] == p_bob[i]:
            correct += 1

    print("明文：{}".format(p_o))
    print("密钥：{}".format(list(k.numpy().astype(np.int))))
    print("密文：{}".format(c))
    print("解密之后的明文：{}".format(p_bob))
    print("正确率：{:.3f}%".format((correct / config['plain']) * 100))
