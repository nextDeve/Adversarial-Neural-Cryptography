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


def check_right(input_str, length, info):
    values = [-1, 1]
    nums_ = input_str.split(' ')
    if len(nums_) != length:
        print("位数错误，请重新输入{}!".format(info))
        return False
    for num_ in nums_:
        try:
            if int(num_) not in values:
                print("输入错误，值域[-1,1]，请重新输入{}!".format(info))
                return False
        except:
            print("输入错误，值域[-1,1]，请重新输入{}!".format(info))
            return False
    return True


def change_to_num(str_list):
    res = []
    for c in str_list:
        res.append(int(c))
    return np.array(res)


if __name__ == '__main__':
    with open('../config.json') as f:
        config = json.load(f)
    # 加载模型
    alice = torch.load('.{}/Alice.pth'.format(config['model_path'])).cpu()
    bob = torch.load('.{}/Bob.pth'.format(config['model_path'])).cpu()

    flag = True
    p, k = [], []
    print('请输入明文([-1,1],{}位,以空格隔开):'.format(config['plain']))
    while flag:
        nums_str = input()
        if check_right(nums_str, config['plain'], '明文'):
            p = change_to_num(nums_str.split(' '))
            flag = False
    flag = True
    print('请输入密钥([-1,1],{}位,以空格隔开):'.format(config['plain']))
    while flag:
        nums_str = input()
        if check_right(nums_str, config['key'], '密钥'):
            k = change_to_num(nums_str.split(' '))
            flag = False

    # 转换p格式
    p = p.astype(np.float32)
    p = torch.from_numpy(p)
    # 转换k格式
    k = k.astype(np.float32)
    k = torch.from_numpy(k)
    # 加密和解密
    c, p_bob = predict(alice, bob, p, k)
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
