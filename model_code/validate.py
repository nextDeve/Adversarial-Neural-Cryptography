from model_code.dataset import BitsDataset
from torch.utils.data import DataLoader
import torch


def validate(alice, bob, eve, config):
    batch_size = config['batch_size']
    device = config['device']
    alice.eval(), bob.eval(), eve.eval()
    dataset_val = BitsDataset(config, False, batch_size)
    val_loader = DataLoader(dataset=dataset_val,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
    correct_e, correct_b = 0, 0
    for plain, key in val_loader:
        plain, key = plain.to(device), key.to(device)
        cipher = alice(plain, key)
        out_eve = eve(cipher)
        out_bob = bob(cipher,key)
        correct_e += torch.sum(torch.abs(plain - out_eve) < 1).item() / plain.shape[1]
        correct_b += torch.sum(torch.abs(plain - out_bob) < 1).item() / plain.shape[1]
    acc_e = correct_e / len(val_loader.dataset)
    acc_b = correct_b / len(val_loader.dataset)
    print('Accuracy(%%): Bob %.1f Eve %.1f' % (100. * acc_b, 100. * acc_e))
    return acc_e, acc_b
