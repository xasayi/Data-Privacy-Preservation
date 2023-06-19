import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def plot_loss_acc(train, valid, type, folder):
    plt.figure()
    plt.plot(train, label = f'Train {type}')
    plt.plot(valid, label = f'Valid {type}')
    plt.xlabel('Steps')
    plt.xlabel(type)
    plt.legend()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{type}.png')

def model_performance(model, test_seq, test_mask, test_y, device):
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
    
    preds = np.argmax(preds, axis = 1)
    print(classification_report(test_y, preds))
    # confusion matrix
    print(pd.crosstab(test_y, preds))