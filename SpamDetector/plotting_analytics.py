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

def model_performance(model, test_seq, test_mask, test_y, device, folder):
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
    
    preds = np.argmax(preds, axis = 1)
    with open(f'{folder}/results.txt', 'w') as f:
        report = classification_report(test_y, preds)
        confusion_matrix = pd.crosstab(test_y, preds)
        f.write(report)
        f.write(str(confusion_matrix))
        print(report)
        print(str(confusion_matrix))

    