import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import os

from models import Weather_NN_embed
from dataset import WeatherDataset
from learning import train, test
from utils import train_test_split

device = torch.device("cuda")
n_features = 32
test_split = 0.2
batch_size = 128
num_epochs = 10
window_size = 1

#model = Weather_NN_embed(n_features, window_size).to(device)
acc_list = []
for layers in range(1,15):
    model = Weather_NN_embed(n_features, window_size,layers).to(device)

    dataset = WeatherDataset(interp = False, device = device, window_size=window_size)

    trainloader, testloader = train_test_split(dataset, test_split, batch_size)

    print(len(dataset), len(trainloader), len(testloader), batch_size)

    model, acc = train(model, trainloader, testloader, device, num_epochs)

    acc_list.append(acc)

fig, ax = pyplot.subplots()
ax.plot(np.arange(1,15,1),acc_list, label='acc')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlim(1,15) 
ax.set_ylim(0.80,0.95)
ax.set_xlabel('layers')
ax.set_ylabel('acc')
pyplot.show()

torch.save(model.state_dict(),"testmodel.pth")


def plot_history(d_list,g_list):
    pyplot.subplot(2, 1, 1)
    pyplot.plot(d_list, label='d_loss')
    pyplot.legend()

    pyplot.subplot(2, 1, 2)
    pyplot.plot(g_list, label='g_loss')
    pyplot.legend()

    pyplot.savefig('loss.png')
    pyplot.close()