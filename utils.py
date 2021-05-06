import pandas as pd
import numpy as np
import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler



def Sequential_data(X,y,device,window_size=1):
    item_list = []
    city_item_list = []
    item_y = []
    city_y = []
    last = 0
    for i in range(y.shape[0]):
        if int(X[i,0]) == last:
            city_item_list.append(X[i,:])
            city_y.append(y[i])
            last = int(X[i,0])
        else:
            item_list.append(city_item_list)
            item_y.append(city_y)
            city_item_list = []
            city_y = []
            city_item_list.append(X[i,:])
            city_y.append(y[i])
            last = int(X[i,0])

    #print(len(item_y))
    y_ = []
    X_ = []
    for i in range(len(item_list)):
        
        for j in range(0,len(item_list[i]),window_size):
            if j+window_size-1>=len(item_list[i]):
                continue

            X_.append(np.vstack(item_list[i][j:j+window_size]))
       
            y_.append(item_y[i][j+window_size-1])
    X_=torch.tensor(X_,device = device)
    y_=torch.tensor(y_,device = device)
    return X_, y_

def load_data(interp = False, device = torch.device("cpu"),window_size=1):
    data = pd.read_csv('weatherAUS.csv').replace(('Yes', 'No'), (1, 0))
    if interp:
        data = data.interpolate(method ='pad', limit_direction ='forward')
    data = data.dropna()
    X = featurize(data.values[:,:-1])
    

    y = torch.tensor(data.values[:,-1:].astype(float), dtype = torch.long).squeeze(-1)

    
    X, y = Sequential_data(X,y,device,window_size)

    #print(X.shape)
    #print(y_.shape)

    return X, y

def featurize(X_raw, device= torch.device("cpu")):
    return torch.tensor(list(map(featurize_one, X_raw)), dtype = torch.float, device = device)

def featurize_one(row):

    return np.concatenate((loc_encoding1(row[1]),
                           time_encoding(row[0]),
                           row[2:7],
                           wind_encoding(row[7]),
                           [row[8]],
                           wind_encoding(row[9]),
                           wind_encoding(row[10]),
                           row[11:])).astype(float)

def sin_cos(n):
    theta = 2 * np.pi * n
    return (np.sin(theta), np.cos(theta))

def loc_encoding(loc):
    locs = np.array(['Cobar', 'CoffsHarbour', 'Moree', 'NorfolkIsland', 'Sydney', 'SydneyAirport',
            'WaggaWagga', 'Williamtown', 'Canberra', 'Sale', 'MelbourneAirport',
            'Melbourne', 'Mildura', 'Portland', 'Watsonia', 'Brisbane', 'Cairns',
            'Townsville', 'MountGambier', 'Nuriootpa', 'Woomera', 'PerthAirport', 'Perth',
            'Hobart', 'AliceSprings', 'Darwin'])
    return np.where(locs == loc, 1, 0)

def loc_encoding1(loc):
    locs = np.array(['Cobar', 'CoffsHarbour', 'Moree', 'NorfolkIsland', 'Sydney', 'SydneyAirport',
            'WaggaWagga', 'Williamtown', 'Canberra', 'Sale', 'MelbourneAirport',
            'Melbourne', 'Mildura', 'Portland', 'Watsonia', 'Brisbane', 'Cairns',
            'Townsville', 'MountGambier', 'Nuriootpa', 'Woomera', 'PerthAirport', 'Perth',
            'Hobart', 'AliceSprings', 'Darwin'])
    
    return (np.where(locs==loc)[0])


def time_encoding(time):
    d = datetime.datetime.strptime(time, '%Y-%m-%d')
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return list(sum(map(lambda t: sin_cos(t), [(d.month - 1)/ 12, (d.day - 1) / months[d.month - 1], d.weekday() / 7]), ()))

def wind_encoding(dir):
    dirs = ['E', 'ENE', 'NE', 'NNE', 'N', 'NNW', 'NW', 'WNW', 'W', 'WSW', 'SW', 'SSW', 'S', 'SSE', 'SE', 'ESE']
    angle = dirs.index(dir) / 16 * np.pi
    return sin_cos(angle)

def train_test_split(dataset, test_split, batch_size):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_size = int(test_split * dataset_size)
    test_idx = np.random.choice(indices, size=test_size, replace=False)
    train_idx = list(set(indices) - set(test_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = DataLoader(dataset, batch_size = batch_size, sampler=train_sampler)
    testloader = DataLoader(dataset, batch_size = batch_size, sampler=test_sampler)

    return trainloader, testloader

if __name__ == '__main__':
    X, y = load_data()
    print(X.shape, y.shape)
    print(X[0], y[0])
    print(wind_encoding('W'))
