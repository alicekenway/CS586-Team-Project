import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

device = torch.device("cuda")
class Weather_NN(nn.Module):

    def __init__(self, n_features, input_len):
        super(Weather_NN, self).__init__()
        self.n_features = n_features
        self.input_len = input_len

        self.fc = nn.Sequential(nn.Linear(self.n_features * self.input_len, 128),
                                nn.Dropout(0.2),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.Dropout(0.2),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.Dropout(0.2),
                                nn.ReLU(),
                                nn.Linear(128, 2))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)


class Weather_NN_embed(nn.Module):

    def __init__(self, n_features, input_len,layers=1):
        super(Weather_NN_embed, self).__init__()
        self.n_features = n_features
        self.input_len = input_len
        self.embed = nn.Embedding(26,3)
        # self.fc = nn.Sequential(nn.Flatten(),
        #                         nn.Linear(self.n_features * self.input_len, 128),
        #                         nn.Dropout(0.2),
        #                         nn.ReLU(),
        #                         nn.Linear(128, 128),
        #                         nn.Dropout(0.2),
        #                         nn.ReLU(),
         
        #                         nn.Linear(128, 128),
        #                         nn.Dropout(0.2),
        #                         nn.ReLU(),
        #                         nn.Linear(128, 2))

        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(self.n_features * self.input_len, 128),
                                nn.Dropout(0.2),
                                nn.ReLU())
        
        for i in range(layers):
            self.fc.add_module('Linear_{}'.format(i),nn.Linear(128, 128))
            self.fc.add_module('Dropout_{}'.format(i),nn.Dropout(0.2))
            self.fc.add_module('ReLU_{}'.format(i),nn.ReLU())
        self.fc.add_module('output',nn.Linear(128, 2))

    def forward(self, x):
   
        x1 = x[:,:,0].type(torch.cuda.LongTensor)
        x2 = x[:,:,1:]
  
        x1 = self.embed(x1)
    
        x = torch.cat((x1,x2),2)
        x = x.view(x.size(0), -1)
 
        x = self.fc(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)
