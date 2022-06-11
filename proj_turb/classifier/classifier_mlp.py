from cv2 import minMaxLoc
from torch import nn, optim, from_numpy, tensor, sigmoid
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# train_df = pd.read_csv('solution_data.csv')
# #train_df['pf'] = train_df['pf'].map({'p': 1, 'f':0})
# train_df['target'] = train_df['target'].map({'p0':2, 'f2':1, 'f1':0})

# min_max_scaler = MinMaxScaler()
# fitted = min_max_scaler.fit(train_df.iloc[:, 1:-2])
# normalization_df = min_max_scaler.transform(train_df.iloc[:, 1:-2])

# x = normalization_df



##dataset class
class SolutionDataset(Dataset):
    def __init__(self):
        ##data preprocessing
        train_df = pd.read_csv('solution_data.csv')
        #train_df['pf'] = train_df['pf'].map({'p': 1, 'f':0})
        train_df['target'] = train_df['target'].map({'p0':2, 'f2':1, 'f1':0})
        #train_df['target'] = train_df['target'].map({'p0':1, 'f2':0, 'f1':0})
        
        train_df['target'] = train_df['target'].astype('float64')
        train_df.iloc[:, [2]] /=100
        train_df.iloc[:, [3]] /=1000
        train_df.iloc[:, [4]] /=100
        train_df.iloc[:, [5]] /=100
        train_df.iloc[:, [6]] /=100
        train_df.iloc[:, [7]] /= 10000
        train_df.iloc[:, [8]] /= 1000
        train_df.iloc[:, [10]] /= 10000
        # min_max_scaler = MinMaxScaler()
        # fitted = min_max_scaler.fit(train_df.iloc[:, 1:-2])
        # normalization_df = min_max_scaler.transform(train_df.iloc[:, 1:-2])
        
        # x = normalization_df
        x = train_df.iloc[:, 2:-1].values
        y = train_df.iloc[:, -1].values
        #print('x: ', x)
        #print('y: ', y)
        self.x_data = torch.FloatTensor(x)
        self.y_data = torch.LongTensor(y)
        self.len = x.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

dataset_ = SolutionDataset()
train_loader = DataLoader(dataset=dataset_,
                          batch_size=4,
                          shuffle=True)

##MLP model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(9, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 3)
        # self.linear2 = nn.Linear(256, 128)
        # self.linear3 = nn.Linear(128, 64)
        # self.linear4 = nn.Linear(64, 32)
        # self.linear5 = nn.Linear(32, 16)
        # self.linear6 = nn.Linear(16, 10)
        # self.linear7 = nn.Linear(10, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.linear1(x))
        out2 = self.relu(self.linear2(out1))
        # out3 = self.relu(self.linear3(out2))
        # out4 = self.relu(self.linear4(out3))
        # out5 = self.relu(self.linear5(out4))
        # out6 = self.relu(self.linear6(out5))
        # y_pred = self.relu(self.linear7(out6))
        #y_pred = F.log_softmax(y_pred, dim=1)
        y_pred = self.linear3(out2)
        return y_pred
    


def train():
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    
    for epoch in range(100):
        model.train()
        for i, data in enumerate(train_loader,0):
            inputs, labels = data
            #labels = labels.squeeze_(dim=-1)
            print(inputs, labels)
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(f'Epoch: {epoch} | Batch: {i+1} | Loss: {loss.item(): .4f}')
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model


if __name__ == '__main__':
    train_df = pd.read_csv('solution_data.csv')
    #train_df['pf'] = train_df['pf'].map({'p': 1, 'f':0})
    train_df['target'] = train_df['target'].map({'p0':2, 'f2':1, 'f1':0})
    #train_df['target'] = train_df['target'].map({'p0':1, 'f2':0, 'f1':0})
    
    train_df['target'] = train_df['target'].astype('float64')
    y = train_df.iloc[:, -1].values
        
    train_df.iloc[:, [2]] /=100
    train_df.iloc[:, [3]] /=1000
    train_df.iloc[:, [4]] /=100
    train_df.iloc[:, [5]] /=100
    train_df.iloc[:, [6]] /=100
    train_df.iloc[:, [7]] /= 10000
    train_df.iloc[:, [8]] /= 1000
    train_df.iloc[:, [10]] /= 10000
    #min_max_scaler = MinMaxScaler()
    #fitted = min_max_scaler.fit(train_df.iloc[:, 1:-1])
    #normalization_df = min_max_scaler.transform(train_df.iloc[:, 1:-1])
    
    
    
    model = train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = 0
    correct = 0 

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(train_loader,0):
            inputs, labels = data
            inputs, labels = torch.FloatTensor(inputs), torch.LongTensor(labels)
            output = model(inputs)
            loss += criterion(output, labels).item()
            prediction = output.max(dim=1)[1]

            correct += prediction.eq(labels.view_as(prediction)).sum().item()
 
    loss /= (len(train_loader.dataset))
    accuracy = 100. * correct / len(train_loader.dataset)
    print('acc: ',accuracy)

    x = torch.FloatTensor(train_df.iloc[[2], 2:-1].values)
    #y = torch.LongTensor(train_df.iloc[:, -1].values)
    #print(y)
    #x = torch.FloatTensor(normalization_df[[1], :])
    print(x)
    #print(tensor(torch.from_numpy(train_df.iloc[1, 1:-2].values)))
    print(model(x))
    print(model(x).max(dim=1)[1])
    
    result = model(torch.FloatTensor(train_df.iloc[:, 2:-1].values))
    #print(result)
    print(result.max(dim=1)[0])
    print(result.max(dim=1)[1])
