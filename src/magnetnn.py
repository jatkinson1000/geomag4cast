import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle

torch.set_default_dtype(torch.float64)

#Some code derived from PyTorch tutorial, "Learning the Basics"

# Data load
#nInput = 159
#nOutput = 4
#nDatapts = 100
#data = np.zeros((nInput,nDatapts))
#obj = np.ones((nOutput,nDatapts))
data, obj = pickle.load(open("data_1year_reduced.p","rb"))
datmean =  np.mean(data,axis=1,keepdims=True)
data -= datmean
datstd = np.std(data,axis=1,keepdims=True)
data /= (1e-10+datstd)
# In a perfect world, store these to normalize the targets and make them part of the model. I don't have time!

print(datmean, datstd)


class CustomMagnetDataset(Dataset):
    def __init__(self,data,target):
        self.data = data
        self.obj = target
        # more?

    def __len__(self):
        return np.shape(self.data)[1]

    def __getitem__(self,idx):
        return self.data[:,idx], self.obj[:,idx]

class MagNet(nn.Module):
    def __init__(self,nInput,nOutput):
        super(MagNet, self).__init__()
        #nInput = 159
        nl1 = 200
        nl2 = 200
        #nOutput = 4
        self.FC = nn.Sequential(
            nn.Linear(nInput, nl1),
            nn.ReLU(),
            nn.Linear(nl1,nl2),
            nn.ReLU(),
            nn.Linear(nl2,nOutput),
        )

    def forward(self, x):
        out = self.FC(x)
        return out

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
nInput = 172
nOutput = 4
model = MagNet(nInput,nOutput).to(device)
print(model)

# Hyperparameters
epochs = 5
batch_size = 32
learning_rate = 1e-3  # ??? WAG

# Format data for use
nTestDays = 12*30
nDataptsPerDay = 24
ntest = int(0.2*np.shape(data)[1]) #nDataptsPerDay*nTestDays
testStartIdx = int(min(42,0.8*np.shape(data)[1]-1))    # prevents test data from running over
testEndIdx = testStartIdx+ntest
#print(testStartIdx, testEndIdx)
data_train = np.concatenate((data[:,0:testStartIdx],data[:,testEndIdx:]),axis=1)
obj_train = np.concatenate((obj[:,0:testStartIdx],obj[:,testEndIdx:]),axis=1)
ds_train = CustomMagnetDataset(data_train, obj_train)
ds_test = CustomMagnetDataset(data[:,testStartIdx:testEndIdx],obj[:,testStartIdx:testEndIdx])
train_dataloader = DataLoader(ds_train, batch_size=batch_size)
test_dataloader = DataLoader(ds_test, batch_size=batch_size)

# Loss + Optimizer
loss_fn = nn.MSELoss()    # Mean Square Error
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Stochastic gradient descent (first attempt, nothing too fancy)

# Optimization loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print_every_n = 100
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % print_every_n == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Optimize
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
