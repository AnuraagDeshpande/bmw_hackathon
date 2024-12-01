import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
#INIT:
#we set the hyper parameters
input_size = 0
hidden_size = 60
num_classes = 2
num_epochs = 1
batch_size = 50
learning_rate = 0.001
#we initate a device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#LOAD DATA:
#Load the CSV file
dataframe = pd.read_csv("clean.csv")
data_y=dataframe['status']
data_x=dataframe.iloc[:,1:]
input_size=data_x.shape[1]
print("OG:")
dataframe.info()
print("Y:")
data_y.info()
print("X:")
data_x.info()

data_x = (data_x - data_x.mean()) / (data_x.std() + 1e-8)

#turn to tensor
x = torch.tensor(data_x.values, dtype=torch.float)
y = torch.tensor(data_y.values, dtype=torch.float)
#i want to slice it to a smaller size for experiments
print(f'the number of rows = {x.shape[0]}')
rows=x.shape[0]
test_size=0.2
length = int(rows*(1-test_size))
y_train=y[:length]
x_train=x[:length,:]
print(f"X tensor is {x_train.size()} Y tensor is {y_train.size()}")
print(x_train)
#we slice the tensor
y_test=y[length:]
x_test=x[length:,:]
#we turn tensors into a dataset
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
#create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 1)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        #no activation and no softmax at the end
        return out

model = NeuralNet(input_size, hidden_size).to(device)

#Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

#Train the model
losses = []
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):  
        features = features.to(device)
        labels = labels.to(device, dtype=torch.float).view(-1,1)  # Ensure labels are `long` for classification
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

plt.figure(figsize=(10, 6))
plt.plot(losses, label="Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.savefig("./images/losses.png")  # Save the figure as a PNG file
plt.close()

model.eval()
# Evaluate the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device, dtype=torch.float).view(-1, 1)
        outputs = model(features)
        
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        n_correct += (predictions == labels).sum().item()
        n_samples += labels.size(0)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the test set: {acc:.2f}%')
#save the model
model_path="./data/model.pth"
torch.save(model.state_dict(), model_path)
print("Training completed!\n")