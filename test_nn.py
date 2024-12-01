import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#from nn import NeuralNet
from torch.utils.data import TensorDataset, DataLoader
#INIT:
#we set the hyper parameters
input_size = 0
hidden_size = 20
num_classes = 2
num_epochs = 2
batch_size = 50
learning_rate = 0.001
#we initate a device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#LOAD DATA:
#Load the CSV file
dataframe = pd.read_csv("clean_test.csv")

data_x=dataframe.iloc[:,1:]
#Retain physical_part_id
physical_part_ids = dataframe['physical_part_id']
#Exclude physical_part_id from features
data_x = dataframe.drop(columns=['physical_part_id'])
data_x = (data_x - data_x.mean()) / (data_x.std() + 1e-8)
input_size=data_x.shape[1]
#turn to tensor
x = torch.tensor(data_x.values, dtype=torch.float)

x_test=x
#we turn tensors into a dataset
test_dataset = TensorDataset(x_test)
#create DataLoaders
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


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
#we load the model
model_path="./data/model.pth"
model = torch.load(model_path)

#Evaluate the model
predictions_list = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for features in test_loader:
        features = features.to(device)
        outputs = model(features)
        
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        predictions_list.extend(predictions)
print('testing complete!')
#Convert predictions and IDs to a DataFrame
results_df = pd.DataFrame({
    'physical_part_id': physical_part_ids,
    'prediction': predictions_list
})

#Save to CSV
results_df.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")