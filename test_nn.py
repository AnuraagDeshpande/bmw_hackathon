import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#from nn import NeuralNet
from torch.utils.data import TensorDataset, DataLoader
#INIT:
#we set the hyper parameters
input_size = 0
hidden_size = 100
num_classes = 2
num_epochs = 10
batch_size = 32
learning_rate = 0.001
#we initate a device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#LOAD DATA:
#Load the CSV file
dataframe = pd.read_csv("clean_test.csv")
#we can't have status
answers=pd.Series()
if "status" in dataframe.columns:
    answers=dataframe['status']
    dataframe=dataframe.drop(columns=['status'])
data_x=dataframe.iloc[:,1:]
#Retain physical_part_id
physical_part_ids = dataframe['physical_part_id']
#Exclude physical_part_id from features
data_x = dataframe.drop(columns=['physical_part_id'])
input_size=data_x.shape[1]
#turn to tensor
x = torch.tensor(data_x.values, dtype=torch.float)

x_test=x
#we turn tensors into a dataset
test_dataset = TensorDataset(x_test)
#create DataLoaders
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(f"Input features in current model: {data_x.shape[1]}")
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
model = NeuralNet(input_size, hidden_size)
model.load_state_dict(torch.load(model_path))
model.eval()

#Evaluate the model
predictions_list = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for batch in test_loader:
        features = batch[0].to(device)
        outputs = model(features)
        
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        predictions_list.extend(prediction.item() for prediction in predictions)

print('testing complete!')
#Convert predictions and IDs to a DataFrame
results_df = pd.DataFrame({
    'physical_part_id': physical_part_ids,
    'prediction': predictions_list
})
#analysis!
ok = len(results_df[results_df['prediction'] == 1])
nok = len(results_df[results_df['prediction'] == 0])
print(f'OK: {ok} NOK: {nok}')#we print the stats
#mapping
mapping={0: "NOK", 1: "OK"}
results_df['prediction']=results_df['prediction'].map(mapping)
#if answers are not empty
if (not answers.empty):
    results_df['answers']=answers
    #we compare:
    comparison = results_df['prediction']==results_df['answers']
    num_matches = comparison.sum()  # True values count as 1
    num_mismatches = len(comparison) - num_matches  # Total elements minus matches
    print(f"Number of matching elements: {num_matches}")
    print(f"Number of non-matching elements: {num_mismatches}")
    print(f"accuracy: {num_matches/len(comparison):.4f}%")

#Save to CSV
results_df.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
