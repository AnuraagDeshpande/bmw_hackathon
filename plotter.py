import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('cleaned.csv')
new_df = pd.read_csv('var.csv')
#we want to plot censor data in 1 plot
def plot_sensors(df, name):
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))  # Create a 5x5 grid of subplots
    axes = axes.flatten()  # Flatten to iterate easily
    cols=25
    #plot each of 25 first columns as a graph
    for i, ax in enumerate(axes[:len(df.columns)]):
        x = df.iloc[:, i+2]
        ax.hist(x, bins=15,  color='purple',linewidth=0.5, edgecolor="white")
        ax.set_title(f"{df.columns[i]}")
    # Adjust spacing between the subplots
    plt.subplots_adjust(hspace=0.6, wspace=0.6)
    fig.savefig('censors'+name+'.png')

#this function plots the distribution of Ok and not Ok
def plot_OK(df, ax):
    x = df['status']
    ax.hist(x, bins=2, color='orange',linewidth=0.5, edgecolor="white")
    # Set custom tick labels for the bins
    ax.set_xticks([0.25, 0.75])  # Set ticks in the middle of the bins
    ax.set_xticklabels(['NOK', 'OK'])  # Assign custom labels to the ticks

    # Optional: Set labels and title for clarity
    ax.set_xlabel('Status')
    ax.set_ylabel('Count')
    ax.set_title('OK vs NOK')
#we have labels for the files and graphs
files=['cleaned.csv','var.csv',"type1.csv","type2.csv","type4.csv","type_nan.csv"]
labels=["","_var", "_type1","_type2","_type4","_nan"]
#we plot the states before the split
for i in range(2):
    file=files[i]
    label=labels[i]
    df = pd.read_csv(file)
    plot_sensors(df,label)
    fig, ax = plt.subplots()
    plot_OK(df, ax)
    fig.savefig('ok'+label+'.png')
#we create graphs for each of the split groups
fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Create a 5x5 grid of subplots
axes = axes.flatten()  # Flatten to iterate easily
for i in range(4):
    indx=i+2
    file=files[indx]
    label=labels[indx]
    df = pd.read_csv(file)
    plot_sensors(df,label)
    ax=axes[i]
    plot_OK(df, ax)
    ax.set_title(label)
fig.savefig('ok'+'_types'+'.png')