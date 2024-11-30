import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('cleaned.csv')
#we want to plot censor data in 1 plot
def plot_sensors(df):
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
    fig.savefig('censors.png')

#this function plots the distribution of Ok and not Ok
def plot_OK(df):
    fig, ax = plt.subplots()
    x = df['status']
    ax.hist(x, bins=2, color='orange',linewidth=0.5, edgecolor="white")
    # Set custom tick labels for the bins
    ax.set_xticks([0.25, 0.75])  # Set ticks in the middle of the bins
    ax.set_xticklabels(['NOK', 'OK'])  # Assign custom labels to the ticks

    # Optional: Set labels and title for clarity
    ax.set_xlabel('Status')
    ax.set_ylabel('Count')
    ax.set_title('OK vs NOK')
    fig.savefig('ok.png')
#we can call the functions now
plot_sensors(df)
plot_OK(df)
