import pandas as pd
from sklearn.utils import resample
#we open the csv
df = pd.read_csv('clean.csv')

# Separate the majority and minority classes
majority = df[df['status'] == 1]
minority = df[df['status'] == 0]

target=len(minority)*3
temp_target=len(minority)*3
print(f"majority count: {len(majority)}")
print(f"minority count: {len(minority)}")
print(f"target count: {target}")
# Upsample the minority class
minority_upsampled = resample(minority,
                              replace=True,  # sample with replacement
                              n_samples=temp_target,  # match the majority class
                              random_state=42)

# Downsample the majority class
majority_downsampled = resample(majority,
                                replace=False,  # sample without replacement
                                n_samples=target,  # match the minority class
                                random_state=42)

# Combine the majority and upsampled minority
balanced_df = pd.concat([majority_downsampled, minority_upsampled])
# Shuffle the DataFrame
print("shuffling data...")
shuffled_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
majority = len(shuffled_df[shuffled_df['status'] == 1])
minority = len(shuffled_df[shuffled_df['status'] == 0])
print(f"new majority count: {majority}")
print(f"new minority count: {minority}")
print("saving...")
shuffled_df.to_csv("balanced.csv",index=False)
print("Balancing completed!\n")

