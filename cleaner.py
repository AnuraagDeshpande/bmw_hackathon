'''
GROUP FOR BMW HACKATHON
This file is supposed to clean the data from an abundance of null values
'''
import pandas as pd
#we open the csv
df = pd.read_csv('train.csv')
print(f'The original frame: {df.shape}')

#NULLS-------------------------------------------------------------------------
#we drop not needed data
df = df.drop(columns=['message_timestamp','weekday','shift','physical_part_id'])
#some columns have a lot of missing data and we drop it
# Set the threshold (45% nulls)
threshold = 0.45
# Find the columns with more than 45% null values
columns_to_drop = df.columns[df.isnull().mean() > threshold]
# Drop those columns
df = df.drop(columns=columns_to_drop)

#we clean the null values in the table
print(f'the types in the table are: {set(df.dtypes)}')
dtypes = df.dtypes.to_dict()
my_type='float64'

#for every column we replace nulls there with mean value if it stores numbers
for col_name, typ in dtypes.items():
    if(typ==my_type):#check type
        skewed=df[col_name].skew()

        if abs(skewed) > 1:  #Highly skewed
            replacement_value = df[col_name].median()
        else:  #Not highly skewed
            replacement_value = df[col_name].mean()
        
        df[col_name] = df[col_name].fillna(replacement_value)
#we check the sizing and contents of a column
print(f'The cleaned frame: {df.shape}')
print(df.info())
print(df.describe())

#NORMALIZATION-----------------------------------------------------------------
#function for normalizing each column to the range [0, 1]
def norm(col):
    return (col - col.min()) / (col.max() - col.min()) 
#we go through the table
for col_name, typ in dtypes.items():
    if(typ==my_type):#check type
        df[col_name]=norm(df[col_name])

#MAPPING
mapping = {'OK': 1, 'NOK': 0}
df['status']=df['status'].map(mapping)

df.to_csv("cleaned.csv")
#SEPARATION
'''
We want to separate the type 1 from type 2,4 and nan
'''
#we filter the data based on part type
df_type1 = df[df['physical_part_type'] == 'type1'].drop(columns=['physical_part_type'])
df_type2 = df[df['physical_part_type'] == 'type2'].drop(columns=['physical_part_type'])
df_type4 = df[df['physical_part_type'] == 'type4'].drop(columns=['physical_part_type'])
df_nan = df[df['physical_part_type'].isna()].drop(columns=['physical_part_type'])
#data is saved back to different csvs
df_type1.to_csv("type1.csv")
df_type2.to_csv("type2.csv")
df_type4.to_csv("type4.csv")
df_nan.to_csv("type_nan.csv")