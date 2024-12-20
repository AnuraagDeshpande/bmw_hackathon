'''
GROUP FOR BMW HACKATHON
This file is supposed to clean the data from an abundance of null values
'''
import pandas as pd
#we open the csv
df = pd.read_csv('train.csv')
print(f'The original frame: {df.shape}')
og_columns=set(df.columns)
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
        #else if:  df[col_name].isnull().sum()>0
        #   replacement_value = df [col_name].mode() [0]
        else:  #Not highly skewed
            replacement_value = df[col_name].mean()
        
        df[col_name] = df[col_name].fillna(replacement_value)
#we check the sizing and contents of a column
print(f'The frame wthout nulls: {df.shape}')
print(df.info())
#we want to discard NAN types and map the part types
df = df.dropna(subset=['physical_part_type'])

df.to_csv("prefilter.csv",index=False)
#VARIANCE FILTERING
#our graph showed us a lot of columns with vary low variance. We dont need those
low_variance_threshold = 0.01
columns_to_remove=[]
for col_name, typ in dtypes.items():
    if(typ==my_type):
        variance = df[col_name].var()
        if (variance <= low_variance_threshold or (variance <= low_variance_threshold and abs(df[col_name].mean()-0)<0.01)):
            columns_to_remove.append(col_name)
df=df.drop(columns=columns_to_remove)
df = df.dropna(axis=1, how='all') 
print(f'The variance filtered frame: {df.shape}')
df.to_csv("var.csv",index=False)
#NORMALIZATION-----------------------------------------------------------------
min_max_df = pd.DataFrame(columns=['Column', 'Min', 'Range'])
#function for normalizing each column to the range [0, 1]
def norm(col):
    col_min = col.min()
    col_range = col.max() - col_min
    return (col - col_min) / col_range, col_min, col_range
#we go through the table
dtypes = df.dtypes.to_dict()
for col_name, typ in dtypes.items():
    if(typ==my_type):#check type
        normalized_col, col_min, col_range = norm(df[col_name])
        df[col_name]=normalized_col
        min_max_df = pd.concat([min_max_df, pd.DataFrame({'Column': [col_name], 'Min': [col_min], 'Range': [col_range]})], ignore_index=True)
min_max_df.to_csv('norm.csv', index=False)
#MAPPING
mapping = {'OK': 1, 'NOK': 0}
df['status']=df['status'].map(mapping)
mapping = {'type1': 1, 'type2': 2, 'type4':3}
df['physical_part_type']=df['physical_part_type'].map(mapping)

print("the data is clear. Here is the info before separation:")
df.info()
new_columns=set(df.columns)
df.to_csv('clean.csv',index=False)
#SEPARATION
'''
We want to separate the type 1 from type 2,4
'''
#we filter the data based on part type
df_type1 = df[df['physical_part_type'] == 1].drop(columns=['physical_part_type'])
df_type2 = df[df['physical_part_type'] == 2].drop(columns=['physical_part_type'])
df_type4 = df[df['physical_part_type'] == 3].drop(columns=['physical_part_type'])
frames=[df_type1,df_type2,df_type4]
labels=['type 1', 'type 2', 'type 4']
for i in range(3):
    label=labels[i]
    frame=frames[i]
    print(f'{label} is shape {frame.shape}')
#data is saved back to different csvs
df_type1.to_csv("type1.csv",index=False)
df_type2.to_csv("type2.csv",index=False)
df_type4.to_csv("type4.csv",index=False)

#we save deleted columns
deleted_columns=og_columns-new_columns

file_name="./data/deleted.txt"
with open(file_name, "w") as file:
    #Write each string in the list to the file, followed by a newline
    for string in deleted_columns:
        file.write(string + "\n")

print(f'removed {len(deleted_columns)} columns were saved to {file_name}')
print("Cleaning completed!\n")
#we cleaned the doc.