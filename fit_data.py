'''
GROUP FOR BMW HACKATHON
This file is for the data to be fit before being inputed into the network
'''
import pandas as pd
#we open the csv
df = pd.read_csv('test.csv')
print(f'The inputed frame: {df.shape}')

#DROP COLUMNS LIKE WHILE TRAINING----------------------------------------------
columns_to_remove=open("./data/deleted.txt").readlines()
columns_to_remove=[el.replace("\n",'') for el in columns_to_remove]
columns_to_remove=[el for el in columns_to_remove if el!= 'physical_part_id']
df.drop(columns=columns_to_remove, inplace=True)
dtypes = df.dtypes.to_dict()
#NULLS-------------------------------------------------------------------------
#for every column we replace nulls there with mean value if it stores numbers
my_type='float64'
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
#NORMALIZATION-----------------------------------------------------------------
#function for normalizing each column to the range [0, 1] 
norms = pd.read_csv("norm.csv")
# Normalize function
def norm(col, col_name):
    # Get the Min and Range for the given column name
    col_min = norms.loc[norms["Column"] == col_name, "Min"].values[0]
    col_range = norms.loc[norms["Column"] == col_name, "Range"].values[0]
    # Normalize the column using the Min and Range
    return (col - col_min) / col_range
#we go through the table
for col_name, typ in dtypes.items():
    if(typ==my_type):#check type
        df[col_name] = norm(df[col_name], col_name)
#MAPPING-----------------------------------------------------------------------
mapping = {'type1': 1, 'type2': 2, 'type4':3}
df['physical_part_type']=df['physical_part_type'].map(mapping)
#DROP NAN IN PHYSICAL PART TYPE------------------------------------------------
df = df.dropna(subset=['physical_part_type'])
#we save the test data
print("data fitted:\n")
df.info()
df.to_csv('clean_test.csv', index=False)
