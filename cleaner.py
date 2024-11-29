'''
GROUP FOR BMW HACKATHON
This file is supposed to clean the data from an abundance of null values
'''
import pandas as pd
#we open the csv
df = pd.read_csv('train.csv')
og_size=df.size#size is saved
print(f'The original frame: {df.size}')

#NULLS
#we clean the null values in the table
print(f'the types in the table are: {set(df.dtypes)}')
dtypes = df.dtypes.to_dict()
my_type='float64'
#for every column we replace nulls there with mean value if it stores numbers
for col_name, typ in dtypes.items():
    if(typ==my_type):#check type
        x = df[col_name].mean()
        df[col_name] = df[col_name].fillna(x)
#we check the sizing and contents of a column
print(f'types of parts{set(df['physical_part_type'])}')
print(f'types of shifts: {set(df['shift'])}')
print(f'types of OK: {set(df['status'])}')
print(f'The final frame: {df.size}')
#NORMALIZATION
#function for normalizing:
def norm(col):
    return col  / col.abs().max() 
#we go through the table
for col_name, typ in dtypes.items():
    if(typ==my_type):#check type
        df[col_name]=norm(df[col_name])
#MAPPING
mapping = {'OK': 1, 'NOK': 0}
df['status']=df['status'].map(mapping)
#SEPARATION
'''
We want to separate the type 1 from type 2,4 and nan
'''
#we filter the data based on part type
df_type1 = df[df['physical_part_type'] == 'type1'].drop(columns=['physical_part_type'])
df_type2 = df[df['physical_part_type'] == 'type2'].drop(columns=['physical_part_type'])
df_type4 = df[df['physical_part_type'] == 'type4'].drop(columns=['physical_part_type'])
df_nan = df[df['physical_part_type'].isna()].drop(columns=['physical_part_type'])
#we count the sizes
df = df.drop(columns=['physical_part_type'])
new_og_size=df.size
size1=df_type1.size
size2=df_type2.size
size4=df_type4.size
size_nan=df_nan.size
total=sum([size1, size2, size4, size_nan])
#final values are printed in the terminal in order to debug in case something is wrong
print(f'The size of part type 1: {size1}')
print(f'The size of part type 2: {size2}')
print(f'The size of part type 4: {size4}')
print(f'The size of part type nan: {size_nan}')
print(f'together that comes to {total}')
print(f'difference is = {new_og_size-total}')
#data is saved back to different csvs
df_type1.to_csv("type1.csv")
df_type2.to_csv("type2.csv")
df_type4.to_csv("type4.csv")
df_nan.to_csv("type_nan.csv")