# Outputs
## clearer.py
```
$ python cleaner.py
The original frame: (44818, 376)
the types in the table are: {dtype('O'), dtype('float64')}
The frame wthout nulls: (44818, 326)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 44818 entries, 0 to 44817
Columns: 326 entries, physical_part_type to s3_sensor3_newtonmeter_step1
dtypes: float64(324), object(2)
memory usage: 111.5+ MB
None
The variance filtered frame: (44816, 286)
/home/tim/Documents/Hackathon2911/bmw_hackathon/cleaner.py:72: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  min_max_df = pd.concat([min_max_df, pd.DataFrame({'Column': [col_name], 'Min': [col_min], 'Range': [col_range]})], ignore_index=True)
the data is clear. Here is the info before separation:
<class 'pandas.core.frame.DataFrame'>
Index: 44816 entries, 0 to 44817
Columns: 286 entries, physical_part_type to s3_sensor3_newtonmeter_step1
dtypes: float64(284), int64(2)
memory usage: 98.1 MB
type 1 is shape (27007, 285)
type 2 is shape (17755, 285)
type 4 is shape (54, 285)
removed 90 columns were saved to ./data/deleted.txt
Cleaning completed!

```
## balancer.py
```
$ python balancer.py
majority count: 42542
minority count: 2274
target count: 6822
shuffling data...
new majority count: 6822
new minority count: 6822
saving...
Balancing completed!

```
## nn.py
```
$ python nn.py
OG:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 13644 entries, 0 to 13643
Columns: 286 entries, physical_part_type to s3_sensor3_newtonmeter_step1
dtypes: float64(284), int64(2)
memory usage: 29.8 MB
Y:
<class 'pandas.core.series.Series'>
RangeIndex: 13644 entries, 0 to 13643
Series name: status
Non-Null Count  Dtype
--------------  -----
13644 non-null  int64
dtypes: int64(1)
memory usage: 106.7 KB
X:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 13644 entries, 0 to 13643
Columns: 285 entries, status to s3_sensor3_newtonmeter_step1
dtypes: float64(284), int64(1)
memory usage: 29.7 MB
status
0    6822
1    6822
Name: count, dtype: int64
the number of rows = 13644
X tensor is torch.Size([10915, 285]) Y tensor is torch.Size([10915])
tensor([[0.0000, 0.5000, 0.6383,  ..., 0.0000, 0.6752, 0.7553],
        [0.0000, 0.5000, 0.5979,  ..., 0.0000, 0.7303, 0.7759],
        [0.0000, 1.0000, 0.6075,  ..., 0.0000, 0.6216, 0.7963],
        ...,
        [0.0000, 0.5000, 0.5249,  ..., 0.0000, 0.6704, 0.8066],
        [1.0000, 1.0000, 0.5106,  ..., 0.0000, 0.6468, 0.7858],
        [1.0000, 0.5000, 0.5286,  ..., 0.0000, 0.7192, 0.7756]])
Training set: 5429 zeros, 5486 ones
Testing set: 1393 zeros, 1336 ones
Epoch [1/1], Step [100/342], Loss: 0.6089
Epoch [1/1], Step [200/342], Loss: 0.4019
Epoch [1/1], Step [300/342], Loss: 0.2042
Accuracy of the network on the test set: 99.89%
1339 1390
Training completed!
```
## fit_data.py
```
$ python fit_data.py
The inputed frame: (400, 375)
data fitted:

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 400 entries, 0 to 399
Columns: 286 entries, physical_part_type to s3_sensor3_newtonmeter_step1
dtypes: float64(284), int64(1), object(1)
memory usage: 893.9+ KB

```
## test_nn.py
```
$ python test_nn.py
Input features in current model: 285
testing complete!
OK: 400 NOK: 0
Predictions saved to predictions.csv

```