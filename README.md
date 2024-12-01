# bmw_hackathon
# Our team:
- person
- person
- ...
# Libraries and tools we used:
Our main tools in this project were different python libraries on which the code depends as well as github for collaborative work.
## Python libraries:
1. Pandas
2. Pytorch
3. Numpy
4. Matplotlib
   
# Files:
- cleaner.py
This file cleans data of null values and separates it according to part type, while also saving intermediate results into a new file
- plotter.py
This file plots data produced by cleaner. Needed for presentation.
# Execution of code:
At first we cleaned the data. There were a lot of null values which we had to deal with. Also, a lot of it can be skewed or have low 
variance. We address these problems in the first file:

[cleaning.md](./docs/cleaning.md)

After the first step is done we need to balance it.

When it is ready for training we use the new file as input in ```nn.py``` file. It trains and tests the model on
the ```train.csv``` data set. 

[model.md](./docs/model.md)

## Running the code:
The execution of ```plotter.py``` can be done at any time after the cleaner is done.
```
python cleaner.py
python balancer.py
python nn.py
python plotter.py
```
The same can be done by simply running:
```
./train.sh
```

In order to predict the following sequence of commands need to be run:
```
python fit_data.py
python test_nn.py
```

In order to remove all csv files when done the following command is used
```
./remove_csv
```