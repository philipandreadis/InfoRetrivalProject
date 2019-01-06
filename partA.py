import numpy
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn

pd.set_option('display.max_columns', None)
trainData = pd.read_csv('train_original.csv')
print(trainData.head())


