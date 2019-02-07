import pandas as pd
import matplotlib.pyplot as plt
import pylab



#first five elements of the csv
pd.set_option('display.max_columns', None)
trainData = pd.read_csv('train_original.csv')
print(trainData.head())

print('Total number of question pairs for training: {}'.format(len(trainData)))

qids = pd.Series(trainData['qid1'].tolist() + trainData['qid2'].tolist())
plt.figure(figsize=(12, 5))
plt.hist(qids.value_counts(), bins=50)
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
pylab.show()
print()
