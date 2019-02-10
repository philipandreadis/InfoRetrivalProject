from partA import binaryVector
import csv
import math
from collections import Counter
import numpy

th = 0.8 # threshold for most similar pairs
m = 100 # number of pairs to be tested


# Defines pairs with similarity over a threshold
# Returns a list with the most common pairs
def mostCommon(model,th):
    threshold = th
    commonPairs = []
    for i in range(1,len(lines)):
        if lines[i][model]>=threshold:
            commonPairs.append(lines[i])
    return commonPairs


lines = []

# Csv file processing
with open("train_original.csv", encoding="utf8") as f:
    reader = csv.reader(f, delimiter=",")
    c = 0
    for i, line in enumerate(reader):
        lines.append(line)
        if i > m:
            break


# Lines iteration
for i in range(1,len(lines)):
    s1 = lines[i][3]
    s2 = lines[i][4]
    #string preprocessing
    s1 = s1.lower()
    s2 = s2.lower()
    s1 = s1.strip('?')
    s2 = s2.strip('?')
    # Calculate binary similarity
    bsim = binaryVector(s1, s2)
    lines[i].append(bsim)

cpairs = mostCommon(6, th)
print("Pair of questions with similarity probability over {}:\n".format(th))
for i in range(len(cpairs)):
    print("id:{} \n question1: {} \n question2: {} \n  binary vector sim probability: {}\n".format(cpairs[i][0],cpairs[i][3],cpairs[i][4],cpairs[i][6]))

