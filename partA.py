import numpy
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn
import pylab
import csv
import math
from collections import Counter




def truncate(number, digits) -> float:
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper

# Binary vector model.
# qvector is the set of the vocabulary of the two sentences
# vec1 and vec2 represent the two sentences with binary weights
# returns the cosine similarity of vec1-vec2, appends it to lines[][6]
def binaryVector(s1,s2):
        words1 = s1.split()
        words2 = s2.split()
        qvector = list(set(words1+words2))
        vector1 = []
        vector2 = []
        for i in range(len(qvector)):
            if qvector[i] in words1:
                vector1.append(1)
            else:
                vector1.append(0)
        for i in range(len(qvector)):
            if qvector[i] in words2:
                vector2.append(1)
            else:
                vector2.append(0)
        # calculate cosine similarity of the two vectors
        cos = numpy.dot(vector1, vector2) / (numpy.sqrt(numpy.dot(vector1, vector1)) * numpy.sqrt(numpy.dot(vector2, vector2)))
        return cos


# Weighted vector model.
# qvector is the set of the vocabulary of the two sentences
# vec1 and vec2 represent the two sentences with TFxIDF weights
# returns the cosine similarity of vec1-vec2, appends it to lines[][7]
def weightedVector(s1, s2):
    words1 = s1.split()
    words2 = s2.split()
    qvector = list(set(words1 + words2))
    vector1 = []
    vector2 = []
    # find max element freq in the two questions
    most_common1, num_most_common1 = Counter(words1).most_common(1)[0]
    most_common2, num_most_common2 = Counter(words2).most_common(1)[0]
    for i in range(len(qvector)):
        if (qvector[i] in words1):
            tf = words1.count(qvector[i]) / num_most_common1
            nt = 0
            for j in range(1, len(lines)):
                if (qvector[i] in lines[j][3]):
                    nt = nt + 1
                if (qvector[i] in lines[j][4]):
                    nt = nt + 1
            if nt == 0:
                nt = nt +1
            idf = numpy.log((len(lines) - 1) * 2 / nt)
            weight = tf * idf
            vector1.append(weight)
        else:
            vector1.append(0)
    for i in range(len(qvector)):
        if (qvector[i] in words2):
            tf = words2.count(qvector[i]) / num_most_common2
            nt = 0
            for j in range(1, len(lines)):
                if (qvector[i] in lines[j][3]):
                    nt = nt + 1
                if (qvector[i] in lines[j][4]):
                    nt = nt + 1
            if nt == 0:
                nt = nt +1
            idf = numpy.log((len(lines) - 1) * 2 / nt)
            weight = tf * idf
            vector2.append(weight)
        else:
            vector2.append(0)
    # calculate cosine similarity of the two vectors
    cos = numpy.dot(vector1, vector2) / (
                numpy.sqrt(numpy.dot(vector1, vector1)) * numpy.sqrt(numpy.dot(vector2, vector2)))
    return cos




# Appends to lines[][7] the cross entropy loss of the model used
# 6--> Binary vector model
# 7--> Weighted vectro model
def logLoss(model):
    crossEntropy = 0
    p = 0
    for i in range(1,len(lines)):
        label = float(lines[i][5])
        p = lines[i][model]
        crossEntropy = -(label*numpy.log(p+0.001)+(1-label)*numpy.log(1-p+0.001))
        if crossEntropy<0.001:
            crossEntropy = 0
        lines[i].append(crossEntropy)


lines = []

# Csv file processing
with open("train_original.csv", encoding="utf8") as f:
    reader = csv.reader(f, delimiter=",")
    c = 0
    for i, line in enumerate(reader):
        lines.append(line)
        if i>4000:
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
    # Calculate weighted similarity
    wsim = weightedVector(s1, s2)
    lines[i].append(wsim)



# Evaluate cross entropy for binary vectors
logLoss(6)
# Evaluate cross entropy for weighted vectors
logLoss(7)


for i in range(1,len(lines)):
    print("id:{} \n question1: {} \n question2: {} \n label:{} | binary vector sim: {} | "
          "cross-entropy loss(bv): {} | \n \t\t  weighted vector sim: {} | cross-entropy loss(wv): {}\n".format(i-1, lines[i][3], lines[i][4], lines[i][5], lines[i][6], lines[i][8],
                                                 lines[i][7], lines[i][9]))


b = 0
w = 0
for i in range(1,len(lines)):
    b = b + lines[i][8]
    w = w + lines[i][9]

b = b/(len(lines)-1)
w = w/(len(lines)-1)
print()
print("Mean cross entropy loss for binary model: {}".format(b))
print("Mean cross entropy loss for weighted model: {}".format(w))
