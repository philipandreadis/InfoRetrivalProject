import numpy
import csv
from random import randint
import math
from collections import Counter

m = 1000 # number of pairs to be tested

# Calculates the weight of each term of the query existing in a question based on tf*idf model
def tf_idf(word,question):
    most_common1, num_most_common1 = Counter(qwords).most_common(1)[0]
    tf = qwords.count(word)/num_most_common1
    nt = 0
    for i in range(len(questions)):
        if word in questions[i]:
            nt = nt + 1
    idf = numpy.log(len(questions)/nt)
    return tf*idf

lines = []
# Csv file processing
with open("train_original.csv", encoding="utf8") as f:
    reader = csv.reader(f, delimiter=",")
    c = 0
    for i, line in enumerate(reader):
        lines.append(line)
        if i > m:
            break

# Create Inverted Index
iIndex = dict()
questions = []
for i in range(1,len(lines)):
    s1 = lines[i][3]
    s2 = lines[i][4]
    # string preprocessing
    s1 = s1.lower()
    s2 = s2.lower()
    s1 = s1.strip('?')
    s2 = s2.strip('?')
    questions.append(s1)
    questions.append(s2)

for i in range(len(questions)):
    words = questions[i].split()
    for j in range(len(words)):
        if not(words[j] in iIndex):
            iIndex[words[j]] = [i]
        else:
            if not(i in iIndex[words[j]]):
                iIndex[words[j]].append(i)

# List of accumulators
S = dict()
# Random question from the train set as the query
q = questions[randint(0,len(questions)-1)]
print("Query question is: {}\n".format(q))
qwords = q.split()

for i in qwords:
    if i in iIndex:
        for j in iIndex[i]:
            if j in S:
                S[j] = S[j] + tf_idf(i,j)
            else:
                S[j] = tf_idf(i,j)
# Print top-3 questions that refer to the same subject as the query question
print("Top-3 similar questions are:")
d = Counter(S)
d.most_common()
for k, v in d.most_common(3):
    print ('Question number {}: {}, with accumulator of {}'.format(k+1,questions[k] ,v))