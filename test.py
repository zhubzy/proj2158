
import gzip
from collections import defaultdict
import math
import operator
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import scipy
import pandas as pd
import numpy as np
class Utilites:
    def readGz(path):
        for l in gzip.open(path, 'rt'):
            yield eval(l)

    def punctuations():
        return set(string.punctuation)

    def readCSV(path):
        f = gzip.open(path, 'rt')
        f.readline()
        for l in f:
            u, b, r = l.strip().split(',')
            r = int(r)
            yield u, b, r

    def Cosine(x1, x2):
        numer = 0
        norm1 = 0
        norm2 = 0
        for a1, a2 in zip(x1, x2):
            numer += a1*a2
            norm1 += a1**2
            norm2 += a2**2
        if norm1*norm2:
            return numer / math.sqrt(norm1*norm2)
        return 0

    def featureTFIDF(dataset, rev, punctuation, words, df):
        tf = defaultdict(int)
        r = ''.join([c for c in rev['review_text'].lower()
                    if not c in punctuation])
        for w in r.split():
            # Note = rather than +=, different versions of tf could be used instead
            tf[w] = 1
        tfidf = dict(
            zip(words, [tf[w] * math.log2(len(dataset) / df[w]) for w in words]))
        maxTf = [(tf[w], w) for w in words]
        maxTf.sort(reverse=True)
        maxTfIdf = [(tfidf[w], w) for w in words]
        maxTfIdf.sort(reverse=True)
        return tfidf, maxTfIdf

    def frequencyCalc(data, size=1000, limit=5):
        wordCount = defaultdict(int)
        punctuation = set(string.punctuation)
        my_stopwords = nltk.corpus.stopwords.words('english')
        for d in data:
            r = ''.join([c for c in d['review_text'].lower()
                        if not c in punctuation])
            for w in r.split():
                if w not in my_stopwords and len(w) >= 2:
                    wordCount[w] += 1

        wordFrequency = [(wordCount[w], w) for w in wordCount]
        wordFrequency.sort()
        wordFrequency.reverse()
        wordFrequency = [(x[0], x[1]) for x in wordFrequency if x[0] >= limit]
        words = [x[1] for x in wordFrequency[:size]]
        return words, wordFrequency
users_review_data = []
reviews_data = []
users_items_data = []
items_data = []
bundle_data = []

for d in Utilites.readGz("user_reviews.json.gz"):
    users_review_data.append(d)

#for d in Utilites.readGz("/work/steam_reviews.json.gz"):
    #reviews_data.append(d)

for d in Utilites.readGz("users_items.json.gz"):
    users_items_data.append(d)

for d in Utilites.readGz("steam_games.json.gz"):
    items_data.append(d)

for d in Utilites.readGz("bundle_data.json.gz"):
    bundle_data.append(d)

users_items_train = []
users_items_validation = []
users_items_test = []
users_per_item_train = defaultdict(set)
items_per_user_train = defaultdict(set)
times_per_game = defaultdict()


for d in users_items_data:
    games_played_by_user = d['items']
    random.shuffle(games_played_by_user)

    for i in range(len(games_played_by_user)):
        if(i < len(games_played_by_user) * 0.6):
            users_items_train.append((d['user_id'], games_played_by_user[i]))
            items_per_user_train[d['user_id']].add((games_played_by_user[i]['item_id'],games_played_by_user[i]['playtime_forever'],games_played_by_user[i]['playtime_2weeks']))
            users_per_item_train[games_played_by_user[i]['item_id']].add((d['user_id'],games_played_by_user[i]['playtime_forever'],games_played_by_user[i]['playtime_2weeks']))
            if(games_played_by_user[i]['item_id'] in times_per_game ):
                times_per_game[games_played_by_user[i]['item_id']][0] += games_played_by_user[i]['playtime_forever']
                times_per_game[games_played_by_user[i]['item_id']][1] += 1
            else:
                times_per_game[games_played_by_user[i]['item_id']] = [games_played_by_user[i]['playtime_forever'],1]
        elif(i < len(games_played_by_user) * 0.8):
            users_items_validation.append((d['user_id'], games_played_by_user[i]))
        else:
            users_items_test.append((d['user_id'], games_played_by_user[i]))

def features(data):
    x =[]
    y = []
    for user,gameInfo in data:
        if(gameInfo['playtime_forever'] ==0): continue
        if(gameInfo['playtime_forever'] <= 45): y.append(0)
        elif(gameInfo['playtime_forever'] <= 220): y.append(1)
        elif(gameInfo['playtime_forever'] <= 840): y.append(2)
        else: y.append(3)
        x.append((user,gameInfo['item_id']))
    return x, y


trainX, trainY = features(users_items_train)
print("starting to train")
trainX_set = set(trainX)
y_pred = []
i = 0
for user,item in trainX_set:
    avg_time = times_per_game[item][0]/ times_per_game[item][1]
    if(avg_time <= 45): y_pred.append(0)
    elif(avg_time <= 220): y_pred.append(1)
    elif(avg_time<= 840): y_pred.append(2)
    else: y_pred.append(3)



correct = np.array(trainY) == np.array(y_pred)
print(sum(correct) / len(correct))