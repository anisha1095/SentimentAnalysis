import nltk
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from decimal import *
from stemming.porter2 import stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def vectorize(all,all_y):
    y=all_y
                                                                                                 #### uni-gram with tf-idf vectorization###
    vectorizer=TfidfVectorizer(min_df=1)
                                                                                                #### bi-gram ###
    #vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
                                                                                                #### n-gram ####
    #vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 3), min_df=1)
    X=vectorizer.fit_transform(all)
    print X
    return X,y




training = '/Users/anishajauhari/PycharmProjects/sentimentalAnalysis/twitter/training.csv'
test='/Users/anishajauhari/PycharmProjects/sentimentalAnalysis/twitter/test.csv'
all=[]
all_y=[]

with open(training) as f:
    reader=csv.reader(f,delimiter=',')
    for row in reader:
        val=row[0]
        if val=='1':
            all.append(row[1])
            all_y.append(row[0])
        else:
            all.append(row[1])
            all_y.append(row[0])

#test set
test1=[]
test_y=[]
with open(test) as f:
    reader=csv.reader(f,delimiter=',')
    for row in reader:
        val=row[0]
        if val=='1':
            test1.append(row[1])
            test_y.append(row[0])
        else:
            test1.append(row[1])
            test_y.append(row[0])

for i in test1:
    all.append(i)
for i in test_y:
    all_y.append(i)
# print all
X,y=vectorize(all,all_y)
#print X
#X_test,y_test=vectorize(test1,test_y)

# X_train=X[:9000]
# X_test=X[9001:12000]
# y_train=y[:9000]
# y_test =y[9001:12000]
# svm=SVC(kernel='linear')
#
# classifier= svm.fit(X_train,y_train)
#
# pred = classifier.predict(X_test)
# print(classification_report(y_test, pred))
#print(classifier.score(X_test, y_test)*100)

