import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



training = '/Users/anishajauhari/PycharmProjects/sentimentalAnalysis/twitter/training.csv'
test='/Users/anishajauhari/PycharmProjects/sentimentalAnalysis/twitter/test.csv'

def vectorize(all,all_y):
    y=all_y
                                                                                                 #### uni-gram with tf-idf vectorization###
    #vectorizer=TfidfVectorizer(min_df=1)
                                                                                                #### bi-gram ###
    #vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
                                                                                                #### n-gram ####
    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 3), min_df=1)
    X=vectorizer.fit_transform(all)
    return X,y

def extract(dir):
    list=[]
    list_y=[]
    with open(dir) as f:
     reader=csv.reader(f,delimiter=',')
     for row in reader:
        val=row[0]
        if val=='1':
            list.append(row[1])
            list_y.append(row[0])
        else:
            list.append(row[1])
            list_y.append(row[0])
    return list,list_y


train1, train_y1= extract(training)
test,test_y=extract(test)

train =train1 +test
train_y = train_y1 +test_y

X,y=vectorize(train,train_y)

X_train = X[:5000]
y_train = y[:5000]

X_test= X[5001:7500]
y_test =y[5001:7500]

gnb = GaussianNB()
classifier =gnb.fit(X_train.toarray(), y_train)
#y_pred = classifier.predict(X_test)
print(classifier.score(X_test.toarray(), y_test)*100)
#print(classification_report(y_test, y_pred))