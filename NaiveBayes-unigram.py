import nltk
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from decimal import *
from stemming.porter2 import stem
import evaluationMetrics


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features[:4000]

def extract_features(document):
    document_words = [stem(word) for word in document]
    features = {}
    for word in word_features:
      features[word] = (word in document_words)
    return features


#training = '/Users/anishajauhari/Documents/twitter/twitter_dataset/training.csv'
#test='/Users/anishajauhari/Documents/twitter/twitter_dataset/test.csv'
training = '/Users/anishajauhari/PycharmProjects/sentimentalAnalysis/twitter/training.csv'
test='/Users/anishajauhari/PycharmProjects/sentimentalAnalysis/twitter/test.csv'
positive=[]
negative=[]
pt=[]
nt=[]
positive_y=[]
negative_y=[]

with open(training) as f:
    reader=csv.reader(f,delimiter=',')
    for row in reader:
        val=row[0]
        if val=='1':
            positive.append(row[1])
            positive_y.append(row[0])
        else:
            negative.append(row[1])
            negative_y.append(row[0])

stop=stopwords.words('english')


all=[]
tweets = []

for w in positive:
  words_filtered = [stem(e.lower()) for e in w.split() if len(e)>4 ]
  #words_filtered=st.stem(words_filtered)
  tweets.append( [words_filtered,'positive'])
  all.append(words_filtered)


for w in negative:
  words_filtered = [stem(e.lower()) for e in w.split() if len(e)>4 ]
 # words_filtered=st.stem(words_filtered)
  tweets.append( [words_filtered,'negative'])
  all.append(words_filtered)


#features
word_features = get_word_features(get_words_in_tweets(tweets))

#test set
test_y=[]
with open(test) as f:
    reader=csv.reader(f,delimiter=',')
    for row in reader:
        val=row[0]
        if val=='1':
            pt.append(row[1])
            test_y.append('positive')
        else:
            nt.append(row[1])
            test_y.append('negative')

##saving tweets into a file



#######   NAIVE BAYES ALGORITHM    ############

training_set = nltk.classify.apply_features(extract_features, tweets)   #returns lists of objects whose values are equal to "positive" and another list whose values are equal to "negative".

classifier = nltk.NaiveBayesClassifier.train(training_set)

words={}
test_pred=[]
for e in pt+nt:
  test_sentence=word_tokenize(e.lower())
  for w in test_sentence:
     w=stem(w)
     words[w]=w in all
  test_pred.append( classifier.classify(words))
  words={}

###### ACCURACY & MOST INFORMATIVE FEATURES ########

size_test=len(test_y)
print "Accuracy:", evaluationMetrics.accuracy(size_test,test_y,test_pred)
precision,recall,fscore = evaluationMetrics.otherMetrics(test_y,test_pred,size_test)
print "Precision:",precision
print "Recall: ",recall
print "Fscore: ", fscore
classifier.show_most_informative_features(10)