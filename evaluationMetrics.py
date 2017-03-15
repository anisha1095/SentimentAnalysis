from decimal import *

def accuracy(size_test,test_y,test_pred):
    count=0
    for i in range(0,size_test):
      if test_pred[i]==test_y[i]:
         count=count+1

    accuracy= (Decimal(count)/Decimal(size_test))*100
    return accuracy

def otherMetrics(test_y,test_pred,size_test):
    truePos=0
    trueNeg=0
    falsePos=0
    falseNeg=0
    for i in range(0,size_test):
        if test_pred[i]=='positive' and test_y[i]=='positive':
            truePos=truePos+1
        if test_pred[i]=='negative' and test_y[i]=='negative':
            trueNeg=trueNeg+1
        if test_pred[i]=='positive' and test_y[i]=='negative':
            falsePos=falsePos+1
        if test_pred[i]=='negative' and test_y[i]=='positive':
            falseNeg=falseNeg+1
    temp=truePos+falsePos
    precision=(Decimal(truePos)/Decimal(temp))*100
    temp=truePos+falseNeg
    recall=(Decimal(truePos)/Decimal(temp))*100
    fscore=(Decimal(2*(precision*recall))/Decimal(precision+recall))
    return precision,recall,fscore