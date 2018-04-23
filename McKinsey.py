#McKinsey Challenge 14th April'18

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
train = pd.read_csv('D://Analytics Vidya//McKinsey Challenge//train.csv')
test = pd.read_csv('D://Analytics Vidya//McKinsey Challenge//test.csv')

#collecting id's of test data
test_id = test['id'].to_frame()

#Describing data
train.head(5)
train.describe()

print("Missing values in train data")
for col in train.columns:
    print('No. of null values in ' + col + ': '+
         str(train[pd.isnull(train[col])].shape[0]))

print('Missing values in test data:')
for col in test.columns:
    print('No. of null values in ' + col + ': '+
         str(test[pd.isnull(test[col])].shape[0]))

train['bmi'].isnull().sum()

per_mis = (train.isnull().sum()/len(train))*100
ratio_mis = per_mis.sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :ratio_mis})
missing_data

#Drop Variables
#train.drop('smoking_status', 1, inplace= True)
#test.drop('smoking_status', 1, inplace= True)

#Filling Null Values
train['bmi'].fillna(train['bmi'].median(), inplace=True)
test['bmi'].fillna(train['bmi'].median(), inplace=True)
train['smoking_status'] = train['smoking_status'].fillna('never_smoked')
test['smoking_status'] = train['smoking_status'].fillna('never_smoked')

#make dummies of categorical variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)

#function for normalized gini score
#train.to_csv('train1.csv', index=False)
#test.to_csv('test1.csv', index=False)
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(mod,X_test,y_true):
    a = y_true    
    #if prob == False:   
    p = mod.predict(X_test)
    #else:
    #    p = mod.predict_proba(X_test)
    return gini(a, p) / gini(a, a)

def gini_normalized_proba(mod,X_test,y_true):
    a = y_true    
    #if prob == False:   
    p = mod.predict_proba(X_test)
    #else:
    #    p = mod.predict_proba(X_test)    
    return gini(a, p[:,1]) / gini(a, a)

#split data set into train and validation and define X and y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,train.columns!='stroke'],
                                                    train['stroke'], test_size=0.33, random_state=42)

#Import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC                
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
#from xgboost import XGBModel
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score

classifers = [LinearDiscriminantAnalysis(),LogisticRegressionCV(),RandomForestClassifier(),
              GradientBoostingClassifier(),DecisionTreeClassifier()]

#train and print scores for all models
#Note:- later xgboost started making some problem in my PC so skipped that part
print('starting training...')  
df = pd.DataFrame(columns = ['clf_name','test_score','train_score'])     
#learn_rate= [.001, .01,.1,.2, .5]
#n_estimators = [10,50, 100,200,500]
for clf in classifers:
    clf.fit(X_train,y_train)
    pred = clf.predict_proba(X_test)
    test_score, train_score = [gini_normalized(clf,X_test,y_test),gini_normalized(clf, X_train,y_train)]
    print(train_score, test_score)
    df.append([[test_score,train_score]])


#ROC_AUC score
for clf in classifers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(roc_auc_score(y_test, y_pred))


#GBM is shwing quite good result
#used GBM to predict probablity of strokes on test data
clf = GradientBoostingClassifier().fit(X_train, y_train)
test['stroke'] = clf.predict_proba(test)[:,1]

test = pd.concat([test], axis=1, join='inner')
test.to_csv('Submission9.csv', columns = ['id', 'stroke'], index = False)

