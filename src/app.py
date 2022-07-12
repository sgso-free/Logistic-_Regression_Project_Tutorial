import pandas as pd
import numpy as numpy
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler


#load data
url = 'https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv'
df=pd.read_csv(url, sep=";")

#have duplicate, drop
df.drop_duplicates(keep='first', inplace=True, ignore_index=False)

#make change for all values, that know
#more frecuency value is the mode
variables = df.columns[df.dtypes == 'object']

#print('Change unknow in each object column')
for v in variables:
    #print(f' {v} to {df[v].mode()[0]}')
    df.loc[df[v] == "unknown", v] = df[v].mode()[0]


#drop outliers, only drop the last value 56
df.drop(df[df['campaign']>45].index, inplace=True)

#basic.9y','basic.6y','basic4y' into 'middle_school'
cleanup_nums = {"marital":     {"married": 1, "single": 2, 'divorced':3},
               "education": {"basic.4y": 1, "high.school": 2, "basic.6y": 1, "basic.9y": 1, "professional.course": 2, "university.degree": 3, "illiterate":4 }, 
               "default": {'no':0,'yes':1}, "housing": {'no':0,'yes':1}, "loan": {'no':0,'yes':1},
               "contact": {'telephone':1,'cellular':2}, 
               "month": {'may':5, 'jun':6, 'jul':7, 'aug':8, 'oct':10, 'nov':11, 'dec':12, 'mar':3, 'apr':4,'sep':9},
               "day_of_week": {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5},
               "poutcome" : {'nonexistent':1, 'failure':2, 'success':3},
               "job" : {'housemaid':1, 'services':2, 'admin.':3, 'blue-collar':4, 'technician':5, 'retired':6, 'management':7, 'unemployed':8, 'self-employed':9, 'entrepreneur':10, 'student':11},
               "y": {'no':0,'yes':1},
               }
#encoding the feature
df_transf = df.replace(cleanup_nums)

#create the group of age
df_transf['age'] = pd.cut(df_transf['age'],bins=[10,20,30,40,50,60,70,80,90,100],labels=[1,2,3,4,5,6,7,8,9])

df= df_transf.copy()

# split dataset into x,y
X = df.drop('y',axis=1)
y = df['y']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)

scaler = MinMaxScaler() #saga solver requires features to be scaled for model conversion
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# define class weights
w = {0: 1, 1: 10}

# define model
lg2 = LogisticRegression(random_state=13, class_weight=w, solver='lbfgs', max_iter=300)

# fit it
lg2.fit(X_train,y_train)

# test
y_pred = lg2.predict(X_test)

# performance
print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
print(f'Recall score: {recall_score(y_test,y_pred)}')

#save the model to file
filename = 'models/finalized_model.sav' #use absolute path
pickle.dump(lg2, open(filename, 'wb'))

#use the model save with new data to predicts prima

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

#Predict using the model 
#predigo el target y para los valores seteados, selecciono cualquiera para ver
print('Predicted ] : \n', loaded_model.predict(X_test[13:17]))
print('Class ] : \n', y_test[13:17])