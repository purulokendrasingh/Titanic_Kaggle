
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn import svm


# In[2]:

df = pd.read_csv('titanic_dataset.txt')
df


# In[3]:

df.replace('NaN',-99999,inplace=True)
df = df.drop(['PassengerId','Name','Ticket','Fare','Cabin','Embarked'],axis=1)
X = np.array(df.drop(['Survived'],1))
y = np.array(df['Survived'])

for i in range(0,len(X)):
    if X[i][1] == 'male':
        X[i][1] = 1
    else:
        X[i][1] = 0

"""for i in range(0,len(X)):
    if X[i][5] == 'S':
        X[i][5] = 1
    elif X[i][5] == 'C':
        X[i][5] = 2
    else:
        X[i][5] = 0"""
        
X


# In[4]:

clf = svm.SVC()
clf.fit(X,y)


# In[5]:

df_test = pd.read_csv('test.txt')
df_test


# In[6]:

df_test.replace('NaN',-99999,inplace=True)
pass_id = df_test['PassengerId']
df_test = df_test.drop(['PassengerId','Name','Ticket','Fare','Cabin','Embarked'],axis=1)
X_test = np.array(df_test)

for i in range(0,len(X_test)):
    if X_test[i][1] == 'male':
        X_test[i][1] = 1
    else:
        X_test[i][1] = 0

"""for i in range(0,len(X_test)):
    if X_test[i][5] == 'S':
        X_test[i][5] = 1
    elif X_test[i][5] == 'C':
        X_test[i][5] = 2
    else:
        X_test[i][5] = 0"""
        
X_test


# In[7]:

pred = clf.predict(X_test)
print(pred)


# In[8]:

#pass_id.to_csv("submission.txt",index=False)
#with open('submission.txt'+'_2', 'wb') as abc:
#    np.savetxt(abc, pred, delimiter=" ")
pred_df = pd.DataFrame(pred)
pred_df
final_df = pd.concat([pass_id, pred_df],axis = 1)
final_df.to_csv('submission.txt',index=False)

