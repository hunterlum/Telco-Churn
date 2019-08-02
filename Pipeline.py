#!/usr/bin/env python
# coding: utf-8

# # Set Up

# In[1]:


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV

def obj_to_float(x):
    try:
        return(float(x))
    except:
        return(None)

cust_info=pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv').drop(['customerID'],axis=1)

churn=cust_info['Churn']
cust_info=cust_info.drop('Churn',axis=1)
#churn is y, cust_info is X
#churn=LabelEncoder().fit_transform(churn)
cust_info['TotalCharges']=cust_info['TotalCharges'].apply(obj_to_float)

cust_info.head()


# In[2]:


cat_feat=['gender','Partner','Dependents'
          ,'PhoneService','MultipleLines','InternetService'
          ,'OnlineSecurity','OnlineBackup','DeviceProtection'
          ,'TechSupport','StreamingTV','StreamingMovies'
          ,'Contract','PaperlessBilling','PaymentMethod']


num_feat=['TotalCharges','SeniorCitizen','MonthlyCharges'
          ,'tenure']

#columns are eitehr caterforical or numerical features
assert(set(cust_info.columns)==set(num_feat).union(set(cat_feat)))


# # Models

# ## Numerical Features

# In[3]:


feat_train, feat_test, churn_train, churn_test = train_test_split(
    cust_info[num_feat], churn, test_size=0.2)

num_clf=Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler()),
    ('classifier',LogisticRegression(solver='lbfgs'))
])

num_clf.fit(feat_train,churn_train)
print('Numerical Features: ')
print('Train score: %0.5f' %num_clf.score(feat_train,churn_train))
print('Test score: %0.5f' %num_clf.score(feat_test,churn_test))
print()


# In[4]:


#num_clf.named_steps


# ## Categorical Features

# In[5]:


feat_train, feat_test, churn_train, churn_test = train_test_split(
    cust_info[cat_feat], churn, test_size=0.2)

cat_clf=Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder()),
    ('classifier',LogisticRegression(solver='lbfgs'))
])

cat_clf.fit(feat_train,churn_train)
print('Categorical Features:')
print('Train score: %0.5f' %cat_clf.score(feat_train,churn_train))
print('Test score: %0.5f' %cat_clf.score(feat_test,churn_test))
print()


# ## Combining Features

# ## Preprocessing

# In[6]:


feat_train, feat_test, churn_train, churn_test = train_test_split(
    cust_info, churn, test_size=0.2)

num_transformer=Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler()),
])
cat_transformer=Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder()),
])

'''
pproc = preprocessor
ColumnTransformer is used to process 
cat and num columns differently
look below for source
'''
pproc=ColumnTransformer([
    ('numerical',num_transformer,num_feat),
    ('categorical',cat_transformer,cat_feat)
])


# ## Training and Model Selection

# In[7]:


lr_clf=Pipeline([
    ('preprocessor',pproc),
    ('decomposition',PCA()),
    ('classifier',LogisticRegression(solver='lbfgs'))
])

params=[
    {
        'preprocessor__categorical__encoder':[OrdinalEncoder(),OneHotEncoder()],
        'decomposition__n_components':list(np.linspace(0.5,1-1e-5,5)),
        'classifier' : [LogisticRegression(solver='lbfgs',max_iter=1e5)],
        'classifier__penalty' : ['l1', 'l2'],
        'classifier__solver' : ['liblinear'],
        'classifier__C': list(np.linspace(0.5,1-1e-5,5))
    },
    {
        'preprocessor__categorical__encoder':[OrdinalEncoder(),OneHotEncoder()],
        'decomposition__n_components':list(np.linspace(0.5,1-1e-5,5)),
        'classifier' : [RandomForestClassifier()],
        'classifier__n_estimators' : list(range(10,251,2)),
        'classifier__max_features':["auto",'sqrt','log2',None]+list(np.linspace(0.1,1-1e-5,5))
    }
]

cv_num=3
grid=GridSearchCV(lr_clf,params,cv=cv_num,n_jobs=-1,verbose=1)

grid.fit(feat_train,churn_train)
print('Best: ')
print('Train score: %0.5f' %grid.score(feat_train,churn_train))
print('Test score: %0.5f' %grid.score(feat_test,churn_test))
print()


# In[ ]:


grid.best_params_


# Source demonstrating how to combine features
# 
# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py
# 
# https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696
# 

# # Saving

# In[ ]:


from joblib import dump
dump(grid,'data/clf_cv'+str(cv_num)+'.joblib')
print('Model saved! Goodbye.')

