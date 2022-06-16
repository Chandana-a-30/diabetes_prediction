#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction

# ## Importing the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing the data and reading the data

# In[2]:


df = pd.read_csv('diabetes.csv')
df.head()


# ## EDA of Data

# In[3]:


df.keys()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isna().sum()


# In[8]:


df.corr()


# ### Correlation Matrix

# In[9]:


plt.figure(figsize=[8,8])
sns.heatmap(df.corr(),annot=True, cmap='Blues', linecolor='Green', linewidths=1.5)
plt.show()


# ### Pairplot of data

# In[10]:


sns.pairplot(df)


# ### Count plot specifying the number of people suffering by diabetes

# In[11]:


sns.countplot(df['Outcome'])
plt.show()


# ## Machine Learning Algorithms part

# ### Separating the data into features and target data

# ### K Nearest Neighbors Classifier Model

# In[12]:


X = df.iloc[:,0 :-1]
y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=63)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.predict(X_test)
score = knn.score(X_train,y_train)
score1 = knn.score(X_test,y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is:",score1)


# ### Logistic Regression Model

# In[13]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=10)
logreg.fit(X_train,y_train)
logreg.predict(X_test)
score = logreg.score(X_train,y_train)
score1 = logreg.score(X_test,y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is",score1)


# ### Decision Tree Classifier

# In[14]:


from sklearn.tree import DecisionTreeClassifier
tre = DecisionTreeClassifier()
tre.fit(X_train,y_train)
tre.predict(X_test)
score= tre.score(X_train,y_train)
score1 =tre.score(X_test,y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is:",score1)


# #### Features Importance Bar Plot

# In[15]:


featur_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Diabetics','Age' ]
features = tre.feature_importances_
features


# In[16]:


plt.figure(facecolor='r')
plt.barh(featur_names,features)
plt.show()


# ### Random Forest Classifier

# In[17]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(X_train,y_train)
forest.predict(X_test)
score = forest.score(X_train,y_train)
score1 = forest.score(X_test,y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is:",score1)


# #### Tuning the parameters of the Model to get some improved results

# In[18]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=0)
forest.fit(X_train,y_train)
forest.predict(X_test)
score = forest.score(X_train,y_train)
score1 = forest.score(X_test,y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is:",score1)


# #### Features Importance Bar Plot

# In[19]:


forest1 = forest.feature_importances_
forest1


# In[20]:


plt.figure(facecolor='y')
plt.barh(featur_names, forest1)
plt.show()

