#!/usr/bin/env python
# coding: utf-8

# ## Cyber Security Case study

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  


# In[2]:


Attack_Back = pd.read_csv('C:/Analytics/Analytics_T_2.2/CASE_STUDIES/12. Capstone Case Study - Cyber Security Case Study/Data_of_Attack_Back.csv')
Attack_Back_BufferOverflow = pd.read_csv('C:/Analytics/Analytics_T_2.2/CASE_STUDIES/12. Capstone Case Study - Cyber Security Case Study/Data_of_Attack_Back_BufferOverflow.csv')
Attack_Back_FTPWrite = pd.read_csv('C:/Analytics/Analytics_T_2.2/CASE_STUDIES/12. Capstone Case Study - Cyber Security Case Study/Data_of_Attack_Back_FTPWrite.csv' , header= None)
Attack_Back_GuessPassword = pd.read_csv('C:/Analytics/Analytics_T_2.2/CASE_STUDIES/12. Capstone Case Study - Cyber Security Case Study/Data_of_Attack_Back_GuessPassword.csv')
Attack_Back_Neptune = pd.read_csv('C:/Analytics/Analytics_T_2.2/CASE_STUDIES/12. Capstone Case Study - Cyber Security Case Study/Data_of_Attack_Back_Neptune.csv')
Attack_Back_NMap = pd.read_csv('C:/Analytics/Analytics_T_2.2/CASE_STUDIES/12. Capstone Case Study - Cyber Security Case Study/Data_of_Attack_Back_NMap.csv')
Attack_Back_Normal = pd.read_csv('C:/Analytics/Analytics_T_2.2/CASE_STUDIES/12. Capstone Case Study - Cyber Security Case Study/Data_of_Attack_Back_Normal.csv')
Attack_Back_PortSweep = pd.read_csv('C:/Analytics/Analytics_T_2.2/CASE_STUDIES/12. Capstone Case Study - Cyber Security Case Study/Data_of_Attack_Back_PortSweep.csv')
Attack_Back_RootKit = pd.read_csv('C:/Analytics/Analytics_T_2.2/CASE_STUDIES/12. Capstone Case Study - Cyber Security Case Study/Data_of_Attack_Back_RootKit.csv')
Attack_Back_Satan = pd.read_csv('C:/Analytics/Analytics_T_2.2/CASE_STUDIES/12. Capstone Case Study - Cyber Security Case Study/Data_of_Attack_Back_Satan.csv')
Attack_Back_Smurf = pd.read_csv('C:/Analytics/Analytics_T_2.2/CASE_STUDIES/12. Capstone Case Study - Cyber Security Case Study/Data_of_Attack_Back_Smurf.csv')


# In[3]:


Attack_Back.head(2)


# In[4]:


Attack_Back_BufferOverflow.head(2)


# In[5]:


Attack_Back_FTPWrite.head(2)


# In[6]:


#adding columns name in the Attack_Back_FTPWrite data as no cols name was there 
Attack_Back_FTPWrite.columns = Attack_Back.columns


# In[7]:


Attack_Back_FTPWrite.head(2)


# In[8]:


Attack_Back_GuessPassword.head(2)


# In[9]:


Attack_Back_Neptune.head(2)


# In[10]:


Attack_Back_NMap.head(2)


# In[11]:


Attack_Back_Normal.head(2)


# In[12]:


Attack_Back_PortSweep.head(2)


# In[13]:


Attack_Back_RootKit.head(2)


# In[14]:


Attack_Back_Satan.head(2)


# In[15]:


Attack_Back_Smurf.head(2)


# ### Combining dataframes 

# In[16]:


#dataframes 
Dataframes = [Attack_Back , Attack_Back_BufferOverflow , Attack_Back_FTPWrite , Attack_Back_GuessPassword ,
              Attack_Back_Neptune , Attack_Back_NMap , Attack_Back_Normal , Attack_Back_PortSweep , Attack_Back_RootKit
             , Attack_Back_Satan , Attack_Back_Smurf]

# List of labels corresponding to each DataFrame
Attacks = ['Back', 'BufferOverflow', 'FTPWrite', 'GuessPassWord', 'Neptune', 
                          'NMap', 'Normal', 'PortSweep', 'RootKit', 'Satan', 'Smurf']

for df , Attack in zip(Dataframes , Attacks ): # adding labels column for each df
    df["Attack"] = Attack

#Combining all in single dfs 
Final_df = pd.concat(Dataframes , ignore_index = True)


# In[17]:


Final_df


# In[18]:


#Checking data types 
Final_df.info()


# In[19]:


#Checking for missing values
Final_df.isna().sum()


# ### Data preprocessing 

# In[20]:


from sklearn.model_selection import train_test_split 


# In[21]:


# Split data into features and target
X = Final_df[Final_df.columns.difference(['Attack'])] #Feature columns
y = Final_df['Attack']


# In[22]:


X_train , X_test , y_train , y_test = train_test_split( X , y , test_size=0.3 , random_state=10)


# In[23]:


X_train.shape ,X_test.shape , y_train.shape , y_test.shape


# ### Binomial classification: Detect anomalies by predicting Activity is normal or attack

# In[24]:


Final_df['Attack'].unique()


# In[25]:


from sklearn.ensemble import RandomForestClassifier


# In[26]:


#Creating a Binary Target varaiable 
df = Final_df.drop(columns="Attack")

df['Binary_label']= np.where(Final_df["Attack"]=="Normal" , 0 , 1)


# In[27]:


# Split data into features and target

X = df[df.columns.difference(['Binary_label'])] #Feature columns
y = df['Binary_label']

X_train , X_test , y_train , y_test = train_test_split( X , y , test_size=0.3 , random_state=10)


# In[28]:


#Training the Model

Rf_clf = RandomForestClassifier()
Rf_clf.fit(X_train , y_train)


# In[29]:


train_pred = Rf_clf.predict(X_train)
test_pred = Rf_clf.predict(X_test)


# In[30]:


from sklearn.metrics import accuracy_score , confusion_matrix , classification_report


# In[31]:


Train_acc = accuracy_score(y_train,train_pred)
Test_acc = accuracy_score(y_test , test_pred)

print(f"Train Accuracy :: {Train_acc}      Test Accuracy:: {Test_acc}")   


# In[32]:


# Print the classification report to see precision, recall, and F1-score

print("classification_report for Train_data")
print(classification_report(y_train,train_pred))

print("classification_report for Test_data")
print(classification_report(y_test,test_pred))


print("The model achieves perfect precision, recall, and F1-score on both the training and test datasets, indicating that it's able to classify instances of both classes ( Normal(0)  and Attack(1) ) with high accuracy.")


# ### Multinomial Classification: Detecting type of activity by predicting Activity is Normal or Back or Buffer Over flow or FTP Write or Guess Password or Neptune or N-Map or Port Sweep or Root Kit or Satan or Smurf
# 

# In[33]:


Final_df


# In[34]:


# Split data into features and target
X = Final_df[Final_df.columns.difference(['Attack'])] #Feature columns
y = Final_df['Attack']

#splitting the data in train & test 
train_X , test_X , train_y , test_y = train_test_split( X , y , test_size=0.3 , random_state=10)


# In[35]:


from sklearn.linear_model import LogisticRegression


# In[36]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV ,  RandomizedSearchCV

DT_clf = DecisionTreeClassifier()


# In[37]:


params = {'criterion':['gini' , 'entropy'],
    'max_depth':[17 , 18 , 19,20],
    'min_samples_split':[8, 12 , 10 ,15],
    'min_samples_leaf':[3,4,5,6]}


# In[38]:


Dt_model = RandomizedSearchCV(DT_clf , param_distributions=params , cv=3 , random_state=10 )


# In[39]:


Dt_model.fit(X_train , y_train)


# In[40]:


Dt_model.best_params_


# In[41]:


Best_model = Dt_model.best_estimator_


# In[42]:


Best_model.fit(X_train , y_train)


# In[43]:


train_preds = Best_model.predict(train_X)
test_preds = Best_model.predict(test_X)


train_acc = accuracy_score(train_preds, train_y)
test_acc = accuracy_score(test_preds, test_y)

print(f"Train Accuracy:: {train_acc:.4f} \n Test Accuracy:: {test_acc:.4f}")




# In[45]:


# Print the classification report to see precision, recall, and F1-score

print("classification_report for Train_data")
print(classification_report(y_train,train_preds))

print("classification_report for Test_data")
print(classification_report(y_test,test_preds))

print("The model achieved almost 100% accuracy on the both train & test data.Most classes have high precision, recall, and F1-score, indicating good performance.")


# In[ ]:




