#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Network

# ### Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf


# ## Data Preprocessing

# ### Importing the dataset

# In[2]:


dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()


# In[3]:


X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


# In[4]:


print(X)


# In[5]:


print(y)


# ### Encoding categorical data

# Label Encoding the "Gender" column

# In[6]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])


# In[7]:


print(X)


# In[8]:


print(X[0])


# One Hot Encoding the "Geography" column

# In[9]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[10]:


#to escape from onehotencoding trap
X=X[:,1:]


# In[11]:


print(X)


# In[12]:


X[0]


# ### Feature Scaling

# In[13]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[14]:


print(X)


# ### Splitting the dataset into the Training set and Test set

# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Building the ANN

# In[16]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# ### Initializing the ANN

# In[17]:


ann = tf.keras.models.Sequential()


# ### Adding the input layer and the first hidden layer

# In[18]:


#input dim 11 output dim 6
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# In[19]:


#with dropout to reduce overfitting
ann.add(tf.keras.layers.Dropout(rate=0.1))


# ### Adding the second hidden layer

# In[20]:


#input dim 6 output dim 6
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# In[21]:


#with dropout to reduce overfitting
ann.add(tf.keras.layers.Dropout(rate=0.1))


# ### Adding the output layer

# In[22]:


#input dim 6 output dim 1
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ## Training the ANN

# ### Compiling the ANN

# In[23]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Training the ANN on the Training set

# In[24]:


ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


# ## Making the predictions and evaluating the model

# ### Predicting the Test set results

# In[25]:


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ### Making the Confusion Matrix

# In[26]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[27]:


from sklearn.metrics import accuracy_score
# Model Accuracy, how often is the classifier correct?
print('Accuracy of Artificial Neural Network on test set: {:.2f}%'.format(accuracy_score(y_test, y_pred)*100))


# ### Predicting a single new observation

# In[28]:


# """
# Predict if the customer with the following informations will leave the bank or not:
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40
# Tenure: 3
# Balance: 60000
# Number of Products: 2
# Has Credit Card: Yes
# Is Active Member: Yes
# Estimated Salary:50000
# """


# In[29]:


#dataset to predict changed to categorical values
new_dataset=np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])
#dataset scaled
new_dataset=sc.transform(new_dataset)
new_dataset


# In[30]:


new_prediction=ann.predict(new_dataset)
#to get ans in true or false
new_prediction=(new_prediction>0.5)


# In[31]:


print(new_prediction[0,0])


# # Alternative Approach- Evaluating the model

# In[32]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


# In[33]:


def build_classifier(optimizer='adam'):
    classifier = tf.keras.models.Sequential()
    classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))
    classifier.add(tf.keras.layers.Dropout(rate=0.1))    
    classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))
    classifier.add(tf.keras.layers.Dropout(rate=0.1))    
    classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


# In[34]:


#10 fold cross validation on dataset for better accuracy
classifier=KerasClassifier(build_fn=build_classifier, batch_size=10,nb_epoch=100)
accuracies=cross_val_score(estimator=classifier, X=X_train, y=y_train,cv=10,n_jobs=-1)


# In[35]:


print(accuracies)


# In[36]:


mean=accuracies.mean()
print(mean)


# In[37]:


variance=accuracies.std()
print(variance)


# ## Tuning the ANN

# In[38]:


from sklearn.model_selection import GridSearchCV


# In[39]:


classifier=KerasClassifier(build_fn=build_classifier)
parameters={'batch_size':[25,32],
           'nb_epoch':[100,500],
           'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10)
grid_search=grid_search.fit(X_train,y_train)
best_parameters=grid_search.best_params_
best_Accuracy=grid_search.best_score_


# In[40]:


print('Best Accuracy: {:.2f}%'.format(best_Accuracy*100))


# In[41]:


print("Best Parameters:",best_parameters)


# In[42]:


batch_size=best_parameters['batch_size']
nb_epoch=best_parameters['nb_epoch']
optimizer=best_parameters['optimizer']


# In[43]:


#using the above parameters to fit the training dataset

tuned_classifier=KerasClassifier(build_fn=build_classifier, batch_size=batch_size,nb_epoch=nb_epoch,optimizer=optimizer)
tuned_classifier.fit(X_train, y_train)


# In[44]:


#run the tuned model on test dataset 
y_pred = tuned_classifier.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[45]:


from sklearn.metrics import accuracy_score
# Model Accuracy, how often is the classifier correct?
print('Accuracy of Artificial Neural Network on test set: {:.2f}%'.format(accuracy_score(y_test, y_pred)*100))


# In[46]:


#predicting the likely of a customer to be associated with the bank

new_prediction=tuned_classifier.predict(new_dataset)
#to get ans in true or false
new_prediction=(new_prediction>0.5)
if(new_prediction[0,0]):
    print("Will Stay as a Customer with the Bank")
else:
    print("Will close the Account with the Bank")

