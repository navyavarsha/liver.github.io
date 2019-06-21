#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[63]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_dcb8ae55a59f4702aaae2d0857d8e00c = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='HX439E0yWD9DCN_NzzTjul0l_0yPtOBNCcBqcDZHWY47',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_dcb8ae55a59f4702aaae2d0857d8e00c.get_object(Bucket='liverpatientanalysis-donotdelete-pr-xbuijx5xvv0fk6',Key='indian_liver_patient.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset= pd.read_csv(body)
dataset


# In[3]:


dataset.corr()


# In[4]:


sns.heatmap(dataset.corr(),annot=True)


# In[5]:


dataset.isnull().any()


# In[6]:


dataset['Albumin_and_Globulin_Ratio'].fillna((dataset['Albumin_and_Globulin_Ratio'].mean()),inplace=True)


# In[7]:


dataset.isnull().any()


# In[8]:


x=dataset.iloc[:,0:10].values


# In[9]:


x.shape


# In[10]:


y=dataset.iloc[:,10:].values


# In[11]:


y


# In[12]:


from sklearn.preprocessing import LabelEncoder 
lb=LabelEncoder()


# In[13]:


x[:,1]=lb.fit_transform(x[:,1])


# In[14]:


x


# In[15]:


x.shape


# In[16]:


x.ndim


# In[17]:


y.ndim


# In[18]:


lb_y=LabelEncoder()
y=lb_y.fit_transform(y)
y


# In[19]:


y.ndim


# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[21]:


x_test


# In[22]:


x_train.shape


# # Logistic regression

# In[23]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()


# In[24]:


classifier.fit(x_train,y_train)


# In[25]:


y_predict=classifier.predict(x_test)
y_predict


# In[26]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[27]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)


# In[28]:


import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_predict)
roc_auc = metrics.auc(fpr, tpr)


# In[29]:


plt.title('patient analysis')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Decision Tree

# In[30]:


from sklearn.tree import DecisionTreeClassifier
classifier_dt=DecisionTreeClassifier(criterion='entropy',random_state=0)


# In[31]:


classifier_dt.fit(x_train,y_train)


# In[32]:


y_predict_dt=classifier_dt.predict(x_test)
y_predict_dt


# In[33]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict_dt)


# In[34]:


import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_predict_dt)
roc_auc_dt= metrics.auc(fpr, tpr)


# In[35]:


plt.title('patient analysis')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Random Forest

# In[36]:


from sklearn.ensemble import RandomForestClassifier
classifier_rf=RandomForestClassifier(n_estimators=30,criterion='entropy',random_state=0)


# In[37]:


classifier_rf.fit(x_train,y_train)


# In[38]:


y_predict_rf=classifier_rf.predict(x_test)
y_predict_rf


# In[39]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict_rf)


# In[40]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict_rf)
cm


# In[41]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(y_test,y_predict_rf)
roc_auc_rf=metrics.auc(fpr,tpr)
roc_auc_rf


# In[42]:


plt.plot(fpr,tpr,label='AUC=%0.2f'%roc_auc)
plt.legend()
plt.show()


# # KNN

# In[43]:


from sklearn.neighbors import KNeighborsClassifier
classifier_knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)


# In[44]:


classifier_knn.fit(x_train,y_train)


# In[45]:


y_predict_knn=classifier_knn.predict(x_test)
y_predict_knn


# In[46]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict_knn)


# In[47]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict_knn)
cm


# In[48]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(y_test,y_predict_knn)
roc_auc_knn=metrics.auc(fpr,tpr)
roc_auc_knn


# In[49]:


plt.plot(fpr,tpr,label='AUC=%0.2f'%roc_auc)
plt.legend()
plt.show()


# In[50]:


x=["LR","KNN","DT","RF"]
data=[roc_auc,roc_auc_knn,roc_auc_dt,roc_auc_rf]
import matplotlib.pyplot as plt
plt.title('AUC')
plt.bar(x, data)
plt.show()


# In[51]:


get_ipython().system('pip install watson-machine-learning-client --upgrade')


# In[52]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[53]:


wml_credentials={
    "access_key": "axYfapUYeHbTa9I0jzr_UQEn92BXEht3JzhVFk3cRgwE",
    "instance_id": "2c8705f0-ccc1-4a32-8b71-7ddf0892e503",
  "password": "ccc6fca3-384b-42d8-81c7-07d461d5c4b4",
  "url": "https://eu-gb.ml.cloud.ibm.com",
  "username": "2c4ccba9-1668-49a4-84d7-d6be77daaa2f"
}


# In[54]:


client = WatsonMachineLearningAPIClient(wml_credentials)
import json


# In[55]:


instance_details = client.service_instance.get_details()
print(json.dumps(instance_details, indent=2))


# In[56]:


model_props = {client.repository.ModelMetaNames.AUTHOR_NAME: "Navya", 
               client.repository.ModelMetaNames.AUTHOR_EMAIL: "navyavarsha99@gmail.com", 
               client.repository.ModelMetaNames.NAME: "Liver Patient Analysis"}


# In[57]:


model_artifact =client.repository.store_model(classifier_dt, meta_props=model_props)


# In[58]:


published_model_uid = client.repository.get_model_uid(model_artifact)


# In[59]:


published_model_uid


# In[60]:


created_deployment = client.deployments.create(published_model_uid, name="Liver Patient Analysis-Project")


# In[61]:


scoring_endpoint = client.deployments.get_scoring_url(created_deployment)
scoring_endpoint


# In[62]:


client.deployments.list()


# In[ ]:




