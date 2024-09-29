#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


heart_des=pd.read_csv('heart.csv')
heart_des


# In[ ]:


#now to create x 
x=heart_des.drop('target',axis=1)
x


# In[ ]:


#also to create y
y=heart_des['target']
y


# In[ ]:


x.head()


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[ ]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()  # Instantiate the classifier
params = clf.get_params()  # Get the parameters
print(params)


# In[ ]:


#fit  model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[ ]:


clf.fit(x_train,y_train)


# In[ ]:


#EVALATE Model
predicted_y=clf.predict(x_test)
predicted_y


# In[ ]:


clf.score(x_train,y_train)


# In[ ]:


clf.score(x_test,y_test)


# In[ ]:


#improve model
for i in range(10,200,10):
    print(f'Running model with {i} estimators')
    clf=RandomForestClassifier(n_estimators=i ).fit(x_train,y_train)
    print(f'Accuratcy is :{clf.score(x_test,y_test)}')


# In[ ]:


#save model
import pickle
pickle.dump(clf,open('heart disese predictor.pkl','wb'))


# In[ ]:


load_model=pickle.load(open('heart disese predictor.pkl','rb'))
load_model.score(x_test,y_test)


# In[ ]:


sklearn_steps=[
    "Getting the data ready",
    "choosing machine learning model",
    "FIT MODEL",
    "evaluate model",
    "improve model",
    "saving the model",
    "summary"
]


# In[ ]:


sklearn_steps


# # 1 Getting your data ready
# ##1.2 split data into features and label,(independent vs dependent varibale),x,y
# ##1.3 filling missing value
# ##1.4 converting data types

# In[ ]:


heart_des.head()


# In[ ]:


x=heart_des.drop('target',axis=1)


# In[ ]:


y=heart_des['target']


# In[ ]:


x.head()


# In[ ]:


y.head()


# In[ ]:


len(heart_des)


# In[ ]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[ ]:


phone_d=pd.read_csv('pandas.csv')
phone_d


# In[ ]:


phone_d.dtypes


# In[ ]:


phone_d['price']=phone_d['price'].str[:-2]
phone_d['price']


# In[ ]:


#phone_d['price']=phone_d['price'].str.replace('[\$]','',regex=True).astype(int)
phone_d['price']=phone_d['price'].str.replace('[\$,\.]', '', regex=True).astype(int)


# In[ ]:


phone_d['price']


# In[ ]:


x=phone_d.drop('price',axis=1)
y=phone_d['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x,y


# In[ ]:


#to fit model
from sklearn.ensemble import RandomForestRegressor
phone_model=RandomForestRegressor()
phone_model.fit(x_train,y_train)
phone_model.score(x_test,y_test)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
features_cat=['make','colour','dim card']
hottie=OneHotEncoder()
transformer=ColumnTransformer([('One_hotti',hottie,features_cat)],remainder="passthrough")
transformed_x=transformer.fit_transform(x)
pd.DataFrame(transformed_x)


# In[ ]:


#2nd method for transformation
transformed_new=pd.get_dummies(phone_d[['make','colour','dim card']])
transformed_new


# In[ ]:


#lets refit the model
x_train,x_test,y_train,y_test=train_test_split(transformed_new,y, test_size=0.2)
phone_model.fit(x_train,y_train)


# In[ ]:


phone_model.score(x_test,y_test)


# In[ ]:


phone_missing_values=pd.read_csv('pandasmissing2.csv')


# In[ ]:


phone_missing_values


# In[ ]:


phone_missing_values.isna().sum()


# In[ ]:


x=phone_missing_values.drop('price',axis=1)
y=phone_missing_values['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x,y
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
features_cat=['make','colour','dim card']
hottie=OneHotEncoder()
transformer=ColumnTransformer([('One_hotti',hottie,features_cat)],remainder="passthrough")
transformed_x=transformer.fit_transform(x)
pd.DataFrame(transformed_x)


# In[ ]:


#fill nan of make column
phone_missing_values['make'].fillna("missing",inplace=True)
phone_missing_values['colour'].fillna("missing",inplace=True)
phone_missing_values['memory'] = pd.to_numeric(phone_missing_values['memory'], errors='coerce')
phone_missing_values['memory'].fillna(phone_missing_values['memory'].mean(),inplace=True)


# In[ ]:


phone_missing_values.isna().sum()


# In[ ]:


phone_missing_values.dropna(inplace=True)


# In[ ]:


phone_missing_values.isna().sum()


# In[ ]:


phone_data_missing=pd.read_csv('pandasmissing2.csv')
phone_data_missing


# In[ ]:


phone_data_missing.isna().sum()


# In[ ]:


phone_data_missing.dropna(subset=['price'],inplace=True)


# In[ ]:


phone_data_missing.isna().sum()


# In[ ]:


x=phone_data_missing.drop('price', axis=1)
x['memory'] = pd.to_numeric(x['memory'], errors='coerce')
y=phone_data_missing['price']


# In[ ]:


#imputation
#filling data with sklearn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
#fill categorical valuer with missing and numerical values with mean
cat_feature=SimpleImputer(strategy='constant',fill_value='missing')
simcard_feature=SimpleImputer(strategy='constant',fill_value=4)
num_feature=SimpleImputer(strategy='mean')
#DEFINE colum
cat_column=['make','colour'] #make	colour	memory	dim card	price
simcard_column=['dim card']
num_column=['memory']

imputer=ColumnTransformer([('cat_feature',cat_feature,cat_column),
                           ('simcard_feature',simcard_feature,simcard_column),
                           ('num_feature',num_feature,num_column)])
filled_x=imputer.fit_transform(x)
filled_x


# In[ ]:


phone_data_filled=pd.DataFrame(filled_x,columns=['make','colour','memory','dim card'])
phone_data_filled


# In[ ]:


phone_data_filled.isna().sum()


# # picking up model for regression model
# ###importing data ,boston Housing
# from sklearn.datasets import  

# In[ ]:


#data_url = "http://lib.stat.cmu.edu/datasets/boston"


# In[ ]:


# boston is not avilable in skitlearn
#from sklearn.datasets import load_boston
#boston=load_boston()
#boston


# In[ ]:


from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
california = fetch_california_housing()
print(california)


# In[ ]:


df_california=pd.DataFrame(california['data'],columns=california['feature_names'])
df_california['target']=pd.Series(california['target'])
df_california.head()


# In[ ]:


len(df_california)


# In[ ]:


#lets try redge regression model
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import test_train_split
#to reproduce same result for fello ml scientist
np.random.seed(1)
#create data
x=df_california.drop('target',axis=1)
y=df_california['target']
#split test and train
x_test,y_test,y_train,y_test=test_train_split(x,y,test_size=0.2)
#institation ridge model
reg_model=Ridge()
reg_model.fit(x_train,y_train)
#check the score of the model
reg_model.score(x_test,y_test)


# In[ ]:


#It looks like there's a typo in the function call for splitting the data. 
#The correct function name is train_test_split not test_train_split.
#Also, ensure the order of the variables in the train_test_split function is correct.

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import train_test_split

# Load the California housing dataset
california = fetch_california_housing(as_frame=True)
df_california = california.frame

# Set random seed for reproducibility
np.random.seed(1)

# Create data
x = df_california.drop('MedHouseVal', axis=1)
y = df_california['MedHouseVal']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Instantiate and fit the Ridge regression model
reg_model = Ridge()
reg_model.fit(x_train, y_train)

# Check the score of the model
score = reg_model.score(x_test, y_test)
print(f"Ridge Regression model score: {score}")


# In[ ]:


heart_d=pd.read_csv("heart.csv")
heart_d


# In[ ]:


len(heart_d)



# In[ ]:


#we imported svc model
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import train_test_split
# Set random seed for reproducibility
np.random.seed(1)
# Create data
x = heart_d.drop('target', axis=1)
y = heart_d['target']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Instantiate and fit the Ridge regression model
svc_model = LinearSVC()
svc_model.fit(x_train, y_train)

# Check the score of the model
svc_model.score(x_test, y_test)
#print(f"Ridge Regression model score: {score}")


# In[ ]:


#using random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import train_test_split
# Set random seed for reproducibility
np.random.seed(1)
# Create data
x = heart_d.drop('target', axis=1)
y = heart_d['target']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Instantiate and fit the Ridge regression model
RFC_model = RandomForestClassifier()
RFC_model.fit(x_train, y_train)

# Check the score of the model
RFC_model.score(x_test, y_test)
#print(f"Ridge Regression model score: {score}")


# In[ ]:


#STRUCRURED DATA  ensemble METHODS
#USTRUCTURED DATA ----DEEP LEARNING


# In[ ]:


sklearn_steps


# In[ ]:


#institate ridge model
reg_model=Ridge()
reg_model.fit(x_train,y_train)
#check thr score of the model
reg_model.score(x_test,y_test)


# In[ ]:


x.head()


# # make prediction

# In[ ]:


#2 ways we can run the predict
#predict()
#predict_proba()


# In[ ]:


#step 1heart_d=pd.read_csv('heart.csv')
heart_d.head()


# In[ ]:


#now to divide the data
x=heart_d.drop('target', axis=1)
y=heart_d['target']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(x_train,y_train)


# In[ ]:


#Now to split data 
from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.2)


# In[ ]:


y_predicted=clf.predict(x_test)
y_predicted  #predicted value


# In[ ]:


np.array(y_test) #ground reality 


# In[ ]:


np.mean(y_predicted==y_test) #method 1


# In[ ]:


clf.score(x_test,y_test) #method 2


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predicted)#method 3


# # 

# In[ ]:


clf.predict(x_test)[:5]


# In[ ]:


clf.predict_proba(x_test)[:5]


# In[ ]:


df_california


# In[ ]:


df_california


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
np.random.seed(1)
#create data
x=df_california.drop('MedHouseVal',axis=1)
y=df_california['MedHouseVal']
#split into test and train
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)
#institate  model
Regression_model =RandomForestRegressor().fit(x_train,y_train)
#make predication
predicted_y=Regression_model.predict(x_test)


# In[ ]:


predicted_y[:10]


# In[ ]:


np.array(y_test[:10])


# In[ ]:


#compare thje predication with ground reality
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,predicted_y)


# In[ ]:


heart_d=pd.read_csv("heart.csv")
heart_d


# #Three ways to elevate sk learn models
# ## estimator 'score' methods
# ###the scoring parameters
# ###problem specific metric function
# heart_d=pd.read_csv("heart.csv")
# heart_d

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
np.random.seed(1)
#split data
x=heart_d.drop('target',axis=1)
y=heart_d['target']
x_train,x_test ,y_train,y_test=train_test_split(x,y, test_size=0.2)
clf=RandomForestClassifier().fit(x_train,y_train)
clf


# In[ ]:


clf.score(x_train,y_train)


# In[ ]:


clf.score(x_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




