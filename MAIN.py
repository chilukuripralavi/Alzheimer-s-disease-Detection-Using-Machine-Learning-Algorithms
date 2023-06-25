
# %% [code]
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn import model_selection

# %% [code]
data=pd.read_csv('oasis_longitudinal.csv')

# %% [code]
data.head()

# %% [code]
data.shape

# %% [code]
data.columns

# %% [code]
data.describe()

# %% [code]
data.info()

# %% [code]
df= data.loc[data['Visit']==1]

# %% [code]
df = df.reset_index(drop=True)

# %% [code]
df.head()

# %% [code]
from sklearn.preprocessing import LabelEncoder 

# %% [code]
lab=LabelEncoder()

# %% [code]
df['M/F']=lab.fit_transform(df['M/F'])
df['Group']=lab.fit_transform(df['Group'])
df['Hand']=lab.fit_transform(df['Hand'])

# %% [code]
df.head()

# %% [code]
df1=df.drop(df[['Subject ID','MRI ID','Visit']],axis=1)
df1.head()

# %% [code]
df1.isnull().sum()

# %% [code]
df2= df1.dropna()
df2.isnull().sum()
df2.head()

# %% [code]
sns.countplot(x='Group',data=df2)

# %% [code]
def bar_chart(feature):
    Demented =df[df['Group']==1][feature].value_counts()
    Nondemented = df[df['Group']==0][feature].value_counts()
    df_bar = pd.DataFrame([Demented,Nondemented])
    df_bar.index = ['Demented','Nondemented']
    df_bar.plot(kind='bar',stacked=True,figsize=(8,5))
    

# %% [code]
bar_chart('M/F')
plt.xlabel('Group')
plt.ylabel('Number of patients')
plt.legend()
plt.title('Gender and Demented rate')

# %% [code]
facet=sns.FacetGrid(df,hue='Group', aspect=3)
facet.map(sns.kdeplot,'MMSE',shade=True)
facet.set(xlim=(0,df['MMSE'].max()))
facet.add_legend()
plt.xlim(15,30)

# %% [code]
facet=sns.FacetGrid(df,hue='Group', aspect=3)
facet.map(sns.kdeplot,'ASF',shade=True)
facet.set(xlim=(0,df['ASF'].max()))
facet.add_legend()
plt.xlim(0.5,2)

# %% [code]
facet=sns.FacetGrid(df,hue='Group', aspect=3)
facet.map(sns.kdeplot,'nWBV',shade=True)
facet.set(xlim=(0,df['nWBV'].max()))
facet.add_legend()
plt.xlim(0.6,0.9)

# %% [code]
facet=sns.FacetGrid(df,hue='Group', aspect=3)
facet.map(sns.kdeplot,'eTIV',shade=True)
facet.set(xlim=(0,df['eTIV'].max()))
facet.add_legend()
plt.xlim(900,2100)

# %% [code]
facet=sns.FacetGrid(df,hue="Group", aspect=3)
facet.map(sns.kdeplot,'EDUC',shade=True)
facet.set(xlim=(df['EDUC'].min(),df['EDUC'].max()))
facet.add_legend()
plt.ylim(0,0.16)

# %% [code]
x=df2.iloc[:,df2.columns!='Group']
y=df2.iloc[:,df2.columns=='Group']

# %% [code]
x.shape

# %% [code]
x.head()

# %% [code]
y.head()

# %% [code]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
xtrain.head()

# %% [code]
ytrain.head()

# %% [markdown]
# ### svm classifier

# %% [code]
from sklearn.svm import SVC

# %% [code]
model1=SVC()

# %% [code]
model1.fit(xtrain,ytrain)

# %% [code]
predic1=model1.predict(xtest)

# %% [code]
from sklearn.metrics import accuracy_score
acc1=accuracy_score(predic1,ytest)
acc1

# %% [code]
from sklearn.metrics import classification_report

# %% [code]
def apply_classifier(model1,xtrain,xtest,ytrain,ytest):
    model1.fit(xtrain,ytrain)
    predictions=model1.predict(xtest)
    print("\n classification report : \n {}",format(classification_report(ytest,predictions)))

# %% [code]
apply_classifier(model1,xtrain,xtest,ytrain,ytest)

# %% [markdown]
# ### SVM with Kernal Tricks

# %% [code]
model_linear_kernal = SVC(kernel='linear')
model_linear_kernal.fit(xtrain, ytrain)

# %% [code]
apply_classifier(model_linear_kernal,xtrain,xtest,ytrain,ytest)

# %% [code]
results = model_selection.cross_val_score(model_linear_kernal, x, y)
print("MeanSqareError(MSE): %.3f (%.3f)" % (results.mean(), results.std()))

# %% [markdown]
# ### Decision Tree Classifier

# %% [code]
from sklearn.tree import DecisionTreeClassifier
decision_tree=DecisionTreeClassifier(random_state= 42)
apply_classifier(decision_tree,xtrain,xtest,ytrain,ytest)

# %% [markdown]
# ### Random Forest

# %% [code]
from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier(random_state=42)
apply_classifier(random_forest,xtrain,xtest,ytrain,ytest)

# %% [markdown]
# ### Gaussian Naive_bayes

# %% [code]
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(xtrain,ytrain)
model.score(xtest,ytest)
apply_classifier(model,xtrain,xtest,ytrain,ytest)

# %% [markdown]
# ###### Machine learning approach to predict the Alzheimer disease using machine learning algorithms is successfully implemented and gives greater prediction accuracy results. The model predicts the disease in the patient and also distinguishes between the cognitive impairment

# %% [markdown]
# ### CONCLUSION
# 

# %% [markdown]
# After training our models on the variables. it is estimated that SVM kernal trics, Naïve Bayes and Random Forest gives higher accuracy of 90 % in the cases on predicting Alziheimer disease. The results of the presented work can be used for enhancing defense against terrorist attacks in coming times.The conduct of the algorithms is compared based on their accuracy. Then the dataset is partitioned according to that ratio and when the algorithms are compared the best one is selected from SVM kernal trics, Naïve Bayes and Random Forest and can be used for next stage of prediction.
# 
# 
