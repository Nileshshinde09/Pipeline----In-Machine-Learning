import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import pickle


df  = pd.read_csv('train.csv')
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace = True)
x_train,x_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']),df['Survived'],test_size = 0.3,random_state = 44)
#imputation transformer
trf1 = ColumnTransformer([
    ("impute_age",SimpleImputer(),[2]),
    ("imputer_embarked",SimpleImputer(strategy = 'most_frequent'),[6])
],remainder = 'passthrough')#--> passthrough means dont drop another columns

#one hot encodeing
trf2 = ColumnTransformer([
    ("O.H.E_sex_embarked",OneHotEncoder(sparse=False,handle_unknown="ignore"),[1,6])
],remainder = "passthrough")

trf3 = ColumnTransformer([
    ("scale",MinMaxScaler(),slice(0,10))
])# slice uses for to apply MinMaxScaler to all the column to the range of (0,10)
trf5 = DecisionTreeClassifier()
#Alternate Syntax
pipe = make_pipeline(trf1,trf2,trf3,trf5)

pipe.fit(x_train,y_train)
print(pipe)

y_pred = pipe.predict(x_test)
accuracy_score(y_test,y_pred)*100
pickle.dump(pipe,open('pipe3.pkl','wb'))