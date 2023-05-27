
from asyncio.windows_utils import pipe
from operator import index
from pickle import NONE, TRUE
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import pickle

data=pd.read_csv('C:\\Users\\Dell\\Desktop\\Python\\ML\\Bengaluru_House_Data.csv')

# print(data.head())
# print(data.shape)
# print(data.info())

# for column in data.columns:
#     print(data[column].value_counts())
#     print('*'*50)

# print(data.isna().sum())

data=data.drop(columns=['area_type','availability','society','balcony','bath'])
# print(data.info())

# print(data['location'].value_counts())
data['location']=data['location'].fillna('Whitefield')
# print(data.info())

# print(data['size'].value_counts())
data['size']=data['size'].fillna('2 BHK')
# print(data.info())

data['bhk']=data['size'].str.split().str.get(0).astype(int)
# print(data['bhk'])

def convertRange(x):
    temp=x.split('-')
    if len(temp)==2:
        return ((float(temp[0])+float(temp[1]))/2)
    try:
        return float(x)
    except:
        return NONE

data['total_sqft']=data['total_sqft'].apply(convertRange)
# print(data.head())

data['price_per_feet']=data['price']*1e5/(pd.to_numeric(data['total_sqft'],errors='coerce'))
# print(data['price_per_feet'])

data['location']=data['location'].apply(lambda x: x.strip())
location_count=data['location'].value_counts()
location_count_less_10=location_count[location_count<=10]
data['location']=data['location'].apply(lambda x: 'others' if x in location_count_less_10 else x)
# print(data['location'].value_counts())

# print((pd.to_numeric(data['total_sqft'],errors='coerce')/data['bhk']).describe())
data=data[((pd.to_numeric(data['total_sqft'],errors='coerce')/data['bhk'])>=300)]
# print(data.describe())
# print(data.shape)

def remove_Outliers_price_per_sqft(df):
    df_output=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_feet)
        sd=np.std(subdf.price_per_feet)

        gen_df=subdf[(subdf.price_per_feet > (m-sd)) & (subdf.price_per_feet <= (m+sd))]
        df_output=pd.concat([df_output,gen_df],ignore_index=True)
    return df_output

data=remove_Outliers_price_per_sqft(data)
# print(data.describe())

def remove_Outliers_bhk(df):
    exclude_rows=np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean': np.mean(bhk_df.price_per_feet),
                'std': np.std(bhk_df.price_per_feet),
                'count': bhk_df.shape[0]
            }
        # print(location,bhk_stats)
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_rows=np.append(exclude_rows,bhk_df[bhk_df.price_per_feet<(stats['mean'])].index.values)
    return df.drop(exclude_rows)

data=remove_Outliers_bhk(data)
# print(data.shape)
# print(data.head())
data=data.drop(columns=['size','price_per_feet'])
# print(data.head())
data.to_csv("Cleaned_data.csv")

# print(data['location'].dtype)
# print(data['total_sqft'].dtype)
# print(data['price'].dtype)
# print(data['bhk'].dtype)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.compose import make_column_transformer

Y=data['price']
X=data.drop(columns=['price'])

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
column_trans=make_column_transformer((OneHotEncoder(sparse=False),['location']),remainder='passthrough')
scalar=StandardScaler()
model=LinearRegression(normalize=True)
pipe=make_pipeline(column_trans,scalar,model)
pipe.fit(X_train,Y_train)
Y_Predict_Model=pipe.predict(X_test)

print(np.round(r2_score(Y_test,Y_Predict_Model)*1e2,2))

# Testing of the Result
input=pd.DataFrame([['6th Phase JP Nagar','1500',4]],columns=['location','total_sqft','bhk'])
prediction=np.round(pipe.predict(input)[0]*1e5,2)
print(str(prediction),'rs.')

pickle.dump(pipe,open("Linear_Regression_Model.pkl",'wb'))