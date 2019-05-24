import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
desired_width = 400
pd.set_option('display.width',desired_width)
pd.set_option('display.max_column',30)
df=pd.read_csv("C:\\Users\\Ayush gupta\\Downloads\\kc_house_data.csv")
df1 = pd.DataFrame(df)
cols=['id','date','price','bedrooms','sqft_living','sqft_lot',
      'floors','sqft_above','sqft_basement','yr_built','zipcode','lat','long','sqft_living15','sqft_lot15']
train1=df.drop(['id','price'],axis=1)
print(df[:].isnull().sum())
print(train1.head())
print(df.describe())
df['bedrooms'].value_counts().plot(kind='pie')
plt.title('Number of bedrooms')
k = [7,2,12,1]
labels = 'Hall','Kitchen','Living_room','Bedroom','Bathroom','Store_room'
plt.pie(k, labels=labels)
plt.xlabel('Bedrooms')
plt.ylabel('count')
sns.despine()
# plt.legend()
plt.show()
reg=LinearRegression()
conv_dates=[1 if values == 2014 else 0 for values in df.date ]
df['date']=conv_dates
labels=df['price']
train1=df.drop(['id','price'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.10,random_state=2)
reg.fit(x_train,y_train)
predictions = reg.predict(x_test)
print(predictions)
pred=reg.score(x_test,y_test)*100
print(pred)

