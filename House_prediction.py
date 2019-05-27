import numpy as np #import packages 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
desired_width = 400 # change the display width in Pycharm console
pd.set_option('display.width',desired_width)
pd.set_option('display.max_column',30)

#Convert the file into CSV format and import it from local system

df=pd.read_csv("C:\\Users\\Ayush gupta\\Downloads\\kc_house_data.csv") 
df1 = pd.DataFrame(df) # converted into dataframe from series

cols=['id','date','price','bedrooms','sqft_living','sqft_lot',
      'floors','sqft_above','sqft_basement','yr_built','zipcode','lat','long','sqft_living15','sqft_lot15'] # Filtered coloumns from data

train1=df.drop(['id','price'],axis=1)
print(df[:].isnull().sum()) # Convert the null values into valid values

# print(train1.head()) # Display the top results on Pycharm Console
# print(df.describe()) # Display the describe function

df['bedrooms'].value_counts().plot(kind='pie') # For data visualization through Matplotlib in Pie bar
plt.title('Number of bedrooms')
k = [7,2,12,1]
labels = 'Hall','Kitchen','Living_room','Bedroom','Bathroom','Store_room'
plt.pie(k, labels=labels)
plt.xlabel('Bedrooms')
plt.ylabel('count')
sns.despine()
# plt.legend()
# plt.show() # Display the pie chart

reg=LinearRegression() # Apply linear regression on data

conv_dates=[1 if values == 2014 else 0 for values in df.date ]
df['date']=conv_dates #Convert the date format into a single valid number

labels=df['price']
train1=df.drop(['id','price'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.10,random_state=2) # Split the data set into train case and test case.

# Applied algorithm
reg.fit(x_train,y_train)
predictions = reg.predict(x_test) # Predict the result
print(predictions)
pred=reg.score(x_test,y_test)*100 # Predict the regression score
print(pred) 

# The coefficients
print('Coefficients: \n', reg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test,predictions))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test,predictions))

