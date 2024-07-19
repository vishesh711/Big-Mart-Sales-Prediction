# Big Mart Sales Prediction Model

## Overview

This project involves building an XGBoost regression model to predict sales of products at various outlets of a retail store. The dataset used for this project is obtained from a CSV file and contains 8523 entries with 12 columns.

## Dataset

The dataset consists of the following columns:
- **Item_Identifier**: Unique product ID
- **Item_Weight**: Weight of the product
- **Item_Fat_Content**: Whether the product is low fat or regular
- **Item_Visibility**: The percentage of total display area of all products in a store allocated to this particular product
- **Item_Type**: The category to which the product belongs
- **Item_MRP**: Maximum Retail Price (list price) of the product
- **Outlet_Identifier**: Unique store ID
- **Outlet_Establishment_Year**: The year in which the store was established
- **Outlet_Size**: The size of the store (Small, Medium, High)
- **Outlet_Location_Type**: The type of city in which the store is located
- **Outlet_Type**: Whether the outlet is just a grocery store or some sort of supermarket
- **Item_Outlet_Sales**: Sales of the product in the particular store (target variable)

## Data Preprocessing

The data preprocessing steps include:
1. Loading the data into a Pandas DataFrame.
2. Checking for null values and filling them appropriately.
3. Descriptive statistics of the dataset.
4. Visualizing the distribution of each feature.
5. Encoding categorical variables using numerical values.

## Handling Missing Values

- **Item_Weight**: Filled missing values with the mean of the column.
- **Outlet_Size**: Filled missing values with the mode of the column based on the `Outlet_Type`.

## Feature Encoding

We encoded categorical variables:
- **Item_Identifier**: Label encoding
- **Item_Fat_Content**: Standardizing labels to 'Low Fat' and 'Regular' and then label encoding
- **Item_Type**: Label encoding
- **Outlet_Identifier**: Label encoding
- **Outlet_Size**: Label encoding
- **Outlet_Location_Type**: Label encoding
- **Outlet_Type**: Label encoding

## Data Visualization

We visualized the distribution of the following features:
- **Item_Weight Distribution**
- **Item_Visibility Distribution**
- **Item_MRP Distribution**
- **Item_Outlet_Sales Distribution**
- **Outlet_Establishment_Year Count**
- **Item_Fat_Content Count**
- **Item_Type Count**
- **Outlet_Size Count**

## Model Building

We used an XGBoost regressor to predict the item outlet sales. The steps involved:
1. Splitting the data into training and testing sets.
2. Training the XGBoost regressor model on the training set.
3. Evaluating the model using the R-squared value on both training and testing sets.

## Model Evaluation

The R-squared values obtained were:
- Training set: 0.8762
- Testing set: 0.5017

These values indicate that the model explains approximately 88% of the variance in the training set and 50% in the testing set.

## Code

Here is the complete code for the project:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# loading the data from csv file to Pandas DataFrame
big_mart_data = pd.read_csv('/content/Train.csv')
# first 5 rows of the dataframe
big_mart_data.head()

# number of data points & number of features
big_mart_data.shape

# getting some information about the dataset
big_mart_data.info()

# checking for missing values
big_mart_data.isnull().sum()

# mean value of "Item_Weight" column
big_mart_data['Item_Weight'].mean()

# filling the missing values in "Item_weight column" with "Mean" value
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)

# mode of "Outlet_Size" column
big_mart_data['Outlet_Size'].mode()

# filling the missing values in "Outlet_Size" column with Mode
mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
print(mode_of_Outlet_size)

miss_values = big_mart_data['Outlet_Size'].isnull()
print(miss_values)

big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])

# checking for missing values
big_mart_data.isnull().sum()

big_mart_data.describe()

sns.set()

# Item_Weight distribution
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Weight'])
plt.show()

# Item Visibility distribution
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Visibility'])
plt.show()

# Item MRP distribution
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_MRP'])
plt.show()

# Item_Outlet_Sales distribution
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Outlet_Sales'])
plt.show()

# Outlet_Establishment_Year column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
plt.show()

# Item_Fat_Content column
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
plt.show()

# Item_Type column
plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=big_mart_data)
plt.show()

# Outlet_Size column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=big_mart_data)
plt.show()

big_mart_data.head()

big_mart_data['Item_Fat_Content'].value_counts()

big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)

big_mart_data['Item_Fat_Content'].value_counts()

encoder = LabelEncoder()

big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])
big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

big_mart_data.head()

X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

regressor = XGBRegressor()

regressor.fit(X_train, Y_train)

# prediction on training data
training_data_prediction = regressor.predict(X_train)

# R squared Value
r2_train = metrics.r2_score(Y_train, training_data_prediction)

print('R Squared value = ', r2_train)

# prediction on test data
test_data_prediction = regressor.predict(X_test)

# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)

print('R Squared value = ', r2_test)
```

## Conclusion

This project demonstrates the process of building an XGBoost regression model to predict item outlet sales. The model shows good predictive power with an R-squared value around 0.88 on the training set. However, the performance on the test set indicates potential overfitting, suggesting that further model tuning and feature engineering could improve its performance.
