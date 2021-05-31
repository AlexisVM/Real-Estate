import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

house_dataset = pd.read_csv('USA_Housing.csv')
house_dataset.head()

house_dataset.columns

X = house_dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y = house_dataset['Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=.40, random_state=101)
X_train

from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(X_train, y_train)

coeff = pd.DataFrame(linear.coef_, X.columns, columns=['Coefficient'])

predictions = linear.predict(X_test)

plt.scatter(y_test, predictions)

sns.displot((y_test-predictions),bins=50)