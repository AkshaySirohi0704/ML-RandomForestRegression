# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# position_salaries.csv contains two feature (Non-Linear Data)
print('-------------------------------------------------')
print('Seprating features and dependent variable . . . ')
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# As dataset's rows are less, Splitting won't be necessary

# Spliting
# from sklearn.model_selection import train_test_split
# print('-------------------------------------------------')
# print('splitting . . . ')
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Displaying
print('-------------------------------------------------')
print('Length of X', len(X))
print(X)
print('-------------------------------------------------')
print('Length of y_train', len(y))
print(y)

# Random Forest regression (Ensemle Learning)
# Ensemble Learning - Takes several Algo or same Algo multiple time
#                     put them together to make something much more
#                     powerful.
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)

# Training
print('-------------------------------------------------')
print('Machine is learning . . .')
regressor.fit(X, y)

# Inverse Transforming and Predicting
print('-------------------------------------------------')
print('Predicting . . .')
print('According to use reading data value should be in between 150000 - 200000')
print(regressor.predict([[6.5]]))

print('-------------------------------------------------')
print('Visualising the Result')

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()