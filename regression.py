import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import functions as f

# open the dataset
dataset_path = './dataset/'
dataset = pd.read_csv(dataset_path + 'not_standardized_dataset_drop.csv')

# f.correlation_plot(dataset)                                 # plot correlation heatmap and dependency with the label

# ridge regression with nested cross validation
X = dataset.drop(columns=['median_house_value']).to_numpy()             # data matrix
y = dataset['median_house_value'].to_numpy()                            # labels vector

# add ones column to X
ones = np.ones((X.shape[0], 1))
X = np.hstack((ones, X))

# initialize results lists
rmse_train = []
rmse_test = []
r2_train = []
r2_test = []

for i in range(20):
    # split the data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

    w = f.ridge_regression(X_train, y_train, 0.001)                         # compute the best parameter w
    pred_train = f.predictions(X_train, w)                                  # compute predictions on training set
    pred_test = f.predictions(X_test, w)                                    # compute predictions on test set

    # results
    r2_train.append(f.r2_score(y_train, pred_train))                        # compute r2 score for training predictions
    rmse_train.append(f.root_mse(y_train, pred_train))                      # compute training error (RMSE)

    r2_test.append(f.r2_score(y_test, pred_test))                           # compute r2 score for test predictions
    rmse_test.append(f.root_mse(y_test, pred_test))                         # compute test error (RMSE)

# print results
f.print_results(r2_train, r2_test, rmse_train, rmse_test)
plt.show()
