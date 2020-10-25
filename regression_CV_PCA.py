import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import functions as f

# open the dataset
dataset_path = './dataset/'
dataset = pd.read_csv(dataset_path + 'not_standardized_dataset_drop.csv')

# outliers removal
dataset = f.remove_outliers(dataset)

# ridge regression with nested cross validation
X = dataset.drop(columns=['median_house_value']).to_numpy()  # data matrix
y = dataset['median_house_value'].to_numpy()  # labels vector

# nested cross validation
alpha = np.arange(0, 100, 1)                                            # values of the hyperparameter alpha
k1 = 10                                                                 # number of outer folds
k2 = 5                                                                  # number of inner folds

# PCA: analysis training and test error with dimensionality reduction, from 1 to 13 components
components = [c for c in range(1, X.shape[1] + 1)]                      # list of components

train_error_list = []                                                   # list of the means of training errors
test_error_list = []                                                    # list of the means of test errors
train_r2_list = []                                                      # list of the means of training r2 score
test_r2_list = []                                                       # list of the means of test r2 score

for comp in components:                                                 # for each component
    pca = PCA(n_components=comp)                                        # initialize the dimensionality reduction
    pca.fit(X)                                                          # fit the PCA model on the dataset
    X_transformed = pca.transform(X)                                    # get transformed data matrix

    # add ones column to X
    ones = np.ones((X.shape[0], 1))
    X_transformed = np.hstack((ones, X_transformed))

    scores = f.nestedCV(X_transformed, y, k1, k2, alpha, info=False)    # compute the nested CV

    # extract values from the dictionary
    rmse_train, rmse_test = scores['training']['rmse'], scores['test']['rmse']
    r2_train, r2_test = scores['training']['r2'], scores['test']['r2']

    # appending mean values of results for mse and r2 score
    train_error_list.append(np.mean(rmse_train))
    test_error_list.append(np.mean(rmse_test))
    train_r2_list.append(np.mean(r2_train))
    test_r2_list.append(np.mean(r2_test))

    # print results
    f.print_pca_results(comp, train_r2_list, train_error_list, test_r2_list, test_error_list, scores)

# plot results
f.plot_pca_results(train_r2_list, test_r2_list, train_error_list, test_error_list)
