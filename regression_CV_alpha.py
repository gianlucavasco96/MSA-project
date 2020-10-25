import pandas as pd
import numpy as np

import functions as f

# open the dataset
dataset_path = './dataset/'
datasets = ['standardized_dataset.csv', 'not_standardized_dataset.csv',
            'standardized_dataset_drop.csv', 'not_standardized_dataset_drop.csv']

for n, name in enumerate(datasets):
    dataset = pd.read_csv(dataset_path + name)

    # outliers removal
    # dataset = f.remove_outliers(dataset)

    # ridge regression with nested cross validation
    X = dataset.drop(columns=['median_house_value']).to_numpy()  # data matrix
    y = dataset['median_house_value'].to_numpy()  # labels vector

    # add ones column to X
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    # k-folds cross validation
    print('K-FOLDS CROSS VALIDATION')
    k = 10                                                                  # number of folds
    alpha = np.arange(0, 10000, 100)                                        # values of the hyperparameter alpha

    # initialize results lists
    rmse_train = []
    rmse_test = []
    r2_train = []
    r2_test = []

    # cross validated risk analysis w.r.t. alpha
    for par in alpha:
        scores = f.k_foldsCV(X, y, k, par)                                  # compute the k-folds CV

        # results average and append
        rmse_train.append(np.mean(scores['training']['rmse']))
        rmse_test.append(np.mean(scores['test']['rmse']))
        r2_train.append(np.mean(scores['training']['r2']))
        r2_test.append(np.mean(scores['test']['r2']))

        print('Alpha = ' + str(par))                                        # print current value of alpha
        f.print_scores(scores)                                              # print k-folds CV results

    # plot results
    f.plot_kfolds_alpha(rmse_train, rmse_test, r2_train, r2_test, alpha, n, datasets)
