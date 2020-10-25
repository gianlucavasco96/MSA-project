import pandas as pd
import numpy as np

import functions as f

# open the dataset
dataset_path = './dataset/'
datasets = ['standardized_dataset.csv', 'not_standardized_dataset.csv',
            'standardized_dataset_drop.csv', 'not_standardized_dataset_drop.csv']

cv_type = 'nested'

for n, name in enumerate(datasets):
    dataset = pd.read_csv(dataset_path + name)
    scores = []                                                                 # list of scores w/ and w/o outliers
    for i in range(2):                                                          # the first loop is with outliers
        if i == 1:                                                              # the second one is without them
            dataset = f.remove_outliers(dataset)                                # outliers removal

        X = dataset.drop(columns=['median_house_value']).to_numpy()             # data matrix
        y = dataset['median_house_value'].to_numpy()                            # labels vector

        # add ones column to X
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        if cv_type == 'k-folds':
            # k-folds cross validation
            print('K-FOLDS CROSS VALIDATION')
            k = 10                                                              # number of folds

            scores.append(f.k_foldsCV(X, y, k, 0.001))                          # compute the k-folds CV
        else:
            print()

            # nested cross validation
            print('NESTED CROSS VALIDATION')
            alpha = np.arange(0, 100, 1)                                        # values of the hyperparameter alpha
            k1 = 10                                                             # number of outer folds
            k2 = 5                                                              # number of inner folds

            scores.append(f.nestedCV(X, y, k1, k2, alpha))                      # compute the nested CV

        f.print_scores(scores[i])                                               # print k_folds or nested CV results
    f.plot_kfolds_bar(scores, n, name)                                          # bar plot
