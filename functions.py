import numpy as np
import pandas as pd
import seaborn as sb
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib import pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")


def correlation_plot(dataset):
    """This function computes the correlation matrix, in order to find the best model, for the data fitting, and
    plots the dependecy of the most correlated features with the label"""

    # find correlations and plot dependences
    plt.figure()
    correlations = dataset.corr()  # compute correlation matrix
    correlations = np.abs(correlations)  # take the absolute value
    sb.heatmap(correlations, annot=True, cmap='Reds')  # plot the heatmap
    plt.draw()

    cor_target = correlations['median_house_value']  # correlation with target
    relevant_features = cor_target[cor_target > 0.5]  # relevant features have more than 0.4 of correlation
    pearson_list = relevant_features.axes
    pearson_df = dataset[np.intersect1d(dataset.columns, pearson_list)]
    pearson_df = pearson_df.drop(columns='median_house_value')

    plt.figure()
    for i, feat in enumerate(pearson_df.keys()):
        plt.subplot(1, len(pearson_df.keys()), i + 1)
        plt.scatter(dataset[feat], dataset['median_house_value'], s=3)
        plt.xlabel(feat)
        plt.ylabel('median house value')
        plt.draw()


def data_splitter(X, y, test_idx):
    """This function splits the dataset into the training and the test part"""
    out_X_train = np.delete(X, test_idx, axis=0)                # outer training data
    out_X_test = X[test_idx, :]                                 # outer test data
    out_y_train = np.delete(y, test_idx, axis=0)                # outer training labels
    out_y_test = y[test_idx]                                    # outer test labels

    return out_X_train, out_X_test, out_y_train, out_y_test


def find_best_index(train_r2_list, test_r2_list, train_error_list, test_error_list):
    """This function finds the indexes of the best configuration of train/test score and train/test error"""

    # create combined lists
    r2_list = [train_r2_list, test_r2_list]
    err_list = [train_error_list, test_error_list]

    # sum element wise
    r2_sum = [sum(x) for x in zip(*r2_list)]
    err_sum = [sum(x) for x in zip(*err_list)]

    r2_idx = [i for i, x in enumerate(r2_sum) if x == max(r2_sum)]      # all argmax for r2 score
    err_idx = [i for i, x in enumerate(err_sum) if x == min(err_sum)]   # all argmin for mse

    inter = intersection(r2_idx, err_idx)                               # intersection of the two lists

    if len(inter) == 1:                                                 # if there is only one index in common
        best_idx = inter[0]                                             # the best idx is the value of the intersection
    else:                                                               # otherwise, the best idx is the first index
        best_idx = err_idx[0]                                           # in the list of error minima

    return best_idx


def get_name(name):
    """This function takes the input csv file name and transforms it in a string without symbols and extensions"""

    name = name[:-4]                                                    # remove the extension '.csv'
    name = name.replace('_', ' ')                                       # replace underscores with spaces
    name = name.capitalize()                                            # capitalize the first letter

    return name


def intersection(lst1, lst2):
    """This function computes the intersection of two lists, returning a list with the common elements"""

    return list(set(lst1) & set(lst2))


def k_foldsCV(data, labels, k, alpha):
    """This function computes the K-folds cross validation, in order to reduce the overfitting fenomenon"""

    # first of all, data matrix and labels vector have to be shuffled
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    X = data[indices]
    y = labels[indices]

    n = X.shape[0]                                              # number of rows of the training set
    rmse_train_list = []                                         # mean squared errors list for training set
    rmse_test_list = []                                          # mean squared errors list for test set
    r2_train_list = []                                          # R2 scores list for training set
    r2_test_list = []                                           # R2 scores list for test set

    for i in range(k):                                          # for each of the k folds
        start = i * (n // k)                                    # set the start index of the test folder
        end = (i+1) * (n // k - 1)                              # set the end index of the test folder
        test_idx = [x for x in range(start, end)]               # create the list of these indexes

        # split training data in k folds
        X_train, X_test, y_train, y_test = data_splitter(X, y, test_idx)

        w = ridge_regression(X_train, y_train, alpha)           # compute the best w
        pred_train = predictions(X_train, w)                    # make predictions on training set
        pred_test = predictions(X_test, w)                      # make predictions on test set

        train_error = root_mse(y_train, pred_train)             # compute training error
        rmse_train_list.append(train_error)                     # append it to the list of training errors
        r2_train = r2_score(y_train, pred_train)                # compute R2 score for training set
        r2_train_list.append(r2_train)                          # append it to the list of training r2 scores

        test_error = root_mse(y_test, pred_test)                # compute the test error
        rmse_test_list.append(test_error)                       # append it to the list of test errors
        r2_test = r2_score(y_test, pred_test)                   # compute R2 score for test set
        r2_test_list.append(r2_test)                            # append it to the list of test r2 scores

    # contruct a dictionary for results
    scores = {'training': {'rmse': rmse_train_list, 'r2': r2_train_list},
              'test': {'rmse': rmse_test_list, 'r2': r2_test_list}}

    return scores


def mean_squared_error(y_true, y_pred):
    """This function takes in input the true and the predicted labels and returns the mean squared error"""
    mse = np.mean((y_true - y_pred)**2)

    return mse


def nestedCV(data, labels, k1, k2, param, info=True):
    """This function computes the nested cross validation, in order to find the best value of the parameter alpha"""

    # first of all, data matrix and labels vector have to be shuffled
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    X = data[indices]
    y = labels[indices]

    n1 = X.shape[0]                                             # number of rows of the data matrix
    rmse_train_list = []                                         # mean squared errors list for training set
    rmse_test_list = []                                          # mean squared errors list for test set
    r2_train_list = []                                          # R2 scores list for training set
    r2_test_list = []                                           # R2 scores list for test set

    for i in range(k1):                                         # for each outer folder
        if info:
            print('Iteration ' + str(i+1) + ' of ' + str(k1))
        start_out = i * (n1 // k1)                              # set the start index for the outer test folder
        end_out = (i+1) * (n1 // k1 - 1)                        # set the end index for the outer test folder
        out_test_idx = [x for x in range(start_out, end_out)]   # create the list of these indexes

        # split data into (outer) training and test part
        out_X_train, out_X_test, out_y_train, out_y_test = data_splitter(X, y, out_test_idx)

        n2 = out_X_train.shape[0]                               # number of rows of the outer training data
        all_te_list = []                                        # list of all test errors, for each inner iteration

        for j in range(k2):                                     # for each inner folder
            start_inn = j * (n2 // k2)                          # set the start index for the inner test folder
            end_inn = (j+1) * (n2 // k2 - 1)                    # set the end index for the inner test folder
            inn_test_idx = [x for x in range(start_inn, end_inn)]

            # split again data into (inner) training and test part
            inn_X_train, inn_X_test, inn_y_train, inn_y_test = data_splitter(out_X_train, out_y_train, inn_test_idx)

            test_error_list = []                                # initialize the list of the test errors

            for alpha in param:                                 # for each value of alpha in the parameter set
                w = ridge_regression(inn_X_train, inn_y_train, alpha)   # compute the best w, inner training
                pred = predictions(inn_X_test, w)                       # make predictions on inner test data
                test_error = root_mse(inn_y_test, pred)                 # compute test error with current alpha
                test_error_list.append(test_error)                      # add current test error to the list

            all_te_list.append(test_error_list)                 # append the computed test errors to an external list

        te_array = np.asarray(all_te_list)                      # transform the list into an array

        mean_te_list = []                                       # create the mean test error values list
        for p in range(len(param)):                             # for each parameter alpha
            mean_te_list.append(np.mean(te_array[:, p]))        # append to the list the mean of each column

        best_idx = np.argmin(mean_te_list)                      # find the index of the minimum of the mean values
        best_alpha = param[best_idx]                            # which corresponds to the best parameter alpha

        if info:
            print('Best parameter alpha is ' + str(best_alpha))
            print()

        w = ridge_regression(out_X_train, out_y_train, best_alpha)      # compute the best w, outer training
        pred_train = predictions(out_X_train, w)                        # make predictions on outer training data
        pred_test = predictions(out_X_test, w)                          # make predictions on outer test data

        training_error = root_mse(out_y_train, pred_train)              # compute training error with the best alpha
        test_error = root_mse(out_y_test, pred_test)                    # compute test error with the best alpha

        r2_train = r2_score(out_y_train, pred_train)                    # compute r2 score for training with best alpha
        r2_test = r2_score(out_y_test, pred_test)                       # compute r2 score for test with best alpha

        rmse_train_list.append(training_error)                           # append current training error to mse list
        rmse_test_list.append(test_error)                                # append current test error to mse list

        r2_train_list.append(r2_train)                                  # append current training r2 score to its list
        r2_test_list.append(r2_test)                                    # append current test R2 score to its list

    # contruct a dictionary for results
    scores = {'training': {'rmse': rmse_train_list, 'r2': r2_train_list},
              'test': {'rmse': rmse_test_list, 'r2': r2_test_list}
              }

    return scores


def one_hot_encoding(dataset, drop=True):
    """This function performs the one-hot encoding of the categorical variables"""
    ocean_prox = dataset['ocean_proximity']
    count = ocean_prox.value_counts()
    print()
    cols = list(count.keys())                                   # list of distinct categorical values

    for col in cols:
        new_feat = np.zeros(len(ocean_prox))                    # inizialize new column with zeros
        idx = np.where(ocean_prox == col)[0]                    # find indexes of the current column value
        new_feat[idx] = 1                                       # set to 1 values corresponding to indexes
        dataset[col] = new_feat                                 # create the new column in the dataset

    dataset.pop('ocean_proximity')                              # removing old column

    if drop:                                                    # if drop flag is set to true
        # removing features with high VIF
        data_copy = dataset.copy()                              # create a copy of the dataset
        label = data_copy['median_house_value']                 # save labels variable
        data_copy.pop('median_house_value')                     # drop labels column

        # dataset standardization
        feat_to_skip = ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND', 'median_house_value']

        for key in data_copy.keys():
            if key in feat_to_skip:                             # one of k features and labels don't have to
                continue                                        # be normalized
            data_copy[key] = standardizer(data_copy[key])       # standardization of each feature

        while True:
            vif = pd.DataFrame()                                # create the Variance Inflator Factor dataframe
            vif['Feature'] = data_copy.columns
            vif['VIF'] = [variance_inflation_factor(data_copy.values, i) for i in range(data_copy.shape[1])]

            # print current vif dataframe
            print(vif.round(1))
            print()
            # print(vif.round(1).sort_values(by='VIF', ascending=False).to_latex(index=False))
            # print()

            # sort values from the maximum to the minimum
            vif_sort = vif.sort_values(by='VIF', ascending=False, ignore_index=True)
            if vif_sort['VIF'][0] > 5.0:                        # if the maximum VIF is greater than 5.0
                max_feat = vif_sort['Feature'][0]               # take the feature with maximum VIF
                data_copy.pop(max_feat)                         # and remove it from the dataset
                print(max_feat + ' has been dropped')
            else:
                break                                           # otherwise, while is concluded

        data_copy['median_house_value'] = label                 # ripristinate labels column
        dataset = dataset[data_copy.keys()]                     # take non standardized values in original dataset

    return dataset


def plot_kfolds_bar(scores, n, name):
    """This function plots the k-folds CV results, comparing the datasets with and without outliers"""

    # results with outliers
    rmse_train_out = np.mean(scores[0]['training']['rmse'])
    rmse_test_out = np.mean(scores[0]['test']['rmse'])
    r2_train_out = np.mean(scores[0]['training']['r2'])
    r2_test_out = np.mean(scores[0]['test']['r2'])

    # results without outliers
    rmse_train_no_out = np.mean(scores[1]['training']['rmse'])
    rmse_test_no_out = np.mean(scores[1]['test']['rmse'])
    r2_train_no_out = np.mean(scores[1]['training']['r2'])
    r2_test_no_out = np.mean(scores[1]['test']['r2'])

    r2_train = [r2_train_out, r2_train_no_out]
    r2_test = [r2_test_out, r2_test_no_out]
    rmse_train = [rmse_train_out, rmse_train_no_out]
    rmse_test = [rmse_test_out, rmse_test_no_out]

    width = 0.3
    x = np.arange(2)
    ticks = ['Normal', 'W/o outliers']

    plt.subplot(2, 4, n + 1)
    plt.xticks(x + width / 2, ticks)
    train = plt.bar(x, r2_train, width=width, color='blue', align='center')
    test = plt.bar(x + width, r2_test, width=width, color='red', align='center')

    if n == 0:
        plt.ylabel('AVG R2 score')

    plt.ylim((0, 1))
    plt.title(get_name(name))
    plt.legend([train, test], ['Training', 'Test'])

    plt.subplot(2, 4, n+5)
    plt.xticks(x + width / 2, ticks)
    train = plt.bar(x, rmse_train, width=width, color='blue', align='center')
    test = plt.bar(x + width, rmse_test, width=width, color='red', align='center')

    if n == 0:
        plt.ylabel('AVG RMSE')

    plt.yticks(fontsize=8)
    plt.legend([train, test], ['Training', 'Test'])
    plt.draw()

    if n == 3:
        plt.show()


def plot_kfolds_alpha(rmse_train, rmse_test, r2_train, r2_test, alpha, n, names):
    """This function plots the CV risk analysis results w.r.t. alpha"""

    x = alpha                                                   # x asis is the alpha vector

    plt.subplot(2, 4, n + 1)
    plt.plot(x, r2_train, color='blue', label='Training')
    plt.plot(x, r2_test, color='red', label='Test')

    if n == 0:
        plt.ylabel('AVG R2 score')

    plt.title(get_name(names[n]))
    plt.ylim((0.2, 0.7))
    plt.legend()

    plt.subplot(2, 4, n + 5)
    plt.plot(x, rmse_train, color='blue', label='Training')
    plt.plot(x, rmse_test, color='red', label='Test')
    plt.yticks(fontsize=7)

    plt.xlabel('Alpha')
    if n == 0:
        plt.ylabel('AVG RMSE')

    plt.ylim((65000, 103000))
    plt.legend()

    plt.draw()

    if n == 3:
        plt.show()


def plot_pca_results(train_r2_list, test_r2_list, train_error_list, test_error_list):
    """This function plots the PCA results with a bar chart"""

    x = np.arange(1, len(train_r2_list) + 1)                    # PCA components
    width = 0.3                                                 # bar chart width

    # find the best index of the performance
    idx = find_best_index(train_r2_list, test_r2_list, train_error_list, test_error_list)

    plt.subplot(211)
    plt.xticks(x + width / 2, x)
    train = plt.bar(x, train_r2_list, width=width, color='blue', align='center')
    test = plt.bar(x + width, test_r2_list, width=width, color='red', align='center')
    plt.plot(idx + 1 + width / 2, train_r2_list[idx], 'x', color='black', markersize=12)

    plt.xlabel('PCA components')
    plt.ylabel('AVG R2 score')
    plt.legend([train, test], ['Training', 'Test'])

    plt.subplot(212)
    plt.xticks(x + width / 2, x)
    train = plt.bar(x, train_error_list, width=width, color='blue', align='center')
    test = plt.bar(x + width, test_error_list, width=width, color='red', align='center')
    plt.plot(idx + 1 + width / 2, train_error_list[idx], 'x', color='black', markersize=12)

    plt.xlabel('PCA components')
    plt.ylabel('AVG RMSE')
    plt.legend([train, test], ['Training', 'Test'])

    plt.tight_layout()
    plt.show()


def predictions(X, w):
    """This function computes the predictions of the model"""
    preds = np.dot(X, w)

    return preds


def print_pca_results(comp, train_r2_list, train_error_list, test_r2_list, test_error_list, scores):
    """This function prints PCA results both for training and for test set"""

    rmse_train, rmse_test = scores['training']['rmse'], scores['test']['rmse']
    r2_train, r2_test = scores['training']['r2'], scores['test']['r2']

    print('PCA with n_components = ' + str(comp))
    print('TRAINING')
    print('Mean of R2 score: ' + str(train_r2_list[-1]))
    print('Standard deviation of R2 score: ' + str(np.std(r2_train)))
    print()
    print('Mean of RMSE: ' + str(train_error_list[-1]))
    print('Standard deviation of RMSE: ' + str(np.std(rmse_train)))
    print()
    print('TEST')
    print('Mean of R2 score: ' + str(test_r2_list[-1]))
    print('Standard deviation of R2 score: ' + str(np.std(r2_test)))
    print()
    print('Mean of RMSE: ' + str(test_error_list[-1]))
    print('Standard deviation of RMSE: ' + str(np.std(rmse_test)))
    print()
    print()


def print_results(r2_train, r2_test, rmse_train, rmse_test):
    """This function simply prints the mean and the standard deviation of the regression results,
    both for training and for test set"""

    print('TRAINING')
    print('AVG R2: ' + str(np.mean(r2_train)) + '   STD R2: ' + str(np.std(r2_train)))
    print('AVG RMSE: ' + str(np.mean(rmse_train)) + '  STD RMSE: ' + str(np.std(rmse_train)))
    print()
    print('TEST')
    print('AVG R2: ' + str(np.mean(r2_test)) + '   STD R2: ' + str(np.std(r2_test)))
    print('AVG RMSE: ' + str(np.mean(rmse_test)) + '  STD RMSE: ' + str(np.std(rmse_test)))


def print_scores(scores):
    """This function takes in input the cross validated results and prints them"""

    # extract values from the dictionary
    rmse_train, rmse_test = scores['training']['rmse'], scores['test']['rmse']
    r2_train, r2_test = scores['training']['r2'], scores['test']['r2']

    # final results
    print('TRAINING')
    print('Mean of R2 score: ' + str(np.mean(r2_train)))
    print('Standard deviation of R2 score: ' + str(np.std(r2_train)))
    print()
    print('Mean of RMSE: ' + str(np.mean(rmse_train)))
    print('Standard deviation of RMSE: ' + str(np.std(rmse_train)))
    print()
    print('TEST')
    print('Mean of R2 score: ' + str(np.mean(r2_test)))
    print('Standard deviation of R2 score: ' + str(np.std(r2_test)))
    print()
    print('Mean of RMSE: ' + str(np.mean(rmse_test)))
    print('Standard deviation of RMSE: ' + str(np.std(rmse_test)))
    print()


def r2_score(y_true, y_pred):
    """This function takes in input the true and the predicted labels and returns the R2 score measure"""
    ssr = np.sum((y_true - y_pred)**2)                          # residual sum of squares
    sst = np.sum((y_true - np.mean(y_true))**2)                 # total sum of squares
    r2 = 1 - ssr / sst                                          # R2 score

    return r2


def remove_outliers(dataset):
    """This function takes the dataset in input, finds the outliers and returns a "clean" version of the dataset"""

    # find Q1, Q3, and interquartile range for each column
    Q1 = dataset.quantile(q=.25)                                        # first quantile
    Q3 = dataset.quantile(q=.75)                                        # third quantile
    IQR = dataset.apply(stats.iqr)                                      # interquantile range

    # only rows in dataframe that have values within 1.5*IQR of Q1 and Q3 are kept
    dataset_clean = dataset[~((dataset < (Q1 - 1.5 * IQR)) | (dataset > (Q3 + 1.5 * IQR))).any(axis=1)]

    return dataset_clean


def ridge_regression(X, y, alpha):
    """This function computes the ridge regression and returns the parameter vector w"""
    n, m = X.shape
    I = np.identity(m)                                                          # identity matrix of size mxm
    w = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X) + alpha * I), X.T), y)       # w that minimizes the loss function

    return w


def root_mse(y_true, y_pred):
    """This function takes in input the true and the predicted labels and returns the root mean squared error"""

    mse = mean_squared_error(y_true, y_pred)

    return np.sqrt(mse)


def standardizer(feature):
    """This function takes in input a feature vector and returns its standardized version"""

    m = np.mean(feature)
    std = np.std(feature)

    return (feature - m) / std
