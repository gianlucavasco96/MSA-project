import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

import functions as f

# house pricing dataset loading and reading
dataset_path = './dataset/'
dataset_name = 'cal-housing.csv'
dataset = pd.read_csv(dataset_path + dataset_name)

# histogram to get the distributions of the different variables
dataset.hist(bins=70, figsize=(20, 20))
plt.show()

# count the values of the columns
print(dataset.count())
print()

# first problem: total_bedrooms feature contains missing values --> median value substitution
tot_bedrooms = dataset['total_bedrooms']

# plot the boxplot to get info about the variable distribution
plt.figure()
sb.boxplot(tot_bedrooms)
plt.show()

tot_bedrooms_clear = tot_bedrooms[~np.isnan(tot_bedrooms)]          # removing NaN values

median = np.median(tot_bedrooms_clear)                              # computing the median of the distribution
tot_bedrooms.fillna(median, inplace=True)                           # replacing NaN values with median value
dataset['total_bedrooms'] = tot_bedrooms                            # assigning fixed feature column to dataset

# second problem: ocean_proximity has categorical values --> one-hot encoding
drop = True

if drop:
    dataset_drop = f.one_hot_encoding(dataset, drop=drop)

    # put the label column at the end of the dataset
    label = dataset_drop['median_house_value']
    dataset_drop.pop('median_house_value')
    dataset_drop['median_house_value'] = label

    # show the correlation between variables
    plt.figure(figsize=(15, 8))
    corr = dataset_drop.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sb.heatmap(abs(corr), linewidths=.5, annot=True, mask=mask, cmap='coolwarm')
    plt.show()

    # save not standardized version of the dataset
    dataset_drop.to_csv(dataset_path + 'not_standardized_dataset_drop.csv', index=False)

    # dataset standardization
    feat_to_skip = ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND', 'median_house_value']

    for key in dataset_drop.keys():
        if key in feat_to_skip:                                 # one of k features and labels don't have to
            continue                                            # be normalized
        dataset_drop[key] = f.standardizer(dataset_drop[key])   # standardization of each feature

    dataset_drop['median_house_value'] = label                  # put back label column (this is not normalized)

    # save standardized version of the dataset
    dataset_drop.to_csv(dataset_path + 'standardized_dataset_drop.csv', index=False)
else:
    dataset = f.one_hot_encoding(dataset, drop=drop)

    # put the label column at the end of the dataset
    label = dataset['median_house_value']
    dataset.pop('median_house_value')
    dataset['median_house_value'] = label

    # show the correlation between variables
    plt.figure(figsize=(15, 15))
    corr = dataset.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sb.heatmap(abs(corr), linewidths=.5, annot=True, mask=mask, cmap='coolwarm')
    plt.show()

    # save not normalized version of the dataset
    dataset.to_csv(dataset_path + 'not_standardized_dataset.csv', index=False)

    # dataset standardization
    feat_to_skip = ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND', 'median_house_value']

    for key in dataset.keys():
        if key in feat_to_skip:                                     # one of k features and labels don't have to
            continue                                                # be normalized
        dataset[key] = f.standardizer(dataset[key])                 # standardization of each feature

    dataset['median_house_value'] = label                           # put back label column (this is not normalized)

    # save normalized version of the dataset
    dataset.to_csv(dataset_path + 'standardized_dataset.csv', index=False)
