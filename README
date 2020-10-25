# MSA-project
Ridge regression algorithm implementation for housing prices prediction

The aim of this project is to present a scratch implementation of the ridge regression algorithm and to analyse it in different settings, comparing all the performances.
For this purpose, the [California housing dataset](https://www.dropbox.com/s/zxv6ujxl8kmijfb/cal-housing.csv?dl=0) has been used and more regression tasks have been considered, 
in order to find the best predictor in every context of application.

## Project structure
The project has been structured as follows:
- Dataset preprocessing
- Ridge regression algorithm implementation
- Predictors comparison in different settings:
  - Simple ridge regression
  - Ridge regression with saturated values removal
  - Ridge regression with outliers removal
  - Ridge regression with k-folds cross validation
  - Ridge regression with cross validated risk estimate analysis with respect to the ridge parameter alpha
  - Ridge regression with nested cross validation
  - Ridge regression with outliers removal, nested cross validation and PCA

For each of these application contexts, four datasets have been analysed and compared:
- Standardized
- Non-standardized
- Standardized with variables dropping
- Non-standardized with variables dropping

After a preprocessing phase, the overfitting and the multicollinearity problem have been analysed, trying to improve the ridge regression results.
The dependence of the cross-validated risk estimate on the regularization term has been analysed and then, with nested cross validation, the hyperparameter $ \alpha $ has been tuned. 
Finally, dimensionality reduction (PCA) has been added to the best experiment, in order to improve the risk estimate.
