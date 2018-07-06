import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import fn_helpers
import matplotlib.pyplot as plt

def normalize_data(x,min,max):
    return (x-min)/(max-min)

df = pd.read_csv("companylist1.csv")
df = df.dropna()

nyse_previous_close = 12412.07
nyse_one_year_close = 11812.20

nyse_performance = math.log(nyse_previous_close/nyse_one_year_close)

# print(nyse_performance)

df = df.infer_objects()
df['last_price'] = df['last_price'].astype('float')

#print(df.dtypes)

df['performance'] = np.log(df['last_price'] /df['previous_year_price'])
df['over/under-perfomance'] = df['performance']>nyse_performance
df['over/under-perfomance'] = df['over/under-perfomance'].astype('int')

#normalize data
df[['cash_ratio','return_to_equity','price_to_book','pe','short_interest_ratio','debt_to_equity','eps']] = normalize(df[['cash_ratio','return_to_equity','price_to_book','pe','short_interest_ratio','debt_to_equity','eps']],axis=0, norm='max')
df.dropna()     # need to dropnan else can't model


cr = normalize_data(df['cash_ratio'],min(df['cash_ratio']),max(df['cash_ratio']))

# set random seed for folding
np.random.seed(2018)
df['fold'] = np.random.uniform(1,10,len(df))

id_train = np.where(df['fold']<=6.5)[0]
id_val = np.intersect1d(np.where(df['fold'] > 6.5)[0],np.where(df['fold'] <= 9)[0])
id_test = np.where(df['fold']>9)[0]

n_train = len(id_train)
n_val = len(id_val)
n_test = len(id_test)



X = df[['cash_ratio','return_to_equity','price_to_book','pe','short_interest_ratio','debt_to_equity','eps']]#'price_to_book','pe','short_interest_ratio','debt_to_equity','eps']]
y = df['over/under-perfomance']







## Prepare train data set, droping Nan
X_train = X.iloc[id_train]
X_train = X_train.dropna()


y_train = y.iloc[id_train]
y_train = y_train.dropna()



## Prepare validation data set
X_val = X.iloc[id_val]
X_val = X_val.dropna()


y_val_true = y.iloc[id_val]
y_val_true = y_val_true.dropna()



## Prepare test data set, droping Nan
X_test = X.iloc[id_test]
X_test = X_test.dropna()

y_test_true = y.iloc[id_test]
y_test_true = y_test_true.dropna()

##TODO Randomly picking, see if the data set is balanced or not
error = 1 - np.divide(np.sum(y_test_true),len(y_test_true))
print('Random guessing error:', error)
##TODO Naive OLS Regression

## Fit Linear Regression OLS to train set
linreg = linear_model.LinearRegression(fit_intercept=True)  # can change with or without intercept
linreg.fit(X_train, y_train)
linreg.get_params()


## Predicting from X_test set, calculate MSE score
y_test_from_linreg = linreg.predict(X_test)
MSE_linear = mean_squared_error(y_test_true,y_test_from_linreg)

#  MSE score without intercept, 0.6015010280065605
#  MSE score with intercept, 0.2609995959038271



###TODO LASSO Regression and Ridge Regression

MSE_LASSO_models_train =[]
MSE_Ridge_models_train =[]
MSE_LASSO_models_val =[]
MSE_Ridge_models_val =[]


x_axis_interval = np.arange(0,0.7, 1e-6)

## finding optimal lambda for the least MSE
for a in x_axis_interval:
    Lassi = Lasso(alpha =a)
    Rachel = Ridge(alpha=a)
    Lassi.fit(X_train, y_train)
    Rachel.fit(X_train, y_train)
    # Lasso.get_params()

    ## Predict from train and val set
    y_train_from_LASSO = Lassi.predict(X_train)
    y_train_from_Ridge = Rachel.predict(X_train)

    y_val_from_LASSO = Lassi.predict(X_val)
    y_val_from_Ridge = Rachel.predict(X_val)

    mse_LASSO_train = mean_squared_error(y_train,y_train_from_LASSO)
    MSE_LASSO_models_train.append(mse_LASSO_train)

    mse_LASSO_val = mean_squared_error(y_val_true, y_val_from_LASSO)
    MSE_LASSO_models_val.append(mse_LASSO_val)

    mse_Ridge_train = mean_squared_error(y_train, y_train_from_Ridge)
    MSE_Ridge_models_train.append(mse_Ridge_train)

    mse_Ridge_val = mean_squared_error(y_val_true, y_val_from_Ridge)
    MSE_Ridge_models_val.append(mse_Ridge_val)

plt.scatter(x_axis_interval,MSE_LASSO_models_train, label="LASSO train", s=0.3)
plt.scatter(x_axis_interval,MSE_LASSO_models_val, label="LASSO val", s=0.3)
plt.scatter(x_axis_interval,MSE_Ridge_models_train, label="Ridge train", s=0.3)
plt.scatter(x_axis_interval,MSE_Ridge_models_val, label="Ridge val", s=0.3)

plt.xlabel('tuning parameter lambda')
plt.ylabel('MSE of train and val data fitting')
plt.legend()
plt.show()

a_range = x_axis_interval


opt_lambda_LASSO = a_range[MSE_LASSO_models_val.index(min(MSE_LASSO_models_val))]
opt_lambda_Ridge = a_range[MSE_Ridge_models_val.index(min(MSE_Ridge_models_val))]

## predicting test data set with optimal lambda/ alpha

Lassi_opt = Lasso(alpha=opt_lambda_LASSO)
Lassi_opt.fit(X_val, y_val_true)
y_test_from_Lassi = Lassi_opt.predict(X_test)
test_MSE_Lassi = mean_squared_error(y_test_true, y_test_from_Lassi)


Rachel_opt = Ridge(alpha=opt_lambda_Ridge)
Rachel_opt.fit(X_val, y_val_true)
y_test_from_Rachel = Rachel_opt.predict(X_test)
test_MSE_Rachel = mean_squared_error(y_test_true, y_test_from_Rachel)



print("optimal lambda/ tuning parameters for LASSO:", opt_lambda_LASSO,"\n","optimal lambda/ tuning parameters for Ridge:", opt_lambda_Ridge)


print("MSE Linear of testing:", MSE_linear,"\n", "MSE LASSO of testing", test_MSE_Rachel,"\n", "MSE Ridge of testing", test_MSE_Rachel)




# SST = np.sum((y_test-y[id_test])**2)
# print(SST)


# clf = svm.SVC(kernel="linear")
# clf.fit(X.iloc[id_train],y.iloc[id_train])
# train_pred = clf.predict(X.iloc[id_train])
# val_pred = clf.predict(X.iloc[id_val])
# test_pred = clf.predict(X.iloc[id_test])
#
# for (intercept, coef) in zip(clf.intercept_, clf.coef_):
#     s = "y = {0:.3f}".format(intercept)
#     for (i, c) in enumerate(coef):
#         s += " + {0:.3f} * x{1}".format(c, i)
#
#     print(s)
#
# print(test_pred)
