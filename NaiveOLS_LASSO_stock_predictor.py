import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def normalize_data(x,min,max):
    return (x-min)/(max-min)

df = pd.read_csv("./resources/2018.csv")
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


np.random.seed(2018)
y_train_from_random = np.random.uniform(1,0,len(y_train))

y_test_from_random = np.random.uniform(1,0,len(y_test_true))



###TODO Naive OLS, LASSO Regression and Ridge Regression

def model_OLS(data_X_train, data_y_train, data_X_val, data_y_val_true, data_X_test):
    linreg = linear_model.LinearRegression(fit_intercept=True)  # can change with or without intercept
    linreg.fit(data_X_train, data_y_train)
    linreg.get_params()
    y_train_from_OLS = linreg.predict(data_X_train)

    linreg.fit(data_X_val,data_y_val_true)
    y_test_from_OLS = linreg.predict(data_X_test)
    return y_train_from_OLS, y_test_from_OLS

def model_Lassi(data_X_train, data_y_train, a, data_X_val):
    Lassi = Lasso(alpha=a, tol=1e-5)
    Lassi.fit(data_X_train, data_y_train)
    y_train_from_LASSO = Lassi.predict(data_X_train)

    y_val_from_LASSO = Lassi.predict(data_X_val)

    return y_train_from_LASSO, y_val_from_LASSO


def model_Rachel(data_X_train, data_y_train, a, data_X_val):
    Rachel = Ridge(alpha=a, tol=1e-5)
    Rachel.fit(data_X_train, data_y_train)
    y_train_from_Ridge = Rachel.predict(data_X_train)

    y_val_from_Ridge = Rachel.predict(data_X_val)

    return y_train_from_Ridge, y_val_from_Ridge


def confusion_matrix_for_reg(predict, real):
    predict = np.array(np.round(predict))
    real = np.array(real)

    ## use AND gate to check for TP
    TP = np.sum(np.logical_and(predict, real))
    ## sum of all 1s in predict - TP = FP
    FP = np.sum(predict ==1) - TP


    ## firat flip predict and real 1 and 0s, and use the same AND gate to check for TN
    TN = np.sum(np.logical_and(np.logical_not(predict), np.logical_not(real)))
    FN = np.sum(predict ==0) - TN



    conf_matrix = [[TN, FP], [FN, TP]]


    return conf_matrix


## TODO Naive OLS

y_train_from_OLS, y_test_from_OLS = model_OLS(X_train, y_train, X_val, y_val_true, X_test)

conf_matrix_train_from_OLS = confusion_matrix_for_reg(y_train_from_OLS, y_train)

conf_matrix_test_from_OLS = confusion_matrix_for_reg(y_test_from_OLS, y_test_true)


## TODO Optimizing LASSO and Ridge Lambda

## initialising
x_axis_interval = np.arange(0,0.7, 1e-3)


conf_matrix_train_from_LASSO = []
FP_train_LASSO =[]

conf_matrix_val_from_LASSO = []
FP_val_LASSO =[]

conf_matrix_train_from_Ridge = []
FP_train_Ridge =[]

conf_matrix_val_from_Ridge =[]
FP_val_Ridge = []


## finding optimal lambda for the least amount of False Positive
for a in x_axis_interval:

    y_train_from_LASSO, y_val_from_LASSO = model_Lassi(X_train, y_train, a, X_val)

    conf_matrix_train_from_LASSO.append(confusion_matrix_for_reg(y_train_from_LASSO, y_train))
    FP_train_LASSO.append(confusion_matrix_for_reg(y_train_from_LASSO, y_train)[1][0])


    conf_matrix_val_from_LASSO.append(confusion_matrix_for_reg(y_val_from_LASSO, y_val_true))
    FP_val_LASSO.append(confusion_matrix_for_reg(y_val_from_LASSO, y_val_true)[1][0])

    y_train_from_Ridge, y_val_from_Ridge = model_Rachel(X_train, y_train, a, X_val)

    conf_matrix_train_from_Ridge.append(confusion_matrix_for_reg(y_train_from_Ridge, y_train))
    FP_train_Ridge.append(confusion_matrix_for_reg(y_train_from_Ridge, y_train)[1][0])

    conf_matrix_val_from_Ridge.append(confusion_matrix_for_reg(y_val_from_Ridge, y_val_true))
    FP_val_Ridge.append(confusion_matrix_for_reg(y_val_from_Ridge, y_val_true)[1][0])


# Plotting False positives for each lambda
plt.scatter(x_axis_interval,FP_train_LASSO, label="LASSO train", s=0.3)
plt.scatter(x_axis_interval, FP_val_LASSO, label="LASSO val", s=0.3)
plt.scatter(x_axis_interval,FP_train_Ridge, label="Ridge train", s=0.3)
plt.scatter(x_axis_interval,FP_val_Ridge, label="Ridge val", s=0.3)

plt.xlabel('tuning parameter lambda')
plt.ylabel('FP of train and val data fitting')
plt.legend()
plt.show()

a_range = x_axis_interval

## Only optimize for minimum FP, long only fund
opt_lambda_LASSO = a_range[FP_val_LASSO.index(min(FP_val_LASSO))]
opt_lambda_Ridge = a_range[FP_val_Ridge.index(min(FP_val_Ridge))]

## predicting test data set with optimal lambda/ alpha, returning confusion matrix

Lassi_opt = Lasso(alpha=opt_lambda_LASSO, tol=1e-5)
Lassi_opt.fit(X_val, y_val_true)
y_test_from_Lassi = Lassi_opt.predict(X_test)

conf_matrix_test_Lassi = confusion_matrix_for_reg(y_test_from_Lassi,y_test_true)


Rachel_opt = Ridge(alpha=opt_lambda_Ridge, tol=1e-5)
Rachel_opt.fit(X_val, y_val_true)
y_test_from_Rachel = Rachel_opt.predict(X_test)

conf_matrix_test_Rachel = confusion_matrix_for_reg(y_test_from_Rachel, y_test_true)




print("optimal lambda/ tuning parameters for LASSO:", opt_lambda_LASSO,"\n","optimal lambda/ tuning parameters for Ridge:", opt_lambda_Ridge)


print("OLS's confusion matrix of training:", conf_matrix_train_from_OLS,"\n",
      "OLS's confusion matrix of testing:", conf_matrix_test_from_OLS,"\n",
      "LASSO's confusion matrix of testing", conf_matrix_test_Lassi,"\n",
      "Ridge's confusion matrix of testing", conf_matrix_test_Rachel)


## TODO Performance metric, calculating precision

## first get confusion matrix of randomly picking

conf_matrix_train_from_random = confusion_matrix_for_reg(y_train_from_random, y_train)
conf_matrix_test_from_random = confusion_matrix_for_reg(y_test_from_random, y_test_true)

conf_matrix_list = [conf_matrix_train_from_random, conf_matrix_test_from_random,
                    conf_matrix_train_from_OLS, conf_matrix_test_from_OLS,
                    conf_matrix_test_Lassi, conf_matrix_test_Rachel]


prec_list = []

for i in conf_matrix_list:
    prec_list.append(i[1][1]/ (i[0][1] + i[1][1]))

print(prec_list)

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
