import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn import linear_model

def normalize_data(x,min,max):
    return (x-min)/(max-min)

df = pd.read_csv("companylist1.csv")
df = df.dropna()

nyse_previous_close = 12412.07
nyse_one_year_close = 11812.20

nyse_performance = math.log(nyse_previous_close/nyse_one_year_close)

print(nyse_performance)

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


df['fold'] = np.random.uniform(1,10,len(df))

id_train = np.where(df['fold']<=6.5)[0]
id_val = np.intersect1d(np.where(df['fold'] > 6.5)[0],np.where(df['fold'] <= 9)[0])
id_test = np.where(df['fold']>9)[0]

n_train = len(id_train)
n_val = len(id_val)
n_test = len(id_test)



X = df[['cash_ratio','return_to_equity','price_to_book','pe','short_interest_ratio','debt_to_equity','eps']]#'price_to_book','pe','short_interest_ratio','debt_to_equity','eps']]
y = df['over/under-perfomance']




linreg = linear_model.LinearRegression(fit_intercept=True)
print(y.dropna)

X_train = X[id_train]
X_train = X_train.dropna

y_train = y[id_train]
y_train = y_train.dropna

linreg.fit(X_train, y_train)
linreg.get_params()
y_test = linreg.predict(X[id_test])
SST = np.sum((y_test-y[id_test])**2)


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
