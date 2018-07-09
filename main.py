import pandas as pd
import math
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def product_metrics(real, pred):
    print("Non 0 results " + str(sum(real != 0)) + " ,0 results " + str(sum(pred == 0)))
    C = confusion_matrix(real, pred)
    tn, fp, fn, tp = C[0, 0], C[0, 1], C[1, 0], C[1, 1];
    precision = float(tp) / float((tp + fp))
    recall = float(tp) / float((tp + fn))
    print("Precision " + str(precision) + " Recall " + str(recall))


df = pd.read_csv("./resources/2018.csv")

df = df.dropna()
df = df.reset_index(drop=True)

nyse_previous_close = 12412.07
nyse_one_year_close = 11812.20

nasdaq_previous_close = 7712.9502
nasdaq_one_year_close = 6233.9502

nasdaq_begin_index = 977

nyse_performance = math.log(nyse_previous_close/nyse_one_year_close)
nasdaq_performance = math.log(nasdaq_previous_close/nasdaq_one_year_close)

df = df.infer_objects()
df['last_price'] = df['last_price'].astype('float')
df['performance'] = np.log(df['last_price'] /df['previous_year_price'])
# status is under/overperformance
df['status'] = 0
nyse_performance = (df.loc[range(0,nasdaq_begin_index),'performance']>nyse_performance).astype('int')
nasdaq_performance = (df.loc[range(nasdaq_begin_index,len(df)),'performance']>nasdaq_performance).astype('int')
df['status'] = nyse_performance
df.loc[nasdaq_begin_index:len(df),'status'] = nasdaq_performance
df['status'] = df['status'].astype('int')

X = df[['cash_ratio','return_to_equity','price_to_book','pe','short_interest_ratio','debt_to_equity','eps']]
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,shuffle=True,stratify=y)

print(X_train.shape)
print(X_test.shape)
clf = svm.SVC(kernel="linear")
clf.fit(X_train,y_train)

for (intercept, coef) in zip(clf.intercept_, clf.coef_):
     s = "y = {0:.3f}".format(intercept)
     for (i, c) in enumerate(coef):
         s += " + {0:.3f} * x{1}".format(c, i)
     print(s)

train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)

print(product_metrics(y_train,train_pred))
print(product_metrics(y_test,test_pred))