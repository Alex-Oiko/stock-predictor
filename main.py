import pandas as pd
import math
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv("./resources/2018.csv")

df = df.dropna()
df = df.reset_index(drop=True)
print(df.shape)
#check if you could use the nasdaq close as well for the corresponding companies
nyse_previous_close = 12412.07
nyse_one_year_close = 11812.20

nyse_performance = math.log(nyse_previous_close/nyse_one_year_close)

print(nyse_performance)

df = df.infer_objects()
df['last_price'] = df['last_price'].astype('float')

df['performance'] = np.log(df['last_price'] /df['previous_year_price'])
df['over/under-perfomance'] = df['performance']>nyse_performance
df['over/under-perfomance'] = df['over/under-perfomance'].astype('int')


X = df[['cash_ratio','return_to_equity','price_to_book','pe','short_interest_ratio','debt_to_equity','eps']]
y = df['over/under-perfomance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,shuffle=True,stratify=y)


clf = svm.SVC(kernel="linear")
clf.fit(X_train,y_train)
train_pred = clf.predict(X_train)
print(sum(train_pred!=0))
print(sum(train_pred==0))
#print(clf.decision_function(X.iloc[id_train]))

# for (intercept, coef) in zip(clf.intercept_, clf.coef_):
#     s = "y = {0:.3f}".format(intercept)
#     for (i, c) in enumerate(coef):
#         s += " + {0:.3f} * x{1}".format(c, i)
#
#     print(s)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# n = 100
#
# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# xs = df['cash_ratio']
# ys = df['return_to_equity']
# zs = y
# ax.scatter(xs, ys, zs)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()
#
# #val_pred = clf.predict(X.iloc[id_val])
# #test_pred = clf.predict(X.iloc[id_test])
#
#
#
#
