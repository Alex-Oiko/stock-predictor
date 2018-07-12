import pandas as pd
import math
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def product_metrics(real, pred):
    print("Non 0 results " + str(sum(pred != 0)) + " ,0 results " + str(sum(pred == 0)))
    C = confusion_matrix(real, pred)
    tn, fp, fn, tp = C[0, 0], C[0, 1], C[1, 0], C[1, 1];
    try:
        precision = float(tp) / float((tp + fp))
    except(ZeroDivisionError):
        precision = 0
    try:
        recall = float(tp) / float((tp + fn))
    except(ZeroDivisionError):
        recall = 0
    print("Precision " + str(precision) + " Recall " + str(recall))
    return precision,recall

def run_model(C,gamma,algo):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12, shuffle=True, stratify=y)
    clf = svm.SVC(kernel=algo,C=C,gamma=gamma)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    train_precision, train_recall = product_metrics(y_train, train_pred)
    test_precision, test_recall = product_metrics(y_test, test_pred)

    return C,gamma,algo,train_precision,train_recall,test_precision,test_recall


#def hyper_parameterization()


df = pd.read_csv("./resources/2018.csv")

results = pd.DataFrame(columns=['C','gamma','algo','test_precision','test_recall','train_precision','train_recall'],index=range(305))

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
print(df['status'].value_counts())

X = df[['cash_ratio','return_to_equity','price_to_book','pe','short_interest_ratio','debt_to_equity','eps']]
y = df['status']


gammas = [2**x for x in range(-10,10,2)]
Cs = [2**x for x in range(-10,10,2)]
algos = ['linear','rbf','sigmoid']
seeds = range(10,100,10)

#counter = 0
#for algo in algos:
#    for c in Cs:
#        for gamma in gammas:
#            results.iloc[counter] = run_model(c,gamma,algo)
#            print(results.iloc[counter])
#            counter = counter+1;

#results.to_csv('./resources/results.csv', sep=',')


