import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt



results = pd.read_csv("./resources/results.csv")

linear = results.iloc[np.where(results['algo']=='linear')]
rbf = results.iloc[np.where(results['algo']=='rbf')]
sigmoid = results.iloc[np.where(results['algo']=='sigmoid')]

linear['precision_score'] = linear['test_precision']+linear['train_precision']
rbf['precision_score'] = rbf['test_precision']+rbf['train_precision']
sigmoid['precision_score'] = sigmoid['test_precision']+sigmoid['train_precision']

linear['recall_score'] = linear['test_recall']+linear['train_recall']
rbf['recall_score'] = rbf['test_recall']+rbf['train_recall']
sigmoid['recall_score'] = sigmoid['test_recall']+sigmoid['train_recall']

linear['score'] = linear['precision_score']+linear['recall_score']
rbf['score'] = rbf['precision_score']+rbf['recall_score']
sigmoid['score'] = sigmoid['precision_score']+sigmoid['recall_score']


linear['f1_train'] = 2*((linear['train_precision']*linear['train_recall'])/(linear['train_precision']+linear['train_recall']))
linear['f1_test'] = 2*((linear['test_precision']*linear['test_recall'])/(linear['test_precision']+linear['test_recall']))

rbf['f1_train'] = 2*((rbf['train_precision']*rbf['train_recall'])/(rbf['train_precision']+rbf['train_recall']))
rbf['f1_test'] = 2*((rbf['test_precision']*rbf['test_recall'])/(rbf['test_precision']+rbf['test_recall']))

sigmoid['f1_train'] = 2*((sigmoid['train_precision']*sigmoid['train_recall'])/(sigmoid['train_precision']+sigmoid['train_recall']))
sigmoid['f1_test'] = 2*((sigmoid['test_precision']*sigmoid['test_recall'])/(sigmoid['test_precision']+sigmoid['test_recall']))

linear['f1'] = linear['f1_train']+linear['f1_test']
rbf['f1'] = rbf['f1_train']+rbf['f1_test']
sigmoid['f1'] = sigmoid['f1_train']+sigmoid['f1_test']


linear_best_precision = linear.loc[linear['precision_score'].idxmax()]
rbf_best_precision = rbf.loc[rbf['precision_score'].idxmax()]
sigmoid_best_precision = sigmoid.loc[sigmoid['precision_score'].idxmax()]

linear_best_recall = linear.loc[linear['recall_score'].idxmax()]
rbf_best_recall = rbf.loc[rbf['recall_score'].idxmax()]
sigmoid_best_recall = sigmoid.loc[sigmoid['recall_score'].idxmax()]

linear_best_score = linear.loc[linear['score'].idxmax()]
rbf_best_score = rbf.loc[rbf['score'].idxmax()]
sigmoid_best_score = sigmoid.loc[sigmoid['score'].idxmax()]

linear_best_f1_test = linear.loc[linear['f1_test'].idxmax()]
rbf_best_f1_test = rbf.loc[rbf['f1_test'].idxmax()]
sigmoid_best_f1_test = sigmoid.loc[sigmoid['f1_test'].idxmax()]

linear_best_f1_train = linear.loc[linear['f1_train'].idxmax()]
rbf_best_f1_train = rbf.loc[rbf['f1_train'].idxmax()]
sigmoid_best_f1_train = sigmoid.loc[sigmoid['f1_train'].idxmax()]

linear_best_f1 = linear.loc[linear['f1'].idxmax()]
rbf_best_f1 = rbf.loc[rbf['f1'].idxmax()]
sigmoid_best_f1 = sigmoid.loc[sigmoid['f1'].idxmax()]


# print("BEST PRECISIONS")
# print("Linear------\n"+str(linear_best_precision))
# print("Rbf------\n" +str(rbf_best_precision))
# print("Sigmoid------\n" +str(sigmoid_best_precision))
#
# print("BEST RECALL")
# print("Linear------\n" +str(linear_best_recall))
# print("Rbf------\n"+str(rbf_best_recall))
# print("Sigmoid------\n"+str(sigmoid_best_recall))
#
# print("BEST SCORE")
# print("Linear------\n" +str(linear_best_score))
# print("Rbf------\n"+str(rbf_best_score))
# print("Sigmoid------\n"+str(sigmoid_best_score))

print("BEST F1 TRAIN")
print("Linear------\n" +str(linear_best_f1_train))
print("Rbf------\n"+str(rbf_best_f1_train))
print("Sigmoid------\n"+str(sigmoid_best_f1_train))

print("\n\n")

print("BEST F1 TEST")
print("Linear------\n" +str(linear_best_f1_test))
print("Rbf------\n"+str(rbf_best_f1_test))
print("Sigmoid------\n"+str(sigmoid_best_f1_test))

print("\n\n")

print("BEST F1")
print("Linear------\n" +str(linear_best_f1))
print("Rbf------\n"+str(rbf_best_f1))
print("Sigmoid------\n"+str(sigmoid_best_f1))

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(linear['C'], linear['gamma'], linear['f1_train'], label='Train f1')
ax.scatter(linear['C'], linear['gamma'], linear['f1_test'], label='Test f1')
ax.legend()
ax.set_title("Linear Kernel")
ax.set_xlabel('C')
ax.set_ylabel('gamma')
ax.set_zlabel('F1 score')

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.scatter(rbf['C'], rbf['gamma'], rbf['f1_train'], label='Train f1')
ax1.scatter(rbf['C'], rbf['gamma'], rbf['f1_test'], label='Test f1')
ax1.legend()
ax1.set_title("Rbf Kernel")
ax1.set_xlabel('C')
ax1.set_ylabel('gamma')
ax1.set_zlabel('F1 Precision')

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.scatter(sigmoid['C'], sigmoid['gamma'], sigmoid['f1_train'], label='Train f1')
ax2.scatter (sigmoid['C'], sigmoid['gamma'], sigmoid['f1_test'], label='Test f1')
ax2.legend()
ax2.set_title("Sigmoid Kernel")
ax2.set_xlabel('C')
ax2.set_ylabel('gamma')
ax2.set_zlabel('F1')

plt.show()
