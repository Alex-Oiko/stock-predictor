import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

def fn_print_coefficients(model):    
    # Get the degree of the polynomial
    deg = len(model.coef_)

    # Get learned parameters as a list
    w = list(model.coef_)
    w.insert(0, model.intercept_)

    print ('Learned polynomial for degree ' + str(deg) + ':')
    w.reverse()
    print (np.poly1d(w))


def fn_plot_predictions(data, model):

    # Get the degree of the polynomial
    deg = len(model.coef_)
    
    # Create 200 points in the x axis and compute the predicted value for each point
    x_pred = pd.DataFrame({'x_1':[i/200.0 for i in range(200)]})
    for idx in range(2, deg + 1):  
        colname = 'x_%d'%idx     
        x_pred[colname] = x_pred['x_1']**idx
        
    y_pred = model.predict(x_pred)
    
    # plot predictions
    fig = plt.figure(figsize=(10, 14))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(data.iloc[0:30]['x_1'], data.iloc[0:30]['y'], 'k.')
    ax1.set_xlabel('x_1', fontsize=20)
    ax1.set_ylabel('y', fontsize=20)
    ax1.plot(x_pred['x_1'], 
             y_pred, 'r-', 
             label='degree ' + str(deg) + ' fit',
             linewidth=5)
    ax1.legend(loc='upper left')
    ax1.axis([0,1,-1.5,2])
    ax1.set_title('training data', fontsize=24)
    
    # plot predictions
    ax2.plot(data.iloc[30:]['x_1'], data.iloc[30:]['y'], 'k.')
    ax2.set_xlabel('x_1', fontsize=20)
    ax2.set_ylabel('y', fontsize=20)
    ax2.plot(x_pred['x_1'], 
             y_pred, 'r-', 
             label='degree ' + str(deg) + ' fit',
             linewidth=5)
    ax2.legend(loc='upper left')
    ax2.axis([0,1,-1.5,2])
    ax2.set_title('test data', fontsize=24)
    