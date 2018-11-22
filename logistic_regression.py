import csv
import numpy as np
from numpy import  genfromtxt
import matplotlib.pyplot as plt

def normalize(X):
    '''
    function to normalize feature matrix, X
    '''
    mins = np.min(X, axis = 0)
    maxs = np.max(X, axis = 0)
    rng = maxs - mins
    norm_X = 1 - ((maxs - X)/rng)
    return norm_X


def sigmoid(beta, X):
    return 1.0/(1 + np.exp(-np.dot(X, beta.T)))


def log_gradient(beta, X, y):
    '''
    logistic gradient function
    '''
    first_calc = sigmoid(beta, X) - y.reshape(X.shape[0], -1)
    final_calc = np.dot(first_calc.T, X)
    return final_calc


def compute_cost(beta, X, y):
    '''
    cost function, J
    '''
    log_func_v = sigmoid(beta, X)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1 - y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.mean(final)


def gradient_descent(X, y, beta, lr=.01, converge_change=.001):
    cost = compute_cost(beta, X, y)
    change_cost = 1
    num_iter = 1
    
    while(change_cost > converge_change):
        old_cost = cost
        beta = beta - (lr * log_gradient(beta, X, y))
        cost = compute_cost(beta, X, y)
        change_cost = old_cost - cost
        num_iter += 1
    
    return beta, num_iter 


def plot_reg(X, y, beta):
    '''
    function to plot decision boundary
    '''
    # labelled observations
    x_0 = X[np.where(y == 0.0)]
    x_1 = X[np.where(y == 1.0)]
    
    # plotting points with diff color for diff label
    plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0')
    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1')
    
    # plotting decision boundary
    x1 = np.arange(0, 1, 0.1)
    x2 = -(beta[0,0] + beta[0,1]*x1)/beta[0,2]
    plt.plot(x1, x2, c='k', label='reg line')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

def main():
    dataset = genfromtxt('dataset1.csv', delimiter=',')
    
    # normalizing feature matrix
    X = normalize(dataset[:, :-1])
    
    # stacking columns wth all ones in feature matrix
    X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X))

    # response vector
    y = dataset[:, -1]

    # initializing beta values
    beta = np.matrix(np.zeros(X.shape[1]))

    #new beta values
    beta, num_iter = gradient_descent(X, y, beta)

    print("Estimated regression coefficients:", beta)
    print("No. of iterations:", num_iter)
    
    # plotting regression line
    plot_reg(X, y, beta)

    
if __name__ == "__main__":
    main()
    
