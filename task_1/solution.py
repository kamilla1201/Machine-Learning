#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    x_train, y_train = get_data("PA1_train.csv", normalize=True)
    x_dev, y_dev = get_data("PA1_dev.csv", normalize=True)
    print "----------------------------------------------"
    print "Part 1.a - training data"
    print "----------------------------------------------"
    num_iterations= 200000
    alphas=[1e-5,1e-6,1e-7]
    first_part(x_train, y_train, alphas, num_iterations)
    print "----------------------------------------------"

    print "----------------------------------------------"
    print "Part 1.b - validation data"
    print "----------------------------------------------"
    alphas=[1e-5]
    first_part(x_dev, y_dev, alphas, num_iterations)
    print "----------------------------------------------"

    print "----------------------------------------------"
    print "Part 2.a - training data"
    print "----------------------------------------------"
    alpha=1e-5
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    second_part(x_train, y_train, alpha, lambdas, num_iterations)
    print "----------------------------------------------"

    print "----------------------------------------------"
    print "Part 2.b - validation data"
    print "----------------------------------------------"
    alpha=1e-5
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    second_part(x_dev, y_dev, alpha, lambdas, num_iterations)
    print "----------------------------------------------"

    x_train, y_train = get_data("PA1_train.csv", normalize=False)
    x_dev, y_dev = get_data("PA1_dev.csv", normalize=False)
    num_iterations= 10000
    alphas=[1, 0, 1e-3, 1e-6, 1e-9, 1e-15] # 1, 0, 10 −3 , 10 −6 , 10 −9 , 10 −15
    print "----------------------------------------------"
    print "Part 3 - training data"
    print "----------------------------------------------"
    first_part(x_train, y_train, alphas, num_iterations)
    print "----------------------------------------------"

    print "----------------------------------------------"
    print "Part 3 - validation data"
    print "----------------------------------------------"
    first_part(x_dev, y_dev, alphas, num_iterations)
    print "----------------------------------------------"
    x_train, y_train = get_data("PA1_train.csv", normalize=True)
    print "----------------------------------------------"
    print "Predicting prices"
    print "----------------------------------------------"
    x_test, tmp = get_data("PA1_test.csv", normalize=True, test=True)
    num_iterations= 200000
    alphas=[1e-5]
    theta = first_part(x_train, y_train, alphas, num_iterations)
    print np.dot(x_test, np.matrix(theta).T)
    np.savetxt('out.txt', np.dot(x_test, np.matrix(theta).T))


def get_data(filename, normalize, test=False):

    x = pd.read_csv(filename)
# -------- 0.a and 0.b ----------
    y = None
    x[['month', 'day', 'year']] = x['date'].str.split('/', expand=True).astype(int)
    del x['id'], x['date']
    if not test:
        y = x['price']
        del x['price']
        y = y.astype(float)
    x = x.astype(float)
# ------------ 0.c --------------
#print (x['waterfront'].value_counts()/x['waterfront'].count())*100
#print (x['grade'].value_counts()/x['grade'].count())*100
#print (x['condition'].value_counts()/x['condition'].count())*100
#x.describe(include='all')
# --------- 0.e ---------
    if normalize:
        for c in x.columns.difference(['dummy']):
            x[c] = (x[c] - x[c].min()) / (x[c].max()-x[c].min())

    return x, y

# --------- 1.a ---------
# m denotes the number of examples here, not the number of features
def linear_regression(x, y, theta, alpha, m, numIterations, lambda_):
    costs = []
    print "============================================================="
    print "Learning rate is %s; Lambda is %s" % (alpha, lambda_)
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2)
        costs.append(cost)

        # avg gradient per example
        gradient = np.dot(x.T, loss)
        # update
        norm = np.linalg.norm(gradient)
        if i % 1000 == 0:
            print("Iteration %d | Cost: %f | Norm %s" % (i, cost, norm))
        theta = theta - alpha * gradient
        if norm < 0.5:
            print "Converged on %s iteration" % i
            break
        if np.isnan(cost) or np.isnan(norm):
            print "Iteration %d diverged!" % i
            break
    print theta
    print "============================================================="
    return theta, costs

def l2_reg(x, y, theta, alpha, m, numIterations, lambda_):
    costs = []
    print "============================================================="
    print "Learning rate is %s; Lambda is %s" % (alpha, lambda_)
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = (np.sum(loss ** 2) / 2) + ((lambda_/2) * np.sum(theta[1:] ** 2))
        costs.append(cost)

        # avg gradient per example
        gradient = np.dot(x.T, loss)
        # update
        norm = np.linalg.norm(gradient)
        if i % 1000 == 0:
            print("Iteration %d | Cost: %f | Norm %s" % (i, cost, norm))
        theta[0] = theta[0] - alpha * gradient[0]
        theta[1:] = theta[1:] - alpha * (gradient[1:] + lambda_ * theta[1:])
        #theta[1:] = (1-2*alpha*lambda_)*theta[1:] - 2*alpha * gradient[1:]
        if norm < 0.5:
            print "Converged on %s iteration" % i
            break
        if np.isnan(cost) or np.isnan(norm):
            print "Iteration %d diverged!" % i
            break
    print theta
    print "============================================================="
    return theta, costs

def first_part(x, y, alphas, num_iterations):
    m, n = np.shape(x)
    plt.figure()
    for alpha in alphas:
        theta = np.ones(n)
        theta, costs = linear_regression(x, y, theta, alpha, m, num_iterations, 0)

        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        print "Last cost is %s" % costs[-1]
        plt.plot(range(len(costs)), costs)
    plt.legend(['alpha= {}'.format(x) for x in alphas], loc='upper right')
    plt.show()
    return theta

def second_part(x, y, alpha, lambdas, num_iterations):
    m, n = np.shape(x)
    plt.figure()
    for lambda_ in lambdas:
        theta = np.ones(n)
        theta, costs = l2_reg(x, y, theta, alpha, m, num_iterations, lambda_)

        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        print "Last cost is %s" % costs[-1]
        plt.plot(range(len(costs)), costs)
    plt.legend(['Lambda= {}'.format(x) for x in lambdas], loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
