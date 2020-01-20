# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:37:31 2019

@author: Abhishek Shambhu
"""
# Building a Logistic Regression model and fitting it on the reg-lr-data dataset
# Importing packages and reading the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
data = pd.read_csv(r'D:\sem3\or610\reg-lr-data.csv')

# splitting the data into the x values and y values as numpy arrays
x = data.iloc[:,:3].values
y = data['y'].values

# Logistic Regression Model using Gradient Descent Method
class Log_Reg_L2:
    """ Defining Logistic Regression with L2 regularization
    Parameters are:
        l2: lambda value for l2 regularization
        n: number of iterations over the dataset
        l_rate: learning rate/step value
    """
    
    def __init__(self, l2=0.0, n=1000, l_rate=0.05):
        self.l2 = l2
        self.n = n
        self.l_rate = l_rate
            
    def sigmoid(self, z):
        # This is the sigmoid function of z
        return 1/(1+ np.exp(-z))
    
    def fit(self, x, y):
        # fit the training data
        
        y = y.reshape(-1,1)
        # initialize the values of the weights to zero
        self.theta = np.zeros((x.shape[1],1))
        m = y.shape[0]
        # adding in padding value so that we never take the log of 0
        pad = 1e-6
        self.cost_values = []
        for i in range(self.n):
            z = self.sigmoid(np.dot(x, self.theta))
            # calculating the gradient with the derived formula
            gradient = x.T.dot(z-y)/m + (self.l2/m*self.theta)
            self.theta -= self.l_rate * gradient
            # implementing the cost (objective) function given
            cost = np.average(-y*np.log(z+pad) - ((1-y)*np.log(1-z+pad)))
            l2_cost = cost + (self.l2/(2*m) * np.linalg.norm(self.theta[1:])**2)  # we don't regularize the intersect
            self.cost_values.append(l2_cost)
        return self
      
    def predict(self, x, threshold=0.5):
        # return the predicted values in (0,1) format
        return np.where(self.sigmoid(x.dot(self.theta)) >= threshold,1,0)
    
    def predict_prob(self, x):
        # return the predicted values in percentage format
        return self.sigmoid(x.dot(self.theta))

# Logistic Regression using Stochastic Gradient Descent       
class Log_Reg_L2_SGD:
    """ Logistic Regression with L2 regularization and Stochastic Gradient Descent
    The parameters are:
        l2: lambda value for l2 regularization
        n: number of iterations over the dataset
        l_rate: learning rate
        batch_size: size of each batch (SGD=1 and full batch = len(x))
    """
    
    def __init__(self, l2=0.0, n=1000, l_rate=0.05, batch_size=1):
        self.l2 = l2
        self.n = n
        self.l_rate = l_rate
        self.batch_size = batch_size
            
    def sigmoid(self, z):
        # This is the sigmoid function of z
        return 1/(1+ np.exp(-z))
    
    def fit(self, x, y):
        # fit the training data
        
        y = y.reshape(-1,1)
        # initialize the values of the weights to zero
        self.theta = np.zeros((x.shape[1],1))
        m = y.shape[0]
        pad = 1e-6
        self.cost_values = []
        for i in range(self.n):
            # shuffling each iteration as to prevent overfitting
            shuffled_values = np.random.permutation(m)
            X_shuffled = x[shuffled_values]
            y_shuffled = y[shuffled_values]
            # iterating over each batch
            for batch in range(0, m, self.batch_size):
                x_batch = X_shuffled[batch:batch+self.batch_size]
                y_batch = y_shuffled[batch:batch+self.batch_size]
                z = self.sigmoid(np.dot(x_batch, self.theta))
                # calculating the gradient with the derived formula
                gradient = x_batch.T.dot(z-y_batch)/m + (self.l2/m*self.theta)
                self.theta -= self.l_rate * gradient
                # implementing the cost (objective) function given
                cost = np.average(-y_batch*np.log(z+pad) - ((1-y_batch)*np.log(1-z+pad)))
                l2_cost = cost + (self.l2/(2*m) * np.linalg.norm(self.theta[1:])**2)  # we don't regularize the intersect
                self.cost_values.append(l2_cost)
        return self

    def predict(self, x, threshold=0.5):
        # return the predicted values in (0,1) format
        return np.where(self.sigmoid(x.dot(self.theta)) >= threshold,1,0)
    
    def predict_prob(self, x):
        # return the predicted values in percentage format
        return self.sigmoid(x.dot(self.theta))

# Function to Plot the Cost Values
def plot_cost(trained_model, printed_values = 30, is_sgd=False):
    # printed values determines how many values are printed to the chart
    # this prevents the chart from becoming too cluttered
    if is_sgd:
        # averaging the values over each iteration
        batch_avg = [np.mean(trained_model.cost_values[i:i+4]) for i in range(1, len(trained_model.cost_values), int(x.shape[0]/trained_model.batch_size))]
        model_plot = [batch_avg[i] for i in range(1, len(batch_avg), int(trained_model.n/printed_values))]
        plt.plot(range(1, len(batch_avg),int(trained_model.n/printed_values)), model_plot, marker='o')
        plt.xlabel('Iteration Number')
        plt.ylabel('Cost Value')
        plt.title('Logistic Regression Cost (L2={})'.format(trained_model.l2))
    else:
        model_plot = [trained_model.cost_values[i] for i in range(1, len(trained_model.cost_values), int(trained_model.n/printed_values))]
        plt.plot(range(1, len(trained_model.cost_values)+1,int(trained_model.n/printed_values)), model_plot, marker='o')
        plt.xlabel('Iteration Number')
        plt.ylabel('Cost Value')
        plt.title('Logistic Regression Cost (L2={})'.format(trained_model.l2))

# Function to Plot the Decision Boundary       
def plot_decision_boundary(trained_model, x, y, is_sgd=False):
    fig, ax = plt.subplots()  
    predictions = model.predict(x)
    #plotting class = 0 correct
    ax.scatter(x[(predictions.flatten() == y) & (y==0)][:,1], x[(predictions.flatten() == y) & (y==0)][:,2], color='b', label="Class 0")
    # plotting class = 1 correct
    ax.scatter(x[(predictions.flatten() == y) & (y==1)][:,1], x[(predictions.flatten() == y) & (y==1)][:,2], color='g', label="Class 1")
    # plotting incorrect classifications
    ax.scatter(x[predictions.flatten() !=y][:,1], x[predictions.flatten() != y][:,2], color='r', label="Wrong")
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    x1_min, x1_max = x[:,1].min()-1, x[:,1].max()+1
    x2_min, x2_max = x[:,2].min()-1, x[:,2].max()+1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    graph_predictions = model.predict(np.array([np.ones((2500,1)).ravel(), xx1.ravel(), xx2.ravel()]).T)
    graph_predictions = graph_predictions.reshape(xx1.shape)
    ax.contourf(xx1, xx2, graph_predictions,alpha=0.2, cmap='bwr')
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    if is_sgd:
        ax.set_title('\n'.join(wrap("SGD LR Model with {} batch size, {} eta, {} iterations, and {} L2".format(trained_model.batch_size,
                        trained_model.l_rate, trained_model.n, trained_model.l2),50)), fontsize=12)
    else:
        ax.set_title('\n'.join(wrap("LR Model with {} eta, {} iterations, and {} L2".format(
                        trained_model.l_rate, trained_model.n, trained_model.l2),50)),fontsize=12)
    plt.show()

#Part 1)
#Part 1A) Logistic Regression Gradient Descent: l_rate=0.05, n=300,000, L2=0.0
model = Log_Reg_L2(l2=0.0, n=300000,l_rate=0.05)
model.fit(x, y)
predictions = model.predict(x)
print("1A) Accuracy: {:.0f}%".format(sum(predictions.flatten() == y)/len(y)*100))

plot_cost(model, printed_values=30)

plot_decision_boundary(model, x, y)

#Part 1B) Logistic Regression Gradient Descent: l_rate=0.05, n=300,000, L2=0.5
model = Log_Reg_L2(l2=0.5, n=300000,l_rate=0.05)
model.fit(x, y)
predictions = model.predict(x)
print("1B) Accuracy: {:.0f}%".format(sum(predictions.flatten() == y)/len(y)*100))
        
plot_cost(model, printed_values=30)

plot_decision_boundary(model, x, y)

#Part 1C) Logistic Regression Gradient Descent: l_rate=0.05, n=300,000, L2=1
model = Log_Reg_L2(l2=1, n=300000, l_rate=0.05)
model.fit(x, y)
predictions = model.predict(x)
print("1C) Accuracy: {:.0f}%".format(sum(predictions.flatten() == y)/len(y)*100))
        
plot_cost(model, printed_values=30)

plot_decision_boundary(model, x, y)

#Part 2)
#Part 2A) Logistic Regression Stochastic Gradient Descent: l_rate=0.05, n=10,000, L2=0.0, batch_size=1

model = Log_Reg_L2_SGD(l2=0, n=10000,l_rate=0.05, batch_size=1)
model.fit(x, y)
predictions = model.predict(x)
print("2A) Accuracy: {:.0f}%".format(sum(predictions.flatten() == y)/len(y)*100))
        
plot_cost(model, printed_values=30, is_sgd=True)

plot_decision_boundary(model, x, y, is_sgd=True)

#Part 2B) Logistic Regression Stochastic Gradient Descent: l_rate=0.05, n=10,000, L2=0.5, batch_size=1
model = Log_Reg_L2_SGD(l2=0.5, n=10000,l_rate=0.05, batch_size=1)
model.fit(x, y)
predictions = model.predict(x)
print("2B) Accuracy: {:.0f}%".format(sum(predictions.flatten() == y)/len(y)*100))
        
plot_cost(model, printed_values=30, is_sgd=True)

plot_decision_boundary(model, x, y, is_sgd=True)

#Part 2C) Logistic Regression Stochastic Gradient Descent: l_rate=0.05, n=10,000, L2=1, batch_size=1
model = Log_Reg_L2_SGD(l2=1, n=10000,l_rate=0.05, batch_size=1)
model.fit(x, y)
predictions = model.predict(x)
print("2C) Accuracy: {:.0f}%".format(sum(predictions.flatten() == y)/len(y)*100))
        
plot_cost(model, printed_values=30, is_sgd=True)

plot_decision_boundary(model, x, y, is_sgd=True)