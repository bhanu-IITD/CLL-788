#!/usr/bin/env python
# coding: utf-8

# In[36]:


#import required libraries
import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
style.use('seaborn-white')
# cvxopt library to solve quadratic optimization problems
import cvxopt                                            
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


# In[37]:


#Read sample data for Hard Margin SVM Training
test_data = pd.read_excel('Data1.xlsx')
X = np.array(test_data[['Variable 1','Variable 2']].copy()).reshape(-1,2)
y = np.array(test_data[['Class']].copy()).reshape(-1,1)
f = sns.pairplot(x_vars=["Variable 1"], y_vars=["Variable 2"], data=test_data, hue="Class", height=5)
plt.title("\n Training Data")
plt.savefig('train.png',dpi=150,bbox_inches='tight')


# In[38]:


#Initializing values and computing Hessian matrix
m,n = X.shape
y = y.reshape(-1,1) * 1.
X_dash = y * X
H = np.dot(X_dash , X_dash.T) * 1.


# In[39]:


#Converting the problem in cvxopt formulation
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(-np.eye(m))
h = cvxopt_matrix(np.zeros(m))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

#Setting solver parameters and tolerance
cvxopt_solvers.options['show_progress'] = True
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])


# In[40]:


#finding Support vectors and parameters for decision function
S = (alphas > 1e-4).flatten()
sv = X[S]
print ("%d support vectors out of %d sample points found :" % (len(sv), len(X)))
print(sv)
w = ((y * alphas).T @ X).reshape(-1,1)
b = y[S] - np.dot(X[S], w)
print("The weights for decision function are :" ,w.flatten())
print("The intercept for decision function is :" ,b[0])


# In[41]:


#plot the decision function and margins for the trained SVM on given data.
sns.pairplot(x_vars=["Variable 1"], y_vars=["Variable 2"], data=test_data, hue="Class", height=5)
ax = plt.gca()
xlim = ax.get_xlim()
a = -w[0] / w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a * xx - b[0] / w[1]
plt.plot(xx, yy)
yy = a * xx - (b[0] - 1) / w[1]
plt.plot(xx, yy, 'k--')
yy = a * xx - (b[0] + 1) / w[1]
plt.plot(xx, yy, 'k--')
plt.scatter(sv[:,0],sv[:,1],color = 'green',s=300, linewidth=2, facecolors='none')
plt.title("\n Hard Margin SVM Classification")
plt.savefig("SVM.png",dpi=150,bbox_inches='tight')


# In[42]:


# Read sample data for soft margin SVM training by using C as regularization parameter
test_data2 = pd.read_excel('Data2.xlsx')
X = np.array(test_data2[['Variable 1','Variable 2']].copy()).reshape(-1,2)
y = np.array(test_data2[['Class']].copy()).reshape(-1,1)
f = sns.pairplot(x_vars=["Variable 1"], y_vars=["Variable 2"], data=test_data2, hue="Class", height=5)
plt.title("\n Training Data")
plt.savefig('train_c.png',dpi=150,bbox_inches='tight')


# In[43]:


#Initializing values and computing Hessian matrix. Please note regularization parameter "C" for soft margin SVM. 
C =1
m,n = X.shape
y = y.reshape(-1,1) * 1.
X_dash = y * X
H = np.dot(X_dash , X_dash.T) * 1.

#Converting into cvxopt format - as previously
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])


# In[44]:


#finding Support vectors and parameters for decision function
w = ((y * alphas).T @ X).reshape(-1,1)
S = (alphas > 1e-1).flatten()
sv = X[S]
b = y[S] - np.dot(X[S], w)
print ("%d support vectors out of %d sample points found :" % (len(sv), len(X)))
print(sv)
print("The weights for decision function are :" ,w.flatten())
print("The intercept for decision function is :" ,b[0])


# In[45]:


#plot the decision function and margins for the trained SVM on given data.
sns.pairplot(x_vars=["Variable 1"], y_vars=["Variable 2"], data=test_data, hue="Class", height=5)
ax = plt.gca()
xlim = ax.get_xlim()
a = -w[0] / w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a * xx - b[0] / w[1]
plt.plot(xx, yy)
yy = a * xx - (b[0] - 1) / w[1]
plt.plot(xx, yy, 'k--')
yy = a * xx - (b[0] + 1) / w[1]
plt.plot(xx, yy, 'k--')
plt.scatter(sv[:,0],sv[:,1],color = 'green',s=300, linewidth=2, facecolors='none')
plt.title('\n Soft Margin SVM Classification')
ax.text(4.5, 4.5, 'C ='+str(C), color='purple', 
        bbox=dict(facecolor='none', edgecolor='black', pad=10.0))
plt.savefig("SVM_c"+str(C)+".png",dpi=150,bbox_inches='tight')


# In[ ]:





# In[ ]:




