#!/usr/bin/env python
# coding: utf-8

# Question 1.1:



import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import math as math

# Question 1.1

entropy_y = 0.6*math.log(10/6.0,2) + 0.4*math.log(10/4.0,2)
print("entropy H(y)=",entropy_y)


# Question 1.2:



# x1 information gain
# math.log(6.0/3,2) 意思是 2為底數
entropy_1_1 = (3.0/6)*math.log(6.0/3,2) + (3.0/6)*math.log(6.0/3,2)
entropy_1_0 = (3.0/4)*math.log(4.0/3,2) + (1.0/4)*math.log(4.0,2)
information_gain_1 = (6.0/10)*(entropy_y - entropy_1_1) + (4.0/10)*(entropy_y - entropy_1_0)
print('Information gain for feature 1:, %0.4f' %(information_gain_1))

# x2 information gain 
ent_2_1 = (5.0/5)*math.log(5.0/5,2)
ent_2_0 = (4.0/5)*math.log(5.0/4,2) + (1.0/5)*math.log(5.0,2)
information_gain_2 = (5.0/10)*(entropy_y - ent_2_1) + (5.0/10)*(entropy_y - ent_2_0)
print('Information gain for feature 2:, %0.4f' %(information_gain_2))

# x3 information gain 
ent_3_1 = (3.0/7)*math.log(7.0/3,2) + (4.0/7)*math.log(7.0/4,2)
ent_3_0 = (1.0/3)*math.log(3.0,2) + (2.0/3)*math.log(3.0/2,2)
information_gain_3 = (7.0/10)*(entropy_y - ent_3_1) + (3.0/10)*(entropy_y - ent_3_0)
print('Information gain for feature 3:, %0.4f' %(information_gain_3))


# x4 information gain 
ent_4_1 = (2.0/7)*math.log(7.0/2,2) + (5.0/7)*math.log(7.0/5,2)
ent_4_0 = (2.0/3)*math.log(3.0/2,2) + (1.0/3)*math.log(3.0,2)
information_gain_4 = (7.0/10)*(entropy_y - ent_4_1) + (3.0/10)*(entropy_y - ent_4_0)
print('Information gain for feature 4:, %0.4f' %(information_gain_4))

# x5 information gain 
ent_5_1 = (1.0/3)*math.log(3.0,2) + (2.0/3)*math.log(3.0/2,2)
ent_5_0 = (3.0/7)*math.log(7.0/3,2) + (4.0/7)*math.log(7.0/4,2)
information_gain_5 = (3.0/10)*(entropy_y - ent_5_1) + (7.0/10)*(entropy_y - ent_5_0)
print('Information gain for feature 5:, %0.4f' %(information_gain_5))


#question 2.1

xt = np.genfromtxt('data/X_train.txt', delimiter=None)
yt = np.genfromtxt('data/Y_train.txt', delimiter=None)
xt,yt = ml.shuffleData(xt,yt)
for i in range(xt.shape[1]):
    print('minimum of x%d:, %0.4f' %(i,min(xt[:,0])))
    print('maximum of x%d:, %0.4f' %(i,max(xt[:,0])))
    print('mean of x%d:, %0.4f' %(i,np.mean(xt[:,0])))
    print('mean of x%d:, %0.4f' %(i,np.var(xt[:,0])))
    print()

#question 2.2

xt_0_10000 = xt[0:10000]
yt_0_10000 = yt[0:10000]

xv_10000_20000 = xt[10000:20000]
yv_10000_20000 = yt[10000:20000]


# training
dt = ml.dtree.treeClassify(xt_0_10000, yt_0_10000, maxDepth = 50)

# do predictions
# yt_0_10000_hat.shape = (10000,)
yt_0_10000_hat = dt.predict(xt_0_10000)
yv_10000_20000_hat = dt.predict(xv_10000_20000)

# 如果想要把(X,) 的array變成 (1,X)
# eg: yt_0_10000_hat.shape = (10000,1)
# then the following two are equivalent
# yt_0_10000_hat = np.asmatrix(dt.predict(xt_0_10000))
# yt_0_10000_hat = dt.predict(xt_0_10000)#[np.newaxis, :]

print('Training Error:, %0.4f' %(np.sum(yt_0_10000!=yt_0_10000_hat)/yt_0_10000.shape[0]))
print('Validation Error:, %0.4f' %(np.sum(yv_10000_20000!=yv_10000_20000_hat)/yv_10000_20000.shape[0]))


#question 2.3
training_errs=[]
validation_errs=[]
for depth in range(16):
    dt = ml.dtree.treeClassify(xt_0_10000, yt_0_10000, maxDepth = depth)
    yt_0_10000_hat = dt.predict(xt_0_10000)
    yv_10000_20000_hat = dt.predict(xv_10000_20000)
    training_err=np.sum(yt_0_10000!=yt_0_10000_hat)/yt_0_10000.shape[0]
    validation_err=np.sum(yv_10000_20000!=yv_10000_20000_hat)/yv_10000_20000.shape[0]
    training_errs.append(training_err)
    validation_errs.append(validation_err)
    
print("training_errs.shape=",training_errs)
plt.plot(range(16), training_errs, 'b-', linewidth=2)
plt.plot(range(16), validation_errs, 'g-', linewidth=2)
plt.xlabel('Depth')
plt.ylabel('Error')
plt.show()


#question 2.4
training_errs=[]
validation_errs=[]
for mp in 2**np.arange(2,12+1,1):
    dt = ml.dtree.treeClassify(xt_0_10000, yt_0_10000, maxDepth = 50, minParent=mp)
    yt_0_10000_hat = dt.predict(xt_0_10000)
    yv_10000_20000_hat = dt.predict(xv_10000_20000)
    training_err=np.sum(yt_0_10000!=yt_0_10000_hat)/yt_0_10000.shape[0]
    validation_err=np.sum(yv_10000_20000!=yv_10000_20000_hat)/yv_10000_20000.shape[0]
    training_errs.append(training_err)
    validation_errs.append(validation_err)    
    
plt.plot(2**np.arange(2,12+1,1), training_errs, 'b-', linewidth=2)
plt.plot(2**np.arange(2,12+1,1), validation_errs, 'g-', linewidth=2)
plt.xscale('log',basex=2)
plt.xlabel('minParent')
plt.ylabel('Error')
plt.show()

#question 2.6
dt = ml.dtree.treeClassify(xt_0_10000, yt_0_10000, maxDepth = 9)
test = dt.roc(xt_0_10000, yt_0_10000)
print(type(test))
print(test.shape)
