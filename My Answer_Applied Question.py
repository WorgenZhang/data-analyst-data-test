
# coding: utf-8

# In[172]:

import pandas as pd
import numpy as np
import numpy.random as randn
from pandas import Series,DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[173]:

cars_mileage = pd.read_csv("/Users/Frank/Desktop/data-analyst-data-test/Cars_mileage.csv")


# In[174]:

cars_mileage.head()


# In[175]:

cars_mileage.info()


# ## (a)create a binary variable
# 

# In[176]:

cars_mileage=cars_mileage.dropna()

median = cars_mileage['mpg'].median()
cars_mileage['mpg_binary'] = Series(np.random.randn(len(cars_mileage['mpg'])))
cars_mileage['mpg_binary'].loc[cars_mileage['mpg'] > median] = 1
cars_mileage['mpg_binary'].loc[cars_mileage['mpg'] <= median] = 0


# In[177]:

cars_mileage.head()


# In[178]:

cars_mileage.dtypes


# In[179]:

cars_mileage=cars_mileage.convert_objects(convert_numeric=True)


# ## (b) Which of the other variables seem most likely to be useful in predicting whether a car's mpg is above or below its median?
# 

# In[180]:

sns.pairplot(cars_mileage,x_vars=['mpg','cylinders','displacement'],
             y_vars=['mpg_binary'])
sns.pairplot(cars_mileage,x_vars=['horsepower','acceleration','year'],
             y_vars=['mpg_binary'])


# In[181]:

plt.scatter(cars_mileage['weight'],cars_mileage['mpg_binary'])


#  #### Conlusion:
#  From the plot, we see that high displacement or high horsepower or high weight may lead to the car's mpg below it's median.
#  
#  I think 'horsepower' is the most likely variable to be useful in predicting whether a car's mpg is above or below its median, since you can see from the plot, nearly after 125 horsepower, a car's mpg are all below its median.

# ## (c) Split the data into a training set and a test set.

# In[182]:

# drop rows where 'horsepower' column contain missing values.
cars_mileage=cars_mileage[np.isfinite(cars_mileage['horsepower'])]


# In[183]:

cars_mileage.count()


# In[184]:

from patsy import dmatrices


# In[185]:

# create dataframes with an intercept column
y,X=dmatrices('mpg_binary~mpg+cylinders+displacement+horsepower+weight+acceleration+year+origin',cars_mileage,
             return_type="dataframe")


# In[186]:

# flatten y into a 1-D array
y = np.ravel(y)


# In[187]:

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error


# In[188]:

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[189]:

print(len(X_train),len(X_test))


# ## (d) Perform two of the following in order to predict mpg_binary:

# In[190]:

cars_mileage.head()


# >###  First: Logistic Regression

# In[191]:

# Logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)


# In[192]:

# check the accuracy on the traning set
model.score(X,y)


# In[193]:

# examine the coefficients
pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))


# ##### Increas in cylinders, displacement, and weight correspond to increase the liklihood of above the median of mileage. Instead, increase in cylinders, horsepower, acceleration, year and origin correspond to decrease the liklihood of above the median of mileage.

# #### Model Evaluation

# In[194]:

model2 = LogisticRegression()
model2.fit(X_train, y_train)


# In[195]:

# predict class labels for the test set
predicted = model2.predict(X_test)
print predicted


# In[196]:

# Meas squared error
mean_squared_error(y_test,predicted)


# In[197]:

# generate class probabilities
probs = model2.predict_proba(X_test)
print probs[0:10]


# #### As you can see, the classifier is predicting a 1 (above the median) any time the probability in the second column is greater than 0.5.

# In[198]:

# generate evaluation metrics
print metrics.accuracy_score(y_test, predicted)


# ##### The accuracy is 95.76%

# In[199]:

# classification report
print metrics.classification_report(y_test, predicted)


# #### Precision of mpg_binary when it equals 1 is 95%.

# In[200]:

# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print scores
print scores.mean()


# #### By cross-validation method (10-fold), we see the mean of 10 folds' accuracy is 94.89%, which is nealy same what we got before.

# >### Second: Ridge Regression

# In[201]:

from sklearn.linear_model import Ridge


# In[219]:

def ridge_regression(X, y, alpha, models_to_plot={}):
    #Fit the model
    model3 = Ridge(alpha=alpha,normalize=True,max_iter=1e5)
    model3.fit(X,y)
    y_pred = model3.predict(X)
    
    #For every alpha, make a plot
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(X,y_pred)
        plt.plot(X,y,'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result with rss,intercept, and coefficients
    rss = sum((y_pred-y)**2)
    ret = [rss]
    ret.extend([model3.intercept_])
    ret.extend(model3.coef_[1:10])
    return ret


# In[220]:

#Set the different values of alpha to be tested
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,9)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(X, y, alpha_ridge[i], models_to_plot)


# In[221]:

coef_matrix_ridge


# ### Observations:
# ####  1. We can see, the RSS increase with alpha increases. Meanwhile, the complexity of model will reduce.
# ####  2 .High alpha will lead to underfitting. But when alpha=1e-15, the RSS is not small, we conclude that using ridge regression to fit the model is not a quite good idea.

# >### Third: Lasso Regression

# In[216]:

from sklearn.linear_model import Lasso
def lasso_regression(X, y, alpha, models_to_plot={}):
    #Fit the model
    model4 = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    model4.fit(X,y)
    y_pred = model4.predict(X)
    
    #For every alpha, make a plot
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(X,y_pred)
        plt.plot(X,y,'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result with rss,intercept, and coefficients
    rss = sum((y_pred-y)**2)
    ret = [rss]
    ret.extend([model4.intercept_])
    ret.extend(model4.coef_[1:10])
    return ret


# In[217]:

#Set the different values of alpha to be tested
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,9)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(X, y, alpha_lasso[i], models_to_plot)


# In[218]:

coef_matrix_lasso


# ### Observations: 
# #### 1. High alpha results in high rss.
# 
# #### 2. When alpha is big, for same alpha, lasso has higher rss than ridge, which means it has poor fit that ridge.
# 
# #### 3. Many of coefficients are zero.

# In[223]:

# check how many 0 coffeicient does each row have
coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1)


# In[ ]:



