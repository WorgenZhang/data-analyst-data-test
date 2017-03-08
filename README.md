# Junior Data Analyst Data Test
This is the data test for the Junior Data Analyst position at Azavea. Please complete the questions below and submit your responses by XXXX date. The first question is conceptual and a **written response** is expected. The second question and it's several parts are applied. A submission of the **code** used to solve the problems, the **answers**, along with any **plots or graphics** generated are expected as responses. 

***

### 1) Conceptual Question

#### This first question is conceptual and written responses are expected. For each item below, indicate whether the appropriate method would be classification or regression, and whether we are most interested in inference or prediction. Also, indidate what n and p are for each section.  

> **(a)** A dataset contains data for 350 manufacturing companies in Europe. The following variables are included in the data for each company: industry, number of employees, salary of the CEO, and total profit. **We are interested in learning which variables impact the CEO's salary.** 

> **(b)** A market research company is hired to help a startup analyze their new product. **We want to know whether the product will be a success or failure.** Similar products exist on the market so the market research company gathers data on 31 similar products. The company records the following data points about each previously launched product: price of the product, competition price, marketing budget, ten other variables, and whether or not it succeeded or failed. 

> **(c)** Every week data is collected for the world stock market in 2012. The data points collected include the % change in the dollar, the % change in the market in the United States, the % change in the market in China, and the % change in the market in France. **We are interested in predicting the % change in the dollar in relation to the changes every week in the world stock markets.** 


### 2) Applied Question

#### For this second applied question you will develop several predictive models. These should be written in R or Python and the code should be submitted. The models will predict whether a car will get high or low gas mileage. The question will be based on the Cars_mileage data set that is a part of this repo.

> **(a)** Create a binary variable that represents whether the car's mpg is above or below its median. Above the median should be represented as 1. Name this variable **mpg_binary**. 

> **(b)** Which of the other variables seem most likely to be useful in predicting whether a car's mpg is above or below its median? **Describe your findings and submit visual representations of the relationship between mpg_binary and other variables.**

> **(c)** Split the data into a training set and a test set.

> **(d)** Perform LDA (Linear discriminant analysis) on the training data in order to predict mpg_binary using the variables that seemed most associated from part (b). **What is the test error of the model obtained?**
      
> **(e)** Perform QDA (Quadratic discriminant analysis) on the training data in order to predict mpg_binary using the variables that seemed most associated from part (b). **What is the test error of the model obtained?**

> **(f)** Perform logistic regression on the training data in order to predict mpg_binary using using the variables that seemed most associated from part (b). **What is the test error of the model obtained?**

> **(g)** Perform KNN (K-nearest neighbors) on the training data, with several values of K, in order to predict mpg_binary. Use only the variables that seemed most associated with mpg_binary in (b). **What test errors do you obtain? Which value of K seems to perform the best on this data set?**