### 1) Conceptual Question

#### This first question is conceptual and written responses are expected. For each item below, indicate whether the appropriate method would be classification or regression, and whether we are most interested in inference or prediction. Please include a written sentence or two explaining why you made this choice. Also, indidate what n and p (predictors) are for each section.

> **(a)** A dataset contains data for 350 manufacturing companies in Europe. The following variables are included in the data for each company: industry, number of employees, salary of the CEO, and total profit. **We are interested in learning which variables impact the CEO's salary.** 

**Answer:** The appropriate method would be **regression**. We are most interested in **inference**. Since CEO's salary is a continuous variable. And we want to know what variables impact it instead of predicting another CEO's salary.

**n** is number of manufacturing companies in Europe (350). **p** are predictors including industry, number of employees, and total profit.

> **(b)** A market research company is hired to help a startup analyze their new product. **We want to know whether the product will be a success or failure.** Similar products exist on the market so the market research company gathers data on 31 similar products. The company records the following data points about each previously launched product: price of the product, competition price, marketing budget, ten other variables, and whether or not it succeeded or failed. 

**Answer:** The appropriate method would be **classification**. We are most interested in **prediction**. Since y can only take two value: success and failure, it's a classification problem. And we want to predict whether this new product will be a success or failure base on data gathered from similar products, so it's a prediction problem.

**n** is number of similar products (31). **p** are predictors including price of the product, competition price, marketing budget, ten other variables.

> **(c)** Every week data is collected for the world stock market in 2012. The data points collected include the % change in the dollar, the % change in the market in the United States, the % change in the market in China, and the % change in the market in France. **We are interested in predicting the % change in the dollar in relation to the changes every week in the world stock markets.** 

**Answer:** The appropriate method would be **regression**. We are most interested in **inference**. Since % change in the dollar is a continous variable, it's a regression problem. And we want to know it's relation to the changes every week in the world stock markets instead of predicting % change in the dollar in 2013, so it's a inference. 

**n** is number of weeks collected for the world stock market in 2012. **p** are predictors including the % change in the market in the United States, the % change in the market in China, and the % change in the market in France.