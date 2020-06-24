# StarbucksCapstoneChallenge
[Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) [Starbucks Capstone Project: Analyzing and Predicting Customer Offers Success](https://medium.com/@vikramverma25/starbucks-capstone-project-analyzing-and-predicting-customer-offers-success-d58392e41a3f)  
  
## Problem Statement / Metrics 
The problem that I chose to solve was to build a model that predicts which offers needs to be given to which type of customers in order to make high offer success. My strategy for solving this problem has four steps. First, I cleaned the datasets to make it ready for analysis.
The data set contains three files. The first file describes offer characteristics including its duration and the amount a customer must spend to complete it (difficulty). The second file contains customer demographic data including their age, gender, income, and when they created an account on the Starbucks rewards mobile application. The third file describes customer purchases and when they received, viewed, and completed an offer.
Second, I assessed the accuracy and F1-score of a naive model that assumes all offers were successful. This provided me a baseline for evaluating the performance of models that I constructed. Accuracy measures how well a model correctly predicts whether an offer is successful. Third, I compared the performance of logistic regression, random forest, gradient boosting, and other models. Fourth, I tuned the model to get better accuracy.

## Results Summary
- Model ranking based on training data [accuracy](https://www.datarobot.com/wiki/accuracy/)  
    1. RandomForestRegressor model accuracy: 100%
    2. K-Nearest Neighbors model accuracy: 100%
    3. LogisticRegression model accuracy: 92.73%
    4. Naive predictor accuracy: 73.42%  

[Bias and variance](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/) are two characteristics of a machine learning model. Bias refers to inherent model assumptions regarding the decision boundary between different classes. On the other hand, variance refers a model's sensitivity to changes in its inputs. 
A logistic regression model constructs a [linear decision boundary](https://datascience.stackexchange.com/questions/6048/decision-tree-or-logistic-regression) to separate successful and unsuccessful offers. However, my exploratory analysis of customer demographics for each offer suggests that this decision boundary will be non-linear. Therefore, an [ensemble method](https://datascience.stackexchange.com/questions/6048/decision-tree-or-logistic-regression) like random forest or gradient boosting should perform better.

Both [random forest](http://blog.citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics) and gradient boosting models are a combination of multiple decision trees. A random forest classifier randomly samples the training data with replacement to construct a set of decision trees that are combined using [majority voting](http://blog.citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics).

## Files  
- Starbucks_Capstone_notebook.ipynb  
  - [Jupyter notebook](https://jupyter.org/) that performs three tasks:  
    - EDA on portfolio, customer demographic, and customer transaction data  
    - Generates training customer demographic data visualizations and computes summary statistics  
    - Generates mechine learning models
- Data
  - portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.).
  - profile.json - demographic data for each customer.
  - transcript.json - records for transactions, offers received, offers viewed, and offers completed.

.gitignore  
  - [Describes](https://git-scm.com/docs/gitignore) files and/or directories that should not be checked into revision control  
- README.md  
  - [Markdown](https://guides.github.com/features/mastering-markdown/) file that summarizes this repository  
	
## Python Libraries Used
-[Python Data Analysis Library](https://pandas.pydata.org/)  
-[Numpy](http://www.numpy.org/)  
-[Matplotlib](https://matplotlib.org/)  
-[seaborn: Statistical Data Visualization](https://seaborn.pydata.org/)  
-[re: Regular expression operations](https://docs.python.org/3/library/re.html)  
-[os — Miscellaneous operating system interfaces](https://docs.python.org/3/library/os.html)  
-[scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/)  

## Licensing, Author, Acknowledgements
Credits must be given to Udacity for the starter codes and Starbucks for provding the data used by this project.