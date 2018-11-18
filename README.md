# us_income
The Jupyter notebook contains code to predict whether an individual's income exceeds US$50000 a year, given socioeconomic information (e.g. age, education, occupation) from the U.S. 1994 Census. Data was obtained from the UCI Machine Learning Repository.

First, I use some visualizations to explore the data. 76.1% of individuals in the sample do not earn more than US$50000 a year (this serves as the baseline accuracy against which I will compare a machine learning model's accuracy later). The probability of earning more than US$50000 a year differs across variables such as sex and race; for example, men are more likely to earn more than US$50000 a year than women.

Next, I recode the infrequent categories of all categorical variables to 'Others'. I define an infrequent category as one which occurs less than 5% of the time. I carry out this recoding exercise to avoid having too many variables after one-hot encoding the categorical variables.

After one-hot encoding the categorical variables, I conduct principal components analysis and add the first 3 principal components to the dataset, in the hope that they may serve as useful features for the machine learning model later. A correlation plot reveals that the first principal component is the feature most strongly correlated with the target variable (correlation coefficient = 0.47).

Finally, I split the dataset into training and test sets, optimize the hyperparameters of an xgboost model via random search (10 iterations, with 3-fold cross validation for each iteration), train the model on the training set, and evaluate the model on the test set. The xgboost model achieves a test accuracy of 86.0%, which is much higher than the baseline accuracy of 76.1%.

The dataset can be accessed from: http://archive.ics.uci.edu/ml/datasets/Adult
