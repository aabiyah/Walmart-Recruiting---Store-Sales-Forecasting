## Walmart Recruiting - Store Sales Forecasting

# Category Encoders

Category encoders are used in machine learning to convert categorical data into numerical data so that algorithms can process them effectively. Different types of category encoders handle this transformation in various ways:

1. **One-Hot Encoding:** 
   - Creates binary columns for each category, where a value of 1 indicates the presence of a category and 0 indicates its absence.

2. **Label Encoding:** 
   - Converts each category into a unique integer value.

3. **Ordinal Encoding:** 
   - Similar to label encoding but assumes an inherent order in the categories.

4. **Binary Encoding:** 
   - Converts categories into binary numbers, which are then split into separate columns.

5. **Target Encoding (Mean Encoding):** 
   - Replaces each category with the mean of the target variable for that category.

6. **Frequency Encoding:** 
   - Replaces each category with the frequency of that category in the dataset.

These techniques help algorithms handle categorical features by converting them into a numerical format suitable for model training and prediction.

# Python Imports

`python
import warnings
warnings.filterwarnings("ignore")`
> Suppresses warning messages to keep the output clean and focused.

`import pandas as pd`
> Provides data structures and data analysis tools for handling and analyzing tabular data.

`import csv`
> Facilitates reading from and writing to CSV files, which is a common format for data storage.

`import matplotlib.pyplot as plt`
> Offers functions for creating static, interactive, and animated plots and visualizations.

`import seaborn as sns`
> Provides a high-level interface for drawing attractive and informative statistical graphics.

`import numpy as np`
> Supports large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these arrays.

`import re`
> Provides support for regular expressions, which are useful for searching and manipulating text.

`import os`
> Offers functions to interact with the operating system, including file and directory management.

`from datetime import datetime`
> Supplies classes for manipulating dates and times in both simple and complex ways.

`from sklearn.ensemble import RandomForestRegressor`
> Implements a random forest regressor model, which is used for predicting continuous values.

`from IPython.core.display import display, HTML`
> Provides functions for displaying rich content like HTML within Jupyter notebooks.

`import category_encoders as ce`
> Offers various techniques for encoding categorical variables into numerical formats suitable for machine learning models.

# Steps

1. Read all the csv files and explore them in different ways, such as checking how many stores of each type there are, how many holidays there are, etc.
2. Visualise the datasets.
3. Data Cleaning & Transformation:
> Convert 'Date' columns in train, test, and features DataFrames to datetime format for easier manipulation.
> Add columns for week, day of the week, month, year, and day based on the 'Date' column in train and test DataFrames.
> Compute and add mean temperature and unemployment values from the features DataFrame to both train and test DataFrames.
> Merge train and test DataFrames with the features DataFrame using 'Store' and 'Date' columns.
> Add store details from the stores DataFrame to the merged train and test DataFrames.
> Remove duplicate 'IsHoliday' columns and rename the remaining one for consistency.
> Change the 'IsHoliday' column values from boolean to binary (0 and 1).
> Convert store types from categorical labels ('A', 'B', 'C') to numeric values (1, 2, 3).
> Display the first few rows and summary statistics of the cleaned train_with_feature and test_with_feature DataFrames to verify changes.
4. Calculating correlations in training and testing datasets:
> Calculate the correlation matrix for numerical columns in the train_with_feature and test_with_feature DataFrame. Plot a heatmap to visualize the correlations.
> Check for missing values by printing the count of missing values for each column in the train_with_feature and test_with_feature DataFrames and replace all missing values with 0 in both train_with_feature and test_with_feature DataFrames.
> Drop unnecessary features.
> Prepare final training and testing datasets by defining train_X (features) and train_y (target) for training, and test_X (features) for testing; dropping  'Weekly_Sales' and 'Date' from train_final to create train_X, and assign 'Weekly_Sales' to train_y; and dropping  'Date' from test_final to create test_X.
> Inspect the final datasets before moving on to training the model.
5. Machine learning model:

> • **Create a Subset for Quick Testing**
Objective: Use a random 10% subset of the training data to speed up testing.
Explanation: train_X.sample(frac=0.1, random_state=42) selects 10% of the rows from train_X randomly. train_y.loc[train_X_subset.index] selects corresponding target values for the subset.

> • **Create and Train the Model**
Objective: Initialize and train a Random Forest model.
Explanation: RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1) creates a Random Forest model with 100 trees, each with a maximum depth of 10, and uses all available CPU cores for computation. clf.fit(train_X_subset, train_y_subset) trains the model using the subset of training data.

> • **Predict and Calculate Accuracy**
Objective: Make predictions and evaluate the model%E2%80%99s accuracy.
Explanation: clf.predict(test_X) generates predictions for the test_X data. clf.score(train_X_subset, train_y_subset) computes the R%C2%B2 score (coefficient of determination) on the training subset, which is then rounded and converted to a percentage. The accuracy is printed.
