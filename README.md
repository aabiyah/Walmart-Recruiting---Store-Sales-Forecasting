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
