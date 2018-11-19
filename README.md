# Kaggle-Competation-Google-Analytics-Customer-Revenue-Prediction

**Install**<br>
This project requires Python and the following Python libraries installed:

NumPy<br>
Pandas<br>
matplotlib<br>
scikit-learn<br>
seaborn<br>
You will also need to have software installed to run and execute a Jupyter Notebook

If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the above packages and more included.

**Problem Statement**<br>
The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies.

In this kaggle competition,we are challenged to analyze a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict revenue per customer.

**File Descriptions**

- train.csv - the training set - contains the same data as the rstudio_train_set.
- test.csv - the test set - contains the same data as the rstudio_test_set.
- sampleSubmission.csv - a sample submission file in the correct format. Contains all fullVisitorIds in test.csv.

**Data Mining**

The data is shared in big query and csv format. The csv files contains some columns filled with json objects.We need to convert them and explore the revised dataset.<br>
For converting the JSON column we need to import:
```
from pandas.io.json import json_normalize
```
There are actually 4 JSON columns device,geoNetwork,totals,trafficSource that we need to normalize before working on the data sets.

**Target Variable Exploration:**

Since we are predicting the natural log of sum of all transactions of the user, summing up the transaction revenue at user level and take a log and then do a scatter plot.<br>
From this above exploration it confirms the first two lines of the competition overview.<br>
And we get,
```
the ratio of revenue generating customers are 1.219%
```
**Cleaning Data**

Check for the columns that have constant values and remove them from the train and test sets.

**Handle Missing Values**<br>
Compute the number of missing values and determine how to handle them. We can return the number of missing values across the DataFrame by:
```
- First, use the Pandas DataFrame method isnull() to return a DataFrame containing Boolean values:
- True if the original value is null
- False if the original value isn't null
- Then, use the Pandas DataFrame method sum() to calculate the number of null values in each column.
```

**Seperate categorical columns and numerical columns from train set**<br>

Investigate Categorical Columns<br>
We need to convert all the columns as numeric columns (int or float data type), and containing no missing values. We just dealt with the missing values above, now find out the number of columns that are of the object data type and then move on to process them into numeric form.

We need to handle missing values and categorical features before feeding the data into a machine learning algorithm, because the mathematics underlying most machine learning models assumes that the data is numerical and contains no missing values. To reinforce this requirement, scikit-learn will return an error if you try to train a model using data that contain missing values or non-numeric values when working with models like linear regression and logistic regression.

So we use `LabelEncoder` for encoding the Categorical Columns.

**Create development and validation splits based on time to build the model.**<br>
```
I have taken the data from Aug 1st 2016 to April 30th 2018 as the training set values and May 1st 2018 to Oct 15th 2018 data as validation set.
```
**Training**<br>
Run light gbm model to train the model

<font color=green>Light GBM is a gradient boosting framework that uses tree based learning algorithm.</font>
```
Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other<br> algorithm grows level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm.
```
Light GBM is prefixed as ‘Light’ because of its high speed. Light GBM can handle the large size of data and takes lower memory to run. Another reason of why Light GBM is popular is because it focuses on accuracy of results. LGBM also supports GPU learning and thus data scientists are widely using LGBM for data science application development.

**Evaluation**

Compute the evaluation metric on the validation data. Do a sum for all the transactions of the user and then take a log transformation on top.Make the values less than 0 to 0 as transaction revenue can only be 0 or more.
```
print(np.sqrt(metrics.mean_squared_error(np.log1p(pred_df_val["transactionRevenue"].values), np.log1p(pred_df_val["PredictedRevenue"].values))))
```
**Parameter tuning**

There are many parameters that can be tuned in light GBM and LightGBM is a relatively new algorithm and it doesn’t have a lot of reading<br> resources on the internet except its documentation. It becomes difficult for a beginner to choose parameters from the long list given in the documentation. Even I reffered to some good kernals to tune the parameters, and obtain the result of 1.73 in Kaggle public score board.

**Submission file**
Convert the transaction revenue column to PredictedLogRevenue by taking expm1 on a predicted test and then grouping by fullVisitorId and finally apply log1p on PredictedLogRevenue and convert to csv and submit.


