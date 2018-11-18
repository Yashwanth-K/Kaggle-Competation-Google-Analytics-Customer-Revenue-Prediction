# Kaggle-Competation-Google-Analytics-Customer-Revenue-Prediction

**Problem Statement**<br>
The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies.

In this kaggle competition,we are challenged to analyze a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict revenue per customer.

**File Descriptions**

train.csv - the training set - contains the same data as the rstudio_train_set.
test.csv - the test set - contains the same data as the rstudio_test_set.
sampleSubmission.csv - a sample submission file in the correct format. Contains all fullVisitorIds in test.csv.

**Data Mining**

The data is shared in big query and csv format. The csv files contains some filed with json objects.We need to convert them and explore the revised dataset.

```
def load_df(csv_path='../input/train.csv'):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    load_dataframe = pd.DataFrame()
    reading_csvfile = pd.read_csv(csv_path, sep=',',
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'},
                    chunksize = 100000)
```
**Target Variable Exploration:**

Since we are predicting the natural log of sum of all transactions of the user, let us sum up the transaction revenue at user level and take a log and then do a scatter plot.

![80](https://user-images.githubusercontent.com/44206279/48670296-65e67500-eb3b-11e8-89f7-6eff04cdf197.png)

From this above exploration it confirms the first two lines of the competition overview.

```* The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies.```

```
nonzero_instance = pd.notnull(train_df["totals.transactionRevenue"]).sum()
nonzero_uniq = (groupdf["totals.transactionRevenue"]>0).sum()
print("Number of instances in train set with non-zero revenue : ", nonzero_instance, " and ratio is : ", nonzero_instance / train_df.shape[0])
print("Number of unique customers with revenue greater than 0. : ", nonzero_uniq, "and the ratio is : ", nonzero_uniq / groupdf.shape[0])
```

```
Number of instances in train set with non-zero revenue :  18514  and ratio is :  0.010837440153786987
Number of unique customers with revenue greater than 0. :  16141 and the ratio is :  0.012193574218307359
```

So the ratio of revenue generating customers to customers with no revenue is in the ratio of 1.219%

**Columns with constant values:**





