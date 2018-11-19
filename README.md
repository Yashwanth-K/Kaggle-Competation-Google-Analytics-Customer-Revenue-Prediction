# Kaggle-Competation-Google-Analytics-Customer-Revenue-Prediction

**Problem Statement**<br>
The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies.

In this kaggle competition,we are challenged to analyze a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict revenue per customer.

**File Descriptions**

- train.csv - the training set - contains the same data as the rstudio_train_set.
- test.csv - the test set - contains the same data as the rstudio_test_set.
- sampleSubmission.csv - a sample submission file in the correct format. Contains all fullVisitorIds in test.csv.

**Data Mining**

The data is shared in big query and csv format. The csv files contains some filed with json objects.We need to convert them and explore the revised dataset.

```
def loading_data_sets(csv_path='../input/train.csv'):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    load_dataframe = pd.DataFrame()
    reading_csvfile = pd.read_csv(csv_path, sep=',',
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'},
                    chunksize = chunk)
```
**Target Variable Exploration:**

Since we are predicting the natural log of sum of all transactions of the user, sum up the transaction revenue at user level and take a log and then do a scatter plot.

![80](https://user-images.githubusercontent.com/44206279/48670296-65e67500-eb3b-11e8-89f7-6eff04cdf197.png)

From this above exploration it confirms the first two lines of the competition overview.

```* The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies.```

```
nonzero_instance = pd.notnull(train_df["totals.transactionRevenue"]).sum()
nonzero_uniq = (groupdf["totals.transactionRevenue"]>0).sum()
print("No. of instances in train set with non-zero revenue : ", nonzero_instance, " and ratio is : ", nonzero_instance / train_df.shape[0])
print("No. of unique customers with revenue greater than 0 : ", nonzero_uniq, "and the ratio is : ", nonzero_uniq / groupdf.shape[0])
```

```
No. of instances in train set with non-zero revenue :  18514  and ratio is :  0.010837440153786987
No. of unique customers with revenue greater than 0 :  16141 and the ratio is :  0.012193574218307359
```

So the ratio of revenue generating customers to customers with no revenue is in the ratio of 1.219%

**Columns with constant values:**

Since the values are constant, we can just drop them from loading it in the feature list and save some memory and time in the modeling process.

**Geographic Information:**

![country](https://user-images.githubusercontent.com/44206279/48670285-4c452d80-eb3b-11e8-824a-d54d05917d83.png)

![continent](https://user-images.githubusercontent.com/44206279/48670286-4ea78780-eb3b-11e8-957e-b92f5ba92192.png)

![network](https://user-images.githubusercontent.com/44206279/48670508-f2defd80-eb3e-11e8-8b16-39bcc585e945.png)

**Inferences:**

- On the continent plot, we can see that America has both higher number of counts as well as highest number of counts where the revenue is non-zero
- Though Asia and Europe has high number of counts, the number of non-zero revenue counts from these continents are comparatively low.
- If the network domain is "unknown.unknown" rather than "(not set)", then the number of counts with non-zero revenue tend to be lower.

**Traffic Source:**

![youtube](https://user-images.githubusercontent.com/44206279/48670541-35a0d580-eb3f-11e8-9740-fa6deeb496c0.png)

**Inferences:**

In the traffic source plot, though Youtube has high number of counts in the dataset, the number of non-zero revenue counts are very less.
Google has a high ratio of non-zero revenue count to total count in the traffic source plot.

**Device Information:**

![browser](https://user-images.githubusercontent.com/44206279/48670288-536c3b80-eb3b-11e8-9635-5b38aab219f9.png)

![desktop](https://user-images.githubusercontent.com/44206279/48670290-55ce9580-eb3b-11e8-883a-6d9d9b3c2fdc.png)

![os](https://user-images.githubusercontent.com/44206279/48670291-56ffc280-eb3b-11e8-8fb0-1889b179a05c.png)

**Inferences:**

- Device browser distribution looks similar on both the count and count of non-zero revenue plots
- On the device category front, desktop seem to have higher percentage of non-zero revenue counts compared to mobile devices.
- In device operating system, though the number of counts is more from windows, the number of counts where revenue is not zero is more for Macintosh.
- Chrome OS also has higher percentage of non-zero revenue counts
- On the mobile OS side, iOS has more percentage of non-zero revenue counts compared to Android

**Date Exploration:**

![monday](https://user-images.githubusercontent.com/44206279/48670292-58c98600-eb3b-11e8-9452-38106091c362.png)

![month](https://user-images.githubusercontent.com/44206279/48670294-5ebf6700-eb3b-11e8-9933-acbb032b8520.png)

![date](https://user-images.githubusercontent.com/44206279/48670295-61ba5780-eb3b-11e8-861d-0c353fe95621.png)

**Inferences:**

We have data from Aug 1st 2016 to April 30th 2018 in our training dataset.<br>
- More customers have visited the store during nov to dec 2017

**Seperate categorical columns and numerical columns from train set**

```category_cols = list()
for i in train_df.columns:
    if train_df[i].dtype=='object' and (not(i.startswith('total'))):
        category_cols.append(i)
category_cols

numeric_cols = list()
for i in train_df.columns:
    if train_df[i].dtype not in ['object', 'bool']:
        numeric_cols.append(i)
numeric_cols
```
**Find the missing values in the columns**

Find the missing values:

```train_df[numerical_cols].isnull().sum()```

|columns             |number of null values|
|-----------------|------------------|
|visitNumber                 |      0|
|totals.transactionRevenue    |     0|
|weekday                       |    0|
|day                            |   0|
|month|                             0|
|visitHour          |               0|
|totals.bounces      |         836759|
|totals.hits          |             0|
|totals.newVisits      |       400907|
|totals.pageviews       |         239|

Fill nan with specific values to get better results

```
train_df['totals.bounces'] = train_df['totals.bounces'].fillna(0)
test_df['totals.bounces'] = test_df['totals.bounces'].fillna(0)

train_df['totals.newVisits'] = train_df['totals.newVisits'].fillna(0)
test_df['totals.newVisits'] = test_df['totals.newVisits'].fillna(0)

train_df['totals.pageviews'] = train_df['totals.pageviews'].fillna(1)
test_df['totals.pageviews'] = test_df['totals.pageviews'].fillna(1)
```
Create development and validation splits based on time to build the model.
```
from datetime import date

val_df = train_new[train_new['date']>='2018-3-30']
val_X = val_df[categorical_cols+numerical_cols]
val_Y = val_df['totals.transactionRevenue']

dev_df = train_new[train_new['date']<'2018-3-30']
train_X = dev_df[categorical_cols+numerical_cols]
train_Y = dev_df['totals.transactionRevenue']
```
**Run light gbm model to train the model**

```
LGBM_train = lgb.Dataset(train_X, label=train_Y,categorical_feature=category_cols)
LGBM_valid = lgb.Dataset(val_X, label=val_Y,categorical_feature=category_cols)
LGBM_model = lgb.train(params, LGBM_train, 2000, valid_sets=[LGBM_valid], early_stopping_rounds=100, verbose_eval=100)
```

Compute the evaluation metric on the validation data. Do a sum for all the transactions of the user and then do a log transformation on top.Make the values less than 0 to 0 as transaction revenue can only be 0 or more.
```
print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
```
**Submit the file**
```
submit = pd.DataFrame({"fullVisitorId":id})
pred_test[pred_test<0] = 0
submit["PredictedLogRevenue"] = np.expm1(pred_test)
submit = submit.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
submit.columns = ["fullVisitorId", "PredictedLogRevenue"]
submit["PredictedLogRevenue"] = np.log1p(submit["PredictedLogRevenue"])
submit.to_csv("D:\\Sub_file.csv", index=False)
```


