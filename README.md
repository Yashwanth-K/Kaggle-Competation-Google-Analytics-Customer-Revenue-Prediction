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

Since the values are constant, we can just drop them from loading it in the feature list and save some memory and time in the modeling process.

Geographic Information:

Inferences:

On the continent plot, we can see that America has both higher number of counts as well as highest number of counts where the revenue is non-zero
Though Asia and Europe has high number of counts, the number of non-zero revenue counts from these continents are comparatively low.
We can infer the first two points from the sub-continents plot too.
If the network domain is "unknown.unknown" rather than "(not set)", then the number of counts with non-zero revenue tend to be lower.

Traffic Source:

Inferences:

In the traffic source plot, though Youtube has high number of counts in the dataset, the number of non-zero revenue counts are very less.
Google plex has a high ratio of non-zero revenue count to total count in the traffic source plot.
On the traffic source medium, "referral" has more number of non-zero revenue count compared to "organic" medium.

Inferences:

Both these variables look very predictive
Count plot shows decreasing nature i.e. we have a very high total count for less number of hits and page views per visitor transaction and the overall count decreases when the number of hits per visitor transaction increases.
On the other hand, we can clearly see that when the number of hits / pageviews per visitor transaction increases, we see that there is a high number of non-zero revenue counts.

Device Information:

Inferences:

Device browser distribution looks similar on both the count and count of non-zero revenue plots
On the device category front, desktop seem to have higher percentage of non-zero revenue counts compared to mobile devices.
In device operating system, though the number of counts is more from windows, the number of counts where revenue is not zero is more for Macintosh.
Chrome OS also has higher percentage of non-zero revenue counts
On the mobile OS side, iOS has more percentage of non-zero revenue counts compared to Android

Date Exploration:

Inferences:

We have data from 1 Aug, 2016 to 31 July, 2017 in our training dataset
In Nov 2016, though there is an increase in the count of visitors, there is no increase in non-zero revenue counts during that time period (relative to the mean).



Now let us compute the evaluation metric on the validation data as mentioned in this new discussion thread. So we need to do a sum for all the transactions of the user and then do a log transformation on top. Let us also make the values less than 0 to 0 as transaction revenue can only be 0 or more.


So we are getting a validation score of 1.70 using this method against the public leaderboard score of 1.44. So please be cautious while dealing with this.


