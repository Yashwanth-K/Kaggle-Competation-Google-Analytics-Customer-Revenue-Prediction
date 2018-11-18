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
