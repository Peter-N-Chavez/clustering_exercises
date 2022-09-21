import env

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import urllib.request
from PIL import Image
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def get_db_url(hostname, username, password, database):
    url = f'mysql+pymysql://{username}:{password}@{hostname}/{database}'
    return url

def get_mallcustomer_data():
    '''
    Reads in all fields from the customers table in the mall_customers schema from data.codeup.com
    
    parameters: None
    
    returns: a single Pandas DataFrame with the index set to the primary customer_id field
    '''
    df = pd.read_sql('SELECT * FROM customers;', get_db_url(env.hostname, env.username, env.password,'mall_customers'))
    return df.set_index('customer_id')

# query will grab our zillow data with notable specs from the exercise set, particularly:
# include logerror (we already joined on this table previously, no biggie)
# include date of transaction
# only include properties with lat+long filled
# narrow down to single unit properties (we will do this with pandas so we can investigate the fields)
query = '''
SELECT
    prop.*,
    predictions_2017.logerror,
    predictions_2017.transactiondate,
    air.airconditioningdesc,
    arch.architecturalstyledesc,
    build.buildingclassdesc,
    heat.heatingorsystemdesc,
    landuse.propertylandusedesc,
    story.storydesc,
    construct.typeconstructiondesc
FROM properties_2017 prop
JOIN (
    SELECT parcelid, MAX(transactiondate) AS max_transactiondate
    FROM predictions_2017
    GROUP BY parcelid
) pred USING(parcelid)
JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                      AND pred.max_transactiondate = predictions_2017.transactiondate
LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
LEFT JOIN storytype story USING (storytypeid)
LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
WHERE prop.latitude IS NOT NULL
  AND prop.longitude IS NOT NULL
  AND transactiondate <= '2017-12-31'
'''

def overview(df):
    '''
    print shape of DataFrame, .info() method call, and basic descriptive statistics via .describe()
    parameters: single pandas dataframe, df
    return: none
    '''
    print('--- Shape: {}'.format(df.shape))
    print('--- Info')
    df.info()
    print('--- Column Descriptions')
    print(df.describe(include='all'))

def nulls_by_columns(df):
    '''
    Get the number and proportion of values per column in the dataframe df
    parameters: single pandas dataframe, df
    return: none
    '''
    return pd.concat([
        df.isna().sum().rename('count'),
        df.isna().mean().rename('percent')
    ], axis=1)

def nulls_by_rows(df):
    '''
    Get the number and proportion of values per row in the dataframe df
    parameters: single pandas dataframe, df
    return: none
    '''
    return pd.concat([
        df.isna().sum(axis=1).rename('n_missing'),
        df.isna().mean(axis=1).rename('percent_missing'),
    ], axis=1).value_counts().sort_index()

def handle_missing_values(df, prop_required_column, prop_required_row):
    '''
    Utilizing an input proportion for the column and rows of DataFrame df,
    drop the missing values per the axis contingent on the amount of data present.
    '''
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1, thresh=n_required_column)
    return df

def acquire():
    '''
    aquire the zillow data utilizing the query defined earlier in this wrangle file.
    will read in cached data from any present "zillow.csv" present in the current directory.
    first-read data will be saved as "zillow.csv" following query.
    parameters: none
    '''
    if os.path.exists('zillow.csv'):
        df = pd.read_csv('zillow.csv')
    else:
        database = 'zillow'
        url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{database}'
        df = pd.read_sql(query, url)
        df.to_csv('zillow.csv', index=False)
    return df

def wrangle_zillow():
    '''
    acquires, gives summary statistics, and handles missing values contingent on
    the desires of the zillow data we wish to obtain.
    parameters: none
    return: single pandas dataframe, df
    '''
    # grab the data:
    df = acquire()
    # summarize and peek at the data:
    overview(df)
    nulls_by_columns(df).sort_values(by='percent')
    nulls_by_rows(df)
    # task for you to decide: ;)
    # determine what you want to categorize as a single unit property.
    # maybe use df.propertylandusedesc.unique() to get a list, narrow it down with domain knowledge,
    # then pull something like this:
    # df.propertylandusedesc = df.propertylandusedesc.apply(lambda x: x if x in my_list_of_single_unit_types else np.nan)
    # we will drop all missing values for our MVP.
    # In our second iteration, we will tune the proportion and e:
    # df = wrangle_zillow.handle_missing_values(df, prop_required_column=.5, prop_required_row=.5)
    df = df.drop_na()
    # take care of any duplicates:
    df = df.drop_duplicates()
    return df

def split_data(df, target, seed=123):
    '''
    This function takes in a dataframe and splits the data into train, validate and test. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, random_state=seed)
    
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=seed)
    return train, validate, test
    
def scale_my_malldata(train, validate, test):
    '''
    scale my data using minmaxscaler and add it back to my input datasets
    '''
    scaler = MinMaxScaler()
    scaler.fit(train[['age', 'annual_income']])
    
    X_train_scaled = scaler.transform(train[['age', 'annual_income']])
    X_validate_scaled = scaler.transform(validate[['age', 'annual_income']])
    X_test_scaled = scaler.transform(test[['age', 'annual_income']])

    train[['age_scaled', 'annual_income_scaled']] = X_train_scaled
    validate[['age_scaled', 'annual_income_scaled']] = X_validate_scaled
    test[['age_scaled', 'annual_income_scaled']] = X_test_scaled
    return train, validate, test

def prep_mall(df):
    '''
    dummy var for gender into is_male
    split on target of 'spending_score'
    scale age and annual income. 
    '''
    df['is_male'] = pd.get_dummies(df['gender'], drop_first=True)['Male']
    train, validate, test = split_data(df, target='spending_score', seed=1349)
    train, validate, test = scale_my_malldata(train, validate, test)
    
    print(f'df: {df.shape}')
    print()
    print(f'train: {train.shape}')
    print(f'validate: {validate.shape}')
    print(f'test: {test.shape}')
    return df, train, validate, test
