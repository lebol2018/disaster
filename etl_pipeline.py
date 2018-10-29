# import libraries
import pandas as pd
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv('messages.csv')
messages.head()

# load categories dataset
categories = pd.read_csv('categories.csv')
categories.head()

# merge datasets
df = messages.merge(categories, on='id')
df.head()

# create a dataframe of the 36 individual category columns
categories = df.categories.str.split(';', expand=True)
categories.head()

# select the first row of the categories dataframe
row = categories.loc[0]
#print(row)

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything
# up to the

second to last character of each string with slicing
category_colnames = [n for n in row.apply(lambda x : x[:-2])]
print(category_colnames)

# rename the columns of `categories`
categories.columns = category_colnames
categories.head()

for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].astype(str).str[-1:]

    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
categories.head()

# drop the original categories column from `df`
df.drop(['categories'], axis=1, inplace=True)

df.head()

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis=1)
df.head()

# check number of duplicates
df.duplicated().sum()

# drop duplicates
df.drop_duplicates(inplace=True)

df.duplicated().sum()

engine = create_engine('sqlite:///messages.db')
df.to_sql('Messages', engine, index=False)
