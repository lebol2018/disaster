import sys
import pandas as pd
from sqlalchemy import create_engine

# Function to load data from csv files specified in the command line parameters
def load_data(messages_filepath, categories_filepath):
    '''
    Input: messages_filepath    Full path and filename of the csv file containing the message data
           categories_filepath  Full path and filename of the csv file containing the categories data
           
    Output: df                  A DataFrame merging the two data sources based on the common id
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    
    return df

# Function to clean the data
def clean_data(df):
    '''
    Input: df   The raw DataFrame created in load_data
    Output      A cleaned DataFrame
    '''
    
    # Create a 'categories' DataFrame of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    # Extract the column names from the first row
    row = categories.loc[0]
    category_colnames = [n for n in row.apply(lambda x : x[:-2])]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])    

    # Drop the original 'categories' column now that we have createed a new one for each
    df.drop(['categories'], axis=1, inplace=True)
    
    # Concatenate the two DataFrames
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    # A few rows in the 'related' column have values not in [0,1]. We drop those.
#    df = df[df.related < 2]
    df.drop('related', axis=1, inplace=True)
    
    # Returning the cleaned DataFrame
    return df

# Function to save a DataFrame to a sqlite file
def save_data(df, database_filename):
    '''
    Input: df                   The DataFrame to be saved
           database_filename    The full path and filename of the sqlite file
    '''
    
    # Create the sqlite DB file
    engine = create_engine('sqlite:///' + database_filename)
    
    # Write the DataFrame to the DB fie
    df.to_sql('Messages', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()