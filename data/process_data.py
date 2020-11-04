import sys

import pandas as pd, numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    assert (messages.id == categories.id).mean()==1, "Messages and categories should share the same index"
    df = pd.concat((messages, categories), axis=1)
    return df

def clean_data(df):
    categories = df.categories.str.split(';', expand=True)

    #select the first row to extract column names
    row = categories.iloc[0]
    categories_colnames = [c.split('-')[0] for c in row]
    categories.columns = categories_colnames

    categories = categories.apply(lambda x: x.str[-1].astype(int))
    categories = categories.where(categories==0, 1)

    df.drop(columns=['categories'], inplace=True)
    df = pd.concat((df, categories), axis=1)
    df = df.drop_duplicates()
    assert df.duplicated().sum()==0, "There are duplicated rows"
    assert df[categories_colnames].isnull().sum().sum()==0, "There are missing values"
    return df

def save_data(df, database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')


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
        print("""Please provide the filepaths of the messages and categories 
        datasets as the first and second argument respectively, as 
        to as the third argument. 
        
        
        Example: python process_data.py 
        disaster_messages.csv disaster_categories.csv 
        DisasterResponse.db""")


if __name__ == '__main__':
    main()
