import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
     this function takes the two file_paths of csv files and merge
     them into a single dataframe.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
    '''
        this function takes a dataframe convert the categorie columns into 
        each label colums and convert columns values into 1 
        and 0 by extracting the information in particular row-col pair.
        
        parameters:
            df = data frame 
        return:
            df = clean data frame having each label as a colums and its value as 1 and 0.
    '''
    categories = df.categories.str.split(';', expand=True)
    row = categories.loc[0]
    column_name = []
    for value in row:
        column_name.append(value.split('-')[0])
    categories.columns = column_name
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand = True)[1]
        # convert column from string to numeric
        categories[column] = categories[column].apply(int)
        
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    df = df.join(categories)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df


def save_data(df, database_filename):
    '''
        it takes the dataframe and filepath and
        store it inot the sql database.
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disasterresponse', engine, index=False, if_exists='replace')
    pass  


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
