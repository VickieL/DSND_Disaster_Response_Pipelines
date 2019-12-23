import sys
import pandas as pd
from sqlalchemy import Column, String, create_engine


def load_data(messages_filepath, categories_filepath):
    # read in file
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id', how='inner')

    return df


def clean_data(df):
    # 将 categories 列的值根据字符 ; 进行分割，每个值是一个新列。
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # 使用 categories 数据框的第一行来创建类别数据的列名。
    # select the first row of the categories dataframe
    row = categories.loc[0, :]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split("-")[0]).tolist()
    print(category_colnames) ########

    # 使用新的列名重命名 categories 的列。
    # rename the columns of `categories`
    categories.columns = category_colnames

    # 转换类别值至数值 0 或 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str.get(1)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], errors='coerce')

    # 替换 df categories 类别列
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join_axes=[df.index])

    # 删除重复列
    # drop duplicates
    df = df.drop_duplicates('id')

    return df


def save_data(df, database_filename):
    # load to database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_categories', engine, index=False, if_exists = 'replace')


def main():
    print(len(sys.argv))
    print(sys.argv)
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