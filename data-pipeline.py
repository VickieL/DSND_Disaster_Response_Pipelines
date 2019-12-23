# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(data_file):
    # read in file
    messages = pd.read_csv('messages.csv')
    categories = pd.read_csv('categories.csv')
    # merge datasets
    df = messages.merge(categories, on='id', how='inner')

    # clean data
    # 分割 categories
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0, :]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split("-")[0]).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames

    # 转换类别值至数值 0 或 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str.get(1)
        # convert column from string to numeric
        categories[column]

    # 替换 df categories 类别列
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join_axes=[df.index])

    # 删除重复列
    # drop duplicates
    df = df.drop_duplicates('id')

    # load to database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('messages_categories', engine, index=False, if_exists = 'replace')


    # define features and label arrays
    X = df[['message','original']]
    Y = df[['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
        'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
        'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport',
        'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']]


    return X, y


def build_model():
    # text processing and model pipeline


    # define parameters for GridSearchCV


    # create gridsearch object and return as final model pipeline


    return model_pipeline


def train(X, y, model):
    # train test split


    # fit model


    # output model test results


    return model


def export_model(model):
    # Export model as a pickle file
    pass



def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
