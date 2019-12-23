import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
# engine = create_engine('sqlite:///../data/DisasterResponse.db')
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    related_counts = df.groupby('related').count()['message']
    related_names = list(related_counts.index)

    request_counts = df.groupby('request').count()['message']
    request_names = list(request_counts.index)

    aid_centers_counts = df.groupby('aid_centers').count()['message']
    aid_centers_names = list(aid_centers_counts.index)

    weather_related_counts = df.groupby('weather_related').count()['message']
    weather_related_names = list(weather_related_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=related_names,
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Related Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "related"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=request_names,
                    y=request_counts
                )
            ],

            'layout': {
                'title': 'Distribution of request Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "request"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=aid_centers_names,
                    y=aid_centers_counts
                )
            ],

            'layout': {
                'title': 'Distribution of aid_centers Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "aid_centers"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=weather_related_names,
                    y=weather_related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of weather_related Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "weather_related"
                }
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
