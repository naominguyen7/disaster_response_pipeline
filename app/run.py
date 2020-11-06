from sqlalchemy import create_engine
import joblib
from plotly.graph_objs import Bar, Figure, Image
from flask import render_template, request, jsonify
from flask import Flask
import pandas as pd
import plotly
import json

import re
import nltk

from plotly.subplots import make_subplots
from skimage import io


app = Flask(__name__)

STOPWORDS = nltk.corpus.stopwords.words("english")


def tokenize(text):
    stemmer = nltk.stem.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    text = text.lower()
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    text = text.split()
    text = [stemmer.stem(w) for w in text if w not in STOPWORDS]
    text = [lemmatizer.lemmatize(w, pos='n')
            for w in text if w not in STOPWORDS]
    text = [lemmatizer.lemmatize(w, pos='v')
            for w in text if w not in STOPWORDS]
    return text


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    category_names = df.columns[4:]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=df[category_names].sum()
                )
            ],

            'layout': {
                'title': 'Distribution of message categories - All',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "category"
                }
            }
        }
    ]

    genres = ['News messages', 'Social messages', 'Direct messages'
              ]
    fig = make_subplots(
        rows=1, cols=len(genres),
        subplot_titles=genres
    )
    for i, g in enumerate(genres):
        img = io.imread(f'img/{g}.png')
        fig.add_trace(Image(z=img), 1, i+1)
    fig.update_layout(coloraxis_showscale=False,
                      title_text="Word cloud of related messages")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    graphs.append(
        fig
    )

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, categories=category_names)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    # Since no child_alone messages in data
    category_names = [v for v in df.columns[4:] if v != 'child_alone']
    classification_results = dict(zip(category_names, classification_labels))

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
