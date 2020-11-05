import sys
import time
import joblib

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score


nltk.download('stopwords')
nltk.download('wordnet')

STOPWORDS = nltk.corpus.stopwords.words("english")
PARAM_SPACE = {
    'n_estimators': np.arange(10, 50, 5),
    'learning_rate': np.power(10, np.arange(-2, 0.1, 0.5)),
    'min_samples_leaf': np.arange(1, 6),
}


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response', engine)
    X = df['message']
    y = df.iloc[:, -36:]
    y.drop(columns=['child_alone'], inplace=True)  # no positive sample in data
    return X, y, list(y.columns)


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


def build_model():
    start = time.time()
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(analyzer=tokenize)),
        ('clf', MultiOutputClassifier(
            RandomizedSearchCV(
                GradientBoostingClassifier(),
                param_distributions=PARAM_SPACE,
                scoring="f1",
                random_state=7,
                n_iter=20,
                n_jobs=-1))), ]
    )
    print(f'     Took {time.time()-start}')
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_preds = model.predict(X_test)
    for i, category in enumerate(Y_test):
        if category not in category_names:
            continue
        print(category)
        print(f"""
        Frequency rate: {Y_test[category].mean()}
        Precision: {precision_score(Y_test[category], y_preds[:, i])}
        Recall: {recall_score(Y_test[category], y_preds[:, i])}
        F-1 score: {f1_score(Y_test[category], y_preds[:, i])}
        """)


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=7)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
