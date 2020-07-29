import base64
import os
from urllib.parse import quote as urlquote

from flask import Flask, send_from_directory
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output, State
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from sklearn.datasets import load_files
from sklearn.metrics import plot_confusion_matrix
import pickle
from nltk.corpus import stopwords
import csv

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

stemmer = WordNetLemmatizer()
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
tfidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))


UPLOAD_DIRECTORY = "Desktop/"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

server = Flask(__name__)
app = dash.Dash(server=server)

@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

# Load data
df = pd.read_csv('data/stockdata2.csv', index_col=0, parse_dates=True)
df.index = pd.to_datetime(df['Date'])

# Initialize the app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True


def get_options(list_stocks):
    dict_list = []
    for i in list_stocks:
        dict_list.append({'label': i, 'value': i})

    return dict_list


app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.H2('Complaints Bucketing System',  style={'color': '#007eff'}),
                                 html.P('Upload Training Set'),
                                 dcc.Upload(
                                     id="upload-data",
                                     children=html.Div(
                                         ["Click to select a file to train a model"]
                                     ),
                                     style={
                                         "width": "100%",
                                         "height": "50px",
                                         "lineHeight": "50px",
                                         "borderWidth": "1px",
                                         "borderStyle": "dashed",
                                         "borderRadius": "5px",
                                         "textAlign": "center",
                                         "margin": "10px",
                                         "color":"white",
                                     },
                                     multiple=True,),
                                     html.Div(id='accuracy', style={'whiteSpace': 'pre-line', "background":"gray","color":"white"}),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                     html.P('Upload Test Set'),
                                     dcc.Upload(
                                     id="upload-data-test",
                                     children=html.Div(
                                     ["Click to select a file to test the trained model"]
                                     ),
                                     style={
                                     "width": "100%",
                                     "height": "50px",
                                     "lineHeight": "50px",
                                     "borderWidth": "1px",
                                     "borderStyle": "dashed",
                                     "borderRadius": "5px",
                                     "textAlign": "center",
                                     "margin": "10px",
                                     "font-color":"white"
                                     },
                                     multiple=True,),
                                     ],
                                     style={'color': 'white'}),
                                                         html.Div([
                                                         html.H4('Test Bucketing'),
                                                         dcc.Textarea(id='textarea-input',
                                                         value='Enter the complaint here',
                                                         style={'width': '100%', 'height': '50px',}),
                                                         html.Button('Submit', id='btn_submit',
                                                         n_clicks=0),
                                                         html.Div(id='textarea-output', style={'whiteSpace': 'pre-line', 'color':'white'})
                                                         ], style={
                                                         "width": "100%",
                                                         "height": "100%",
                                                         "lineHeight": "50px",
                                                         "borderWidth": "1px",
                                                         "borderStyle": "solid",
                                                         "borderRadius": "5px",
                                                         "textAlign": "center",
                                                         "padding": "10px",
                                                         "margin": "10px",
                                                         "font-color":"white"
                                                         },)
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
                                 dcc.Graph(id='timeseries', config={'displayModeBar': False}, animate=True)
                             ]),
                              ]),
        ])

def process_text(complaint_list):
    documents = []
    for sen in range(0, len(complaint_list)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(complaint_list[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)

    return documents

def train_model(documents, complaints, topics):

    complaints = vectorizer.fit_transform(documents).toarray()

    complaints = tfidfconverter.fit_transform(documents).toarray()

    X_train, X_test, y_train, y_test = train_test_split(complaints, topics, test_size=0.2, random_state=0)

    #classifier = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)

    #classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    #print(confusion_matrix(y_test,y_pred))
    #print(classification_report(y_test,y_pred))
    return accuracy_score(y_test, y_pred)

def initiate_training(name):
    reviews = [row for row in csv.reader(open(os.path.join(UPLOAD_DIRECTORY, name),'r'))]
    reviews = reviews[1:]
    complaints = [row[0] for row in reviews]
    topics = [row[1].lower() for row in reviews]
    documents = process_text(complaints)
    return train_model(documents,complaints,topics)

def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))
    return name

def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files

def plotgraph(topics, counts):
    data = [go.Bar(
    x = topics,
    y = counts)]
    return {'data':data,'layout': go.Layout(
        autosize=False,
        title={'text': 'Freq Dist', 'font': {'color': 'white'}, 'x': 0.5},
        yaxis=dict(
        title_text="Y-axis Title",
        ticktext=topics,
        tickmode="array",
        titlefont=dict(size=30),
        range=[0, sum(counts)],
    )),}


def predict(name):
    testdata = [row for row in csv.reader(open(os.path.join(UPLOAD_DIRECTORY, name), "r"))]
    complaints_test = [classifier.predict(vectorizer.transform([row[0]])) for row in testdata]
    predicted = []
    wordfreq = []
    for comp in complaints_test:
        for item in comp:
            predicted.append(item)
    for w in predicted:
        wordfreq.append(predicted.count(w))
    return [predicted,wordfreq]

@app.callback(
    Output("accuracy", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)
            return [html.Li("Accuracy: " + "{0:.0f}%".format(initiate_training(name)*100),style={"color":"#f55442","background":"#1e1e1e"})]

# Callback for timeseries price
@app.callback(Output('timeseries', 'figure'),
              [Input("upload-data-test", "filename"), Input("upload-data-test", "contents")])
def update_graph(uploaded_filenames, uploaded_file_contents):
    figure = go.Figure([go.Bar(x=[],y=[])])
    name = ''
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            name = save_file(name, data)
        predicted = predict(name)
        figure = plotgraph(predicted[0], predicted[1])
        return figure
    else:
        return figure

@app.callback(
    Output('textarea-output', 'children'),
    [Input('btn_submit', 'n_clicks')],
    [State('textarea-input', 'value')]
)
def update_output(n_clicks, value):
    if(n_clicks > 0):
        return 'Complaint is bucketed to: {}'.format(classifier.predict(vectorizer.transform([value])))

if __name__ == '__main__':
    app.run_server(debug=True)
