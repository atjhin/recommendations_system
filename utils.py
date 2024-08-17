import time as time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------- Common functions ----------------------------------

def calculate_time(func, *args, **kwargs):
    start = time.time()
    output = func(*args, **kwargs)
    total_time = time.time() - start
    if total_time > 200:
        minute = np.floor(total_time/60)
        second = round(total_time%60, 2)
        print(f"Time taken is {minute} minutes and {second} seconds")
    else:
        print(f"Time taken is {round(total_time, 2)} seconds")
    return output


def plot_pie_cat(series, title='Genre Distribution'):
    category_counts = series

    # Set seaborn style
    sns.set_theme()
    colors = sns.color_palette('deep')

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(category_counts, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title(title, fontsize=20)
    plt.legend(category_counts.index, title="Genre", 
               loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # Show the plot
    plt.show()
    
    
def split_data(df, features, target, test_size=.2, random_state=2024):
    # Standardize the feature values
    scaler = StandardScaler()
    X = df[features].copy()
    X_fit = scaler.fit_transform(X)
    y = df[target].copy()

    # Split the data into training and testing sets
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X_fit, y, test_size=test_size, 
                                                            random_state=2024)
    else:
        X_train = X_fit
        y_train = y
        X_test = pd.DataFrame(columns = features)
        y_test = pd.DataFrame(columns = [target])
    return X_train, X_test, y_train, y_test, scaler
    
    
# ---------------------------------- data_processing functions ----------------------------------

def spotify(client_id, client_secret):
    # Authenticate
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, 
                                                          client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp


def get_tracks_from_username(username, sp, genre_dict):
    track_ls = []
    playlists = sp.user_playlists(username)
    for playlist in playlists['items']:
        if playlist['owner']['id'] == username:
            name = playlist['name']
            genre = [key for key in genre_dict if name in genre_dict[key]]
            if len(genre) == 0:
                print(f"Playlist {name} does not belong in any genre, skipping this playlist")
                continue
            elif len(genre) > 1:
                print(f"Playlist {name} is in multiple genres, skipping this playlist")
                continue
            genre = genre[0]
            results = sp.playlist(playlist['id'], fields="tracks,next")
            tracks = results['tracks']

            for i, item in enumerate(tracks['items']):
                track_ls.append((name, item['track']['id'], item['track']['name'], genre))
    id_df = pd.DataFrame(track_ls, columns=['playlist', 'id', 'name', 'genre'])
    return id_df


def get_audio_features(list_of_id, sp):
    l = [sp.audio_features(track_id)[0] for track_id in list_of_id]
    return pd.DataFrame(l)

# ---------------------------------- logistic_regression functions ----------------------------------

def manual_tuning_lr(X_train, X_test, y_train, y_test, c_ls):
    # local variables
    accuracy_ls = []
    precision_ls = []
    f1_ls = []
    recall_ls = []
    max_accuracy = 0
    for c in c_ls:
        # Loop for each c, calculate and store evaluation metrics
        lr_model = LogisticRegression(penalty='l2', C=c, max_iter=1000, random_state=2024)
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        micro_avg_precision = report['macro avg']['precision']
        micro_avg_recall = report['macro avg']['recall']
        micro_avg_f1 = report['macro avg']['f1-score']

        weighted_avg_precision = report['weighted avg']['precision']
        weighted_avg_recall = report['weighted avg']['recall']
        weighted_avg_f1 = report['weighted avg']['f1-score']

        precision_ls.append((micro_avg_precision, weighted_avg_precision))
        recall_ls.append((micro_avg_recall, weighted_avg_recall))
        f1_ls.append((micro_avg_f1, weighted_avg_f1))

        accuracy = report['accuracy']
        max_accuracy = max(accuracy, max_accuracy)
        if max_accuracy == accuracy:
            print(f"c:{c}, max_accuracy: {max_accuracy}")
        accuracy_ls.append(accuracy)
    return accuracy_ls, precision_ls, recall_ls, f1_ls


def get_manual_tuning_results(X_train, X_test, y_train, y_test, c_ls):
    accuracy_ls, precision_ls, recall_ls, f1_ls = manual_tuning_lr(X_train, X_test, 
                                                                   y_train, y_test, c_ls)
    metrics_ls = [precision_ls, recall_ls, f1_ls]
    names = ['precision', 'recall', 'f1']
    fig = make_subplots(rows=3, cols=1, shared_yaxes=True, subplot_titles=names, vertical_spacing=0.15)
    
    # Add traces for Plot 1
    for i in range(len(metrics_ls)):
        ls = metrics_ls[i]
        # Plot for first subplot
        y1, y2 = zip(*ls)
        fig.add_trace(go.Scatter(x=c_ls, y=y1, mode='lines+markers', name='macro average', line_color="red",
                                 marker=dict(symbol='circle')), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=c_ls, y=y2, mode='lines+markers', name='weighted average', line_color="blue",
                                 marker=dict(symbol='x')), row=i+1, col=1)
        if i == 0:
            fig.add_trace(go.Scatter(x=c_ls, y=accuracy_ls, mode='lines+markers', 
                                     name='accuracy', marker=dict(symbol='square')), 
                          row=i+1, col=1)
        fig.update_xaxes(type='log', tickvals=[.01, .1, 1, 10, 100, 1000], 
                         ticktext=['0.01', '0.1', '1', '10', '100', '500'], row=i+1, col=1)
    fig.update_layout(
        title_text="Logistic regression results",
        xaxis_title="C",
        # yaxis_title="Y-axis",
        height=800, width=1000,
        showlegend=True
    )
    fig.show()
    
    
def eval_lr_model(X,y, **kwargs):
    # Fit and predict with logistic regression
    lr_model = LogisticRegression(penalty='l2', **kwargs)
    lr_model.fit(X, y)
    y_pred = lr_model.predict(X)
    print(classification_report(y, y_pred))
    return y_pred, lr_model


def plot_importance_lr(coef, features, classes):
    # Plot the feature importances for each class using Plotly
    fig = go.Figure()

    for i in range(len(classes)):
        fig.add_trace(go.Bar(
            x=features,
            y=coef[i],
            name=f'Class {classes[i]}',
        ))

    fig.update_layout(
        title='Feature Importances for Logistic Regression (Multi-Class)',
        xaxis_title='Features',
        yaxis_title='Importance (Coefficient)',
        barmode='group',
        xaxis_tickangle=-45
    )

    fig.show()