import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re, time, os, joblib
from tqdm.notebook import trange, tqdm
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer, WordNetLemmatizer

from tqdm.notebook import tqdm, trange
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image as WordImage
from IPython.display import Image
from IPython.display import display, display_html, HTML
from plotnine import *
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso


from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN, OPTICS,
                             cluster_optics_dbscan)
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)

# ML
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Regression Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier

# Classifier Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.metrics import (precision_score, recall_score,
                             accuracy_score, f1_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV

# Explainability
import shap
from dice_ml import Data, Model, Dice

import warnings
warnings.filterwarnings('ignore')


def df_exploration(df, display_df=True):
    """
    Displays the number of data points, nulls, and preview of the data frame.
    """
    display_html(f'<b>Number of Data Points:</b> {df.shape[0]}',
                 raw=True)
    display_html(f'<b>Number of Features:</b> {df.shape[1]}',
                 raw=True)

    # Check if any null values exist
    if df.isna().any().any():
        # Get the columns with null values
        columns_with_nulls = df.columns[df.isna().any()].tolist()
        # Display the number of null values per column for 
        # columns with null values
        null_cols_info = (", ".join([f"{col}={df[col].isna().sum()}" 
                                     for col in columns_with_nulls]))
        null_cols_text = f'<b>Null Values:</b> \n{null_cols_info}'
        display_html(null_cols_text, raw=True)

    else:
        display_html('<b>No Null Values Found.</b>', raw=True)

    if display_df:
        display(df.head(3))


def adjust_popularity(row):
    """
    This function adjusts the 'popularity' column in a given row based on the
    value of the 'popularity_bins' column. It multiplies the 'popularity'
    value by a specific factor depending on whether 'popularity_bins' 
    is 'Low', 'Medium', or another value. 
    The adjusted 'popularity' value is then returned.
    """
    if row['popularity_bins'] == 'Low':
        return row['popularity'] * 30000000
    elif row['popularity_bins'] == 'Medium':
        return row['popularity'] * 100000000
    else:
        return row['popularity'] * 150000000


def translate_ids_to_salaries(main_df, target_column,
                              people_df, new_column_name):
    """
    This function adds a new column to a given DataFrame with the
    total salary for each row, based on the IDs in a specified column.
    The function extracts the IDs from the specified column,
    looks up their corresponding salary values from another DataFrame,
    and calculates the total salary by summing up all the extracted
    salary values for each ID. The new column is added with the name
    specified in 'new_column_name'.
    """
    for index, value in main_df[target_column].items():

        total_salary = 0

        if not pd.isna(value):
            values = value.split(", ")

            for element in values:

                id = int(element)
                salary = people_df.loc[people_df["tmdb_id"]
                                       == id, "salary"].item()

                total_salary += salary

            main_df.loc[index, new_column_name] = total_salary
            
def load_data():
    """
    Load data from CSV files and return a tuple of pandas DataFrames.
    """
    series_data = pd.read_csv("data/series.csv")
    genres_data = pd.read_csv("data/genres.csv")
    networks_data = pd.read_csv("data/networks.csv")
    people_data = pd.read_csv("data/people.csv")
    production_companies_data = pd.read_csv("data/production_companies.csv")
    
    return series_data, people_data           
            
def create_popularity_bins(data):
    """
    Create a new column 'popularity_bins' based on the 'popularity'
    column values and adjust 'salary' column values.
    """
    low = np.percentile(data['popularity'], 50)
    mid = np.percentile(data['popularity'], 75)
    custom_bins = [0, low, mid, 100]
    data['popularity_bins'] = pd.cut(data['popularity'],
                                     bins=custom_bins,
                                     labels=['Low', 'Medium', 'High'])
    
    data['salary'] = data.apply(adjust_popularity, axis=1)
    return data

def add_production_cost(data):
    """
    Add a new column 'production_cost' to the input DataFrame 
    based on the 'salary' and 'number_of_episodes' columns.
    """
    data['production_cost'] = data.salary * data.number_of_episodes
    return data

def preprocess_series_data(data):
    """
    Preprocesses a DataFrame of TV series data.
    """
    # Drop rows with missing keywords
    data = data.dropna(subset=['keywords'])

    # Forward fill missing values
    data = data.ffill()

    # Backward fill missing values
    data = data.backfill()

    # Create target column
    data['target'] = (data['average_rating'] > 7.7).astype(int)

    # Drop average_rating column
    data = data.drop(['average_rating'], axis=1)

    # Return preprocessed data
    return data

def visualize_target_distribution(data):
    """
    Visualize the distribution of the target variable in a bar chart.
    """
    colors = ['#d9534f', '#5cb85c']
    # Compute the target value counts
    target_counts = data['target'].value_counts()

    # Create a plot using seaborn
    plt.figure(figsize=(6, 4))
    sns.set_style('whitegrid')
    ax = sns.barplot(x=target_counts.index,
                     y=target_counts.values, palette=colors)
    ax.set_title('Popularity Distribution')
    ax.set_xlabel('Target')
    ax.set_ylabel('Count')
    ax.set_xticklabels(['Not Popular', 'Popular'])
    
    # Save the plot as a PNG file
    plt.savefig("images/balance.png")

    # Create an HTML img tag to display the image
    img_tag = (f'<img src="images/balance.png" alt="balance"'
               f'style="display:block; margin-left:auto;'
               f'margin-right:auto; width:80%;">')

    # display the HTML <img> tag
    display(HTML(img_tag))
    plt.close()
    
    
def assign_quadrant(df):
    """
    Assign a quadrant number to each TV series in the input DataFrame 
    based on its production cost and popularity.
    """
    scaler = StandardScaler()
    df_quad = scaler.fit_transform(df[['production_cost', 'popularity']])
    scaled_productioncost = df_quad[:, 0]
    scaled_popularity = df_quad[:, 1]

    # Assign quadrant numbers
    df['Quadrant'] = np.select(
        [
            (scaled_productioncost >= 0) & (scaled_popularity >= 0),
            (scaled_productioncost < 0) & (scaled_popularity >= 0),
            (scaled_productioncost < 0) & (scaled_popularity < 0),
        ],
        [1, 2, 3],
        default=4
    )

    row_index = df[df['tmdb_id'] == 154825].index[0]
    df.loc[row_index, 'Quadrant'] = 2

    return df

def plot_quadrant_visualization(df_quad):
    """
    Create a scatter plot to visualize the TV series in different
    quadrants and display a quadrant count table.
    """

    colors = ['#5bc0de', '#5cb85c',  '#f0ad4e', '#d9534f']

    # Create a scatter plot with color-coded quadrants
    sns.scatterplot(data=df_quad, x='production_cost',
                    y='popularity', hue='Quadrant', palette=colors)
    plt.title('Quadrant Visualization')
    plt.xlabel('Production Cost')
    plt.ylabel('Popularity')

    # Save the plot as a PNG file
    plt.savefig("images/quadrant.png")

    # Create an HTML img tag to display the image
    img_tag = (f'<img src="images/quadrant.png" alt="quadrant"'
               f'style="display:block; margin-left:auto;'
               f'margin-right:auto; width:80%;">')

    # display the HTML <img> tag
    display(HTML(img_tag))
    plt.close()

    # Get the quadrant counts
    quad_counts = df_quad['Quadrant'].value_counts().sort_index()
    quad_counts_df = pd.DataFrame(quad_counts, columns=['Quadrant'])

    # Get the colors used in the scatter plot
    colors = ['#5cb85c', '#5bc0de', '#d9534f', '#f0ad4e']

    html = ('<table style="border-collapse:collapse; margin-left: auto; margin-right: auto;">'
            '<tr><th></th><th colspan="2" style="text-align:center;font-size:16px;">Production Cost</th></tr>'
            '<tr><th rowspan="2" style="text-align:center;font-size:16px;">Popularity <br> Rating</th>'
            '<td style="text-align:center;background-color:{};font-size:40px;">{}</td><td style="text-align:center;background-color:{};font-size:40px;">{}</td></tr>'
            '<tr><td style="text-align:center;background-color:{};font-size:40px;">{}</td><td style="text-align:center;background-color:{};font-size:40px;">{}</td></tr>'
            '</table>'.format(
                colors[0], quad_counts_df.loc[2]['Quadrant'], colors[1], quad_counts_df.loc[1]['Quadrant'],
                colors[3], quad_counts_df.loc[3]['Quadrant'], colors[2], quad_counts_df.loc[4]['Quadrant']))

    # Display the HTML table
    display(HTML(html))
    
    
def preprocess_keywords_data(data):
    """
    Preprocess the keywords data by cleaning and encoding 
    the 'keywords' column into a one-hot encoded dataframe.
    """
    # Replace any spaces and commas with just a comma in the keywords column
    data['keywords'] = data['keywords'].str.replace(' *[,] *', ',', regex=True)

    # Convert the keywords column into a one-hot encoded dataframe
    keywords_df = data['keywords'].str.get_dummies(',')
    keywords_df = keywords_df.drop(
        columns=['blindness', 'south korea', 'k bl'])

    # Concatenate the one-hot encoded dataframe with the target 
    # and quadrant columns
    final_df = pd.concat([data[['target', 'Quadrant']], keywords_df], axis=1)

    # Fill any missing values with the previous value in the column
    final_df = final_df.fillna(method='ffill')

    # Perform exploration on the final dataframe
    df_exploration(final_df)

    return final_df

def prepare_quadrant_splits(df_final):
    """
    Prepare train-test splits for each quadrant of the input DataFrame.
    """

    # Define an empty dictionary to store the train-test splits for each quadrant
    quadrant_splits = {}

    # Loop through each quadrant and do the train-test split
    for q in range(1, 5):
        # Get the data for the current quadrant
        df_quadrant = df_final[df_final['Quadrant'] == q]

        # Split the data into X and y
        X, y = df_quadrant.drop(['target', 'Quadrant'],
                                axis=1), df_quadrant['target']

        # Do the train-test split
        X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
            X, y, test_size=0.30, random_state=10053)

        # Add the train-test splits to the dictionary
        quadrant_splits[q] = (X_trainval, X_holdout, y_trainval, y_holdout)

    return quadrant_splits

# Create a scoring function
def custom_scorer(estimator, X, y):
    """
    A custom scoring function that calculates the absolute difference between
    the number of non-zero features and a target number of non-zero features.
    """
    non_zero_count = np.count_nonzero(estimator.coef_)
    penalty = abs(non_zero_count - target_non_zero_features)
    return -penalty

# Define the desired number of non-zero features
target_non_zero_features = 50

# Set up the hyperparameter grid
alphas = np.logspace(-5, 1, 50)

def fit_lasso_and_plot(quadrant_splits, custom_scorer, top_n=5):
    """
    This function fits a Lasso model for each quadrant and stores the fitted
    models and their feature weights in a dictionary. It creates a horizontal
    bar chart for each quadrant showing the top positive and negative feature
    weights, with the number of features to display being customizable.
    The bar chart for each quadrant is saved as a PNG file and displayed
    using an HTML img tag. The function returns the dictionary of fitted
    Lasso models and their feature weights.
    """
    # Initialize dictionary to store Lasso models and their feature weights for each quadrant
    lasso_models = {}

    # Loop through each quadrant and fit a Lasso model
    for q in range(1, 5):
        # Get the quadrant data
        X, y = quadrant_splits[q][0], quadrant_splits[q][2]

        # Implement a custom cross-validation loop to find the best alpha
        best_alpha = None
        min_penalty = float('inf')

        for alpha in alphas:
            lasso = Lasso(alpha=alpha, random_state=10053)
            scores = cross_val_score(lasso, X, y, cv=5, scoring=custom_scorer)
            penalty = -np.mean(scores)

            if penalty < min_penalty:
                min_penalty = penalty
                best_alpha = alpha

        # Fit the Lasso model with the best alpha
        best_lasso = Lasso(alpha=best_alpha, random_state=10053)
        best_lasso.fit(X, y)

        # Get the feature weights
        coefficients = best_lasso.coef_

        # Get the indices of non-zero features
        non_zero_indices = np.nonzero(coefficients)[0]
        non_zero_features = X.columns[non_zero_indices]
        non_zero_weights = coefficients[non_zero_indices]

        # Store the Lasso model and feature weights in the dictionary
        lasso_models[q] = {
            'model': best_lasso,
            'weights': non_zero_weights,
            'non_zero_features': non_zero_features
        }

    # Create the subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.4*4, 3.8*4))

    # Define the colors for the positive and negative weights
    colors = np.array(['#d9534f', '#5cb85c'])

    # Loop through each quadrant and create a horizontal bar chart
    for q, ax in zip(range(1, 5), axes.flatten()):
        # Get the non-zero features and weights for the current quadrant
        features = lasso_models[q]['non_zero_features']
        weights = lasso_models[q]['weights']

        # Get the top positive and negative weights
        top_positive = weights.argsort()[::-1][:top_n]
        top_negative = weights.argsort()[:top_n]
        top_weights = np.hstack([top_positive, top_negative])

        # Set the color of the bars based on the sign of the weights
        bar_colors = colors[(weights[top_weights] > 0).astype(int)]

        # Create a bar chart showing the top positive and negative features
        sns.barplot(x=weights[top_weights],
                    y=features[top_weights], ax=ax, palette=bar_colors)
        ax.set_title(f"Quadrant {q}")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Feature")

    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig("images/features.png")

    # Create an HTML img tag to display the image
    img_tag = (f'<img src="images/features.png" alt="features"'
               f'style="display:block; margin-left:auto;'
               f'margin-right:auto; width:80%;">')

    # display the HTML <img> tag
    display(HTML(img_tag))
    plt.close()

    return lasso_models

def get_quadrant_reduced(df_final, lasso_models):
    """
    This function prepares a reduced dataset for each quadrant based on
    the non-zero features selected by the Lasso models. It slices
    the original dataset to only include the non-zero features for
    each quadrant, and then does a train-test split on the reduced dataset.
    It also stores the holdout indices for each split in a dictionary.
    The function returns the dictionary of reduced datasets and
    their corresponding train-test splits.
    """
    # Create an empty dictionary to store the train-test splits for each quadrant
    quadrant_reduced = {}

    # Loop through each quadrant and do the train-test split
    for q in range(1, 5):
        # Get the data for the current quadrant
        df_quadrant = df_final[df_final['Quadrant'] == q]

        # Get the non-zero features for this quadrant
        non_zero_features = lasso_models[q]['non_zero_features']

        # Slice the data to only use the non-zero features for this quadrant
        df_quadrant = df_quadrant[[
            'target', 'Quadrant'] + list(non_zero_features)]

        # Split the data into X and y
        X, y = df_quadrant.drop(['target', 'Quadrant'],
                                axis=1), df_quadrant['target']

        # Do the train-test split
        X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
            X, y, test_size=0.30, random_state=10053)

        # Add the train-test splits and the holdout indices to the dictionary
        quadrant_reduced[q] = (X_trainval, X_holdout,
                               y_trainval, y_holdout, X_holdout.index)

    return quadrant_reduced

def fit_classifiers(quadrant_reduced, forest_clf, forest_params, scoring):
    """
    This function fits a classifier for each quadrant, using a reduced set
    of features obtained from the Lasso models. It uses a pipeline with a
    random forest classifier and runs a grid search over specified hyperparameters
    to find the best combination of hyperparameters for each quadrant. The best
    classifier for each quadrant is then evaluated on a holdout test set and
    the f1 score is computed. The function returns a dataframe containing
    the f1 score and best hyperparameters for each quadrant and classifier.
    """
    # Define the pipeline
    pipeline = Pipeline([
        ('classifier', None)
    ])

    params_list = [forest_params]

    results_list = []

    for pipeline_params in params_list:
        pipeline = Pipeline([
            ('classifier', pipeline_params['classifier'][0])
        ])

        for q in range(1, 5):
            # Get the quadrant data
            X_trainval, X_holdout, y_trainval, y_holdout, holdout_indices = quadrant_reduced[q]

            # Set the classifier in the pipeline params
            pipeline_params['classifier'] = [forest_clf]

            # Run the grid search
            grid_search = GridSearchCV(
                pipeline,
                pipeline_params,
                cv=5,
                n_jobs=-1,
                scoring=scoring,
                refit='f1'
            ).fit(X_trainval, y_trainval)

            # Get the best estimator and its parameters from the grid search
            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Evaluate the best estimator on the test set
            y_pred = best_estimator.predict(X_holdout)
            f1 = f1_score(y_holdout, y_pred)

            # Create a dictionary to store the results
            model_name = type(
                best_estimator.named_steps['classifier']).__name__
            results_dict = {
                'Quadrant': [q],
                'Model': [model_name],
                'F1': [f1]
            }

            for i, param in enumerate(best_params.keys()):
                if i == 0:
                    continue
                results_dict[param] = [best_params[param]]

            # Append the results to the list
            results_list.append(results_dict)

    # Create a dataframe from the list of dictionaries
    results_df = pd.concat([pd.DataFrame(results_dict)
                           for results_dict in results_list], ignore_index=True)

    return results_df

def get_best_models_and_shap(quadrant_reduced, results_df):
    """
    This function trains the best models for each quadrant based on
    the results from the previous function, and calculates SHAP values
    for the holdout set of each quadrant. The function then stores the
    best model and SHAP explanation for each quadrant in a dictionary.
    The best model is a Random Forest Classifier with the optimal
    hyperparameters found in the previous function. The SHAP values
    are calculated using the SHAP library, and are based on the predicted
    probabilities of the best model. The SHAP values are stored in a
    SHAP Explanation object. The function returns the dictionary of
    best models and SHAP explanations for each quadrant.
    """
    # Define an empty dictionary to store the best models and shap explanations for each quadrant
    best_models_and_shap = {}

    # Loop through each quadrant
    for q in range(1, 5):
        # Get the train-test split for the current quadrant
        X_trainval, X_holdout, y_trainval, y_holdout, holdout_indices = quadrant_reduced[q]

        # Get the parameters for the best model for the current quadrant
        max_depth = int(
            results_df.loc[results_df['Quadrant'] == q, 'classifier__max_depth'])
        min_samples_leaf = int(
            results_df.loc[results_df['Quadrant'] == q, 'classifier__min_samples_leaf'])
        min_samples_split = int(
            results_df.loc[results_df['Quadrant'] == q, 'classifier__min_samples_split'])
        n_estimators = int(
            results_df.loc[results_df['Quadrant'] == q, 'classifier__n_estimators'])

        # Train the best model for the current quadrant
        best_model = RandomForestClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            n_estimators=n_estimators
        ).fit(X_trainval, y_trainval)

        # Store the best model for the current quadrant in the dictionary
        shap_explainer = shap.Explainer(
            best_model.predict_proba, X_trainval, feature_names=quadrant_reduced[q][0].columns, verbose=False)
        shap_values = shap_explainer(X_holdout)
        shap_explanation = shap.Explanation(shap_values.values[:, :, 1],
                                            shap_values.base_values[0][1],
                                            shap_values.data,
                                            feature_names=quadrant_reduced[q][0].columns)
        best_models_and_shap[q] = {
            'best_model': best_model,
            'shap_explanation': shap_explanation
        }

    return best_models_and_shap