import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN, OPTICS,
                             cluster_optics_dbscan)
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)

from scipy.cluster.hierarchy import dendrogram
from fastcluster import linkage

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer, WordNetLemmatizer

from tqdm.notebook import tqdm, trange
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image as WordImage
from IPython.display import Image
from IPython.display import display, display_html
from IPython.display import HTML

from plotnine import ggplot, aes, geom_bar
from plotnine.scales import scale_fill_manual
from plotnine import *
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# # Global settings
# %matplotlib inline

# Pandas settings
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)

# Error Filters
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


def read_data():
    """
    This function reads three csv files: rr-recipes.csv,
    rr-ratings.csv and recipes.csv.
    It then processes the data and returns two dataframes
    containing food and nutrients data and one dataframe
    containing ratings data.
    """
    # Read the csv files into dataframes
    food = pd.read_csv("rr-recipes.csv")
    ratings = pd.read_csv("rr-ratings.csv")
    nutrients = pd.read_csv("recipes.csv")
    ratings = (ratings.rename(columns={'userid': 'User_ID',
                                       'itemid': 'Food_ID',
                                       'rating': 'Rating'}))

    # Return the food and nutrients dataframes and the ratings dataframe
    return food, nutrients, ratings


def add_nutrients(food_raw, nutrients):
    """
    This function merges the food and nutrients dataframes
    based on the url of the recipe.
    """
    nutrients['url'] = nutrients['url'].apply(lambda x: x.split("/")[4])
    food_raw['url'] = food_raw['url'].apply(lambda x: x.split("/")[4])
    nutrients = pd.merge(nutrients, food_raw, on='url', how='inner')
    duplicates = nutrients.duplicated(subset=['url'], keep='first')
    food_raw = nutrients[~duplicates].drop_duplicates(
        subset=['url'], keep='first')
    return food_raw


def get_food(food_nutrients):
    """
    This function returns a dataframe containing food
    and their nutritional information.
    """
    # Call read_data function to obtain dataframes

    food = (food_nutrients[['itemid', 'title',
                           'prep_time', 'cook_time', 'ready_time',
                            'ingredients_y', 'directions_y',
                           'photo_url', 'servings', 'yield', 'calories',
                            'carbohydrates_g',
                           'sugars_g', 'fat_g', 'saturated_fat_g',
                           'cholesterol_mg', 'protein_g', 'dietary_fiber_g',
                           'sodium_mg', 'calories_from_fat']]
            .reset_index(drop=True))
    food = (food.rename(columns={'itemid': 'Food_ID',
                                 'title': 'Name',
                                 'ingredients_y': 'ingredients'}))
    return food


def df_exploration(df, display_df=True):
    """
    Displays the number of data points, nulls, and preview of the data frame.
    """
    display_html(f'<b>Number of Data Points:</b> {df.shape[0]}',
                 raw=True)
    display_html(f'<b>Number of Features:</b> {df.shape[1]}',
                 raw=True)
    display_html(f'<b>Number of Nulls:</b> {df.isna().sum().sum()}',
                 raw=True)

    if display_df:
        display(df.head(3))


def get_ratings(ratings, food_raw):
    """
    This function returns a dataframe containing food ratings.
    """
    food = food_raw.rename(columns={'itemid': 'Food_ID', 'title': 'Name'})
    ratings = (ratings.rename(columns={'userid': 'User_ID',
                                       'itemid': 'Food_ID',
                                       'rating': 'Rating'}))
    # Filter the ratings dataframe to include only the food
    # items that are in the food dataframe
    merged_df = (pd.merge(food,
                          ratings[['User_ID', 'Food_ID', 'Rating']],
                          on='Food_ID', how='inner'))

    user_counts = (merged_df.groupby('User_ID').size()
                   .reset_index(name='count'))
    popular_users = user_counts[user_counts['count'] > 100]['User_ID']
    merged_df = merged_df[merged_df['User_ID'].isin(popular_users)]
    merged_df = merged_df[['Food_ID', 'User_ID', 'Rating']]

    # Return the ratings dataframe
    return merged_df


def clean_recipe_text(text):
    """
    This function cleans the given recipe text by converting it to lowercase,
    removing stopwords, lemmatizing words and returning a cleaned string.

    :param text: Recipe text to be cleaned
    :return: Cleaned recipe text
    """
    stop_words = get_stop_words()
    lemmatizer = get_lemmatizer()

    if isinstance(text, str):
        # Convert the text to lowercase
        text = text.casefold()

        # Compile word pattern
        word_pattern = re.compile(r'\b[a-z\-]+\b')

        # Remove stopwords and lemmatize words
        text_list = [
            lemmatizer.lemmatize(word)
            for word in word_pattern.findall(text)
            if word not in stop_words
        ]

        # Join words to form cleaned text
        return ' '.join(text_list)
    else:
        return ''


def get_stop_words():
    """
    This function returns a set of stopwords to be removed from recipe text.
    """
    return frozenset(stopwords.words('english') + [
        'chef', 'easy', 'make', 'recipe', 'john'
    ] + list(STOPWORDS))


def get_lemmatizer():
    """
    This function returns an instance of a WordNetLemmatizer object.
    """
    return WordNetLemmatizer()


def plot_ratings(df_ratings):
    # Calculate the percentage of each rating value
    total = df_ratings.Rating.count()
    percent_plot = pd.DataFrame({"Total": df_ratings.Rating.value_counts()})
    percent_plot.reset_index(inplace=True)
    percent_plot.rename(columns={"index": "Rating"}, inplace=True)
    percent_plot["Percent"] = percent_plot["Total"].apply(
        lambda x: (x/total)*100)

    # Plot the first bar chart
    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1)
    sns.barplot(x="Rating", y="Total", data=percent_plot, color="#808080")
    plt.xlabel("Rating")
    plt.ylabel("Total")
    plt.title("Total Ratings per Rating Value")

    # Plot the second bar chart
    ax2 = plt.subplot(1, 2, 2)
    sns.barplot(x="Rating", y="Percent", data=percent_plot, color="#ff7f0e")
    plt.xlabel("Rating")
    plt.ylabel("Percent")
    plt.title("Percentage of Ratings per Rating Value")

    # Save the plot as a PNG file
    plt.savefig("plots.png")

    # Create an HTML img tag to display the image
    img_tag = (f'<img src="plots.png" alt="plots"'
               'style="display:block; margin-left:auto;'
               'margin-right:auto;width:80%;">')

    # Display the img tag in the Jupyter Notebook
    display(HTML(img_tag))

    # Close the plot
    plt.close()


def get_ingredient_matrix(food):
    """
    This function preprocesses the ingredients text in the food dataframe
    and returns a dataframe containing the TF-IDF matrix for the ingredients.

    :param food: Dataframe containing recipe details
    :return: Dataframe containing the TF-IDF matrix for the ingredients
    """
    # Clean the ingredient text using the clean_recipe_text function
    cleaned_ingredients = food['ingredients'].apply(
        lambda x: clean_recipe_text(x))

    # Initialize the TF-IDF vectorizer with appropriate parameters
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'[a-z-]+',
                                       stop_words='english',
                                       ngram_range=(1, 2),
                                       min_df=.01)

    # Compute the TF-IDF matrix for the cleaned ingredients
    corpus_food = tfidf_vectorizer.fit_transform(cleaned_ingredients)
    # Get the feature names for the TF-IDF matrix
    feature_names = tfidf_vectorizer.get_feature_names_out()
    # Extract the non-zero rows from the TF-IDF matrix
    # and convert it to a numpy array
    corpus_data = corpus_food[corpus_food.sum(axis=1).nonzero()[0]].toarray()
    # Create a dataframe from the TF-IDF matrix with appropriate column names
    ingredient_df = pd.DataFrame(corpus_data, columns=feature_names)
#     print(ingredient_df.index, get_food().index)
    # Merge the two DataFrames based on the index
    global main_df
    main_df = ingredient_df.merge(food, left_index=True, right_index=True)
    main_df.set_index('Name', inplace=True)

    # Return the dataframe
    return ingredient_df


def plot_variance_explained(data):
    """
    Plot the cumulative variance explained by increasing
    number of singular values for a given dataset.

    :param data: NumPy array or Pandas DataFrame
    :return: None
    """
    # Compute the number of components for the SVD
    n_components = data.shape[1]

    tsvd = TruncatedSVD(n_components)

    # Compute the SVD and get the transformed data
    X_tsvd = tsvd.fit_transform(data)

    # Compute the explained variance ratio
    var_exp = tsvd.explained_variance_ratio_
    var_exp_cumsum = np.cumsum(var_exp)
    global n_sv
    n_sv = np.argmax(var_exp_cumsum >= 0.8) + 1

    # Create a dataframe of the variance explained by each singular value
    var_df = pd.DataFrame({'Singular Value': range(1, len(var_exp)+1),
                           'Cumulative Variance Explained': var_exp_cumsum})
    theme_set(theme_bw())

    # Create the plot using ggplot
    plot = (ggplot(var_df, aes(x='Singular Value',
                               y='Cumulative Variance Explained',
                               color='orange')) +
            geom_line(color='orange') +
            scale_x_continuous(expand=(0, 0)) +
            scale_y_continuous(expand=(0, 0), limits=(0, 1)) +
            geom_vline(xintercept=n_sv, linetype='dashed', color='green') +
            geom_hline(yintercept=var_exp_cumsum[n_sv-1],
                       linetype='dashed', color='green') +
            labs(x='Number of Singular Values', y='Variance Explained',
                 title='Figure 7. Cumulative Variance Explained') +
            theme(axis_text=element_text(size=12),
                  axis_title=element_text(size=14,
                                          margin=dict(t=10, r=0, b=10, l=0))))

    # Save the plot as a PNG file
    ggsave(plot=plot, filename="plot_var_exp.png")

    # Create an HTML img tag to display the image
    img_tag = (f'<img src="plot_var_exp.png" alt="plot_var_exp"'
               'style="display:block; margin-left:auto;'
               'margin-right:auto;width:50%;">')

    # Display the img tag in the Jupyter Notebook
    display(HTML(img_tag))

def perform_svd(n_components, df_corpus):
    """
    This function performs the truncated singular value decomposition
    on the df_corpus. 
    """
    global final_tsvd
    final_tsvd = (TruncatedSVD(n_components=n_sv, random_state=10053)
                  .fit_transform(df_corpus))
    global feature_names
    feature_names = df_corpus.columns.tolist()
    # Create a dataframe from the SVD results
    df_svd = (pd.DataFrame(final_tsvd,
                           columns=[f'SV{i}' for i in range(1, n_sv+1)],
                           index=main_df.index))
    return df_svd


def evaluate_kmeans_clusters(final_tsvd):
    """
    This function evaluates the KMeans clusters for the input data
    and computes various cluster evaluation metrics
    for different number of clusters.

    :param final_tsvd: Dataframe containing the reduced feature matrix
    :return: Tuple containing the silhouette scores,
    Calinski-Harabasz scores, and Davies-Bouldin scores for different
    number of clusters
    """
    # Initialize lists to store the evaluation metrics
    # for different number of clusters
    global silhouette_scores
    global ch_scores
    global db_scores
    silhouette_scores = []
    ch_scores = []
    db_scores = []

    # Create a figure with subplots for each k value
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    axs = axs.flatten()

    # Iterate over different number of clusters and
    # compute the KMeans clusters and evaluation metrics
    for k, ax in zip(range(2, 11), axs):
        kmeans = KMeans(n_clusters=k, random_state=10053, n_init=10)
        y_predict = kmeans.fit_predict(final_tsvd)

        # Compute the cluster evaluation metrics
        silhouette = silhouette_score(final_tsvd, y_predict)
        ch = calinski_harabasz_score(final_tsvd, y_predict)
        db = davies_bouldin_score(final_tsvd, y_predict)

        # Add the evaluation metrics to the corresponding lists
        silhouette_scores.append(silhouette)
        ch_scores.append(ch)
        db_scores.append(db)

        # Create scatter plot of the clusters
        sns.scatterplot(x=final_tsvd[:, 0], y=final_tsvd[:, 1], ax=ax,
                        hue=y_predict, palette='bright', legend=False)
        ax.set_title(f'k={k}', fontsize=20)
        ax.set_xlabel('SV1')
        ax.set_ylabel('SV2')

    # Set the plot title and show the plot
    fig.suptitle('K-Means Clustering', fontsize=30)
    plt.show()

def show_internal_validation(silhouette_scores, ch_scores, db_scores):
    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, sharex=True, figsize=(6.4*3, 4.8*1))

    # Plot the Silhouette Coefficient on the first subplot
    (sns.lineplot(x=range(2, 11), y=silhouette_scores, color='blue',
                  marker='o', label='Silhouette Coefficient', ax=ax1))
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Coefficient')
    ax1.legend(loc='upper left')

    # Plot the Calinski-Harabasz index on the second subplot
    (sns.lineplot(x=range(2, 11), y=ch_scores,
                  color='red', marker='s',
                  label='Calinski-Harabasz Index', ax=ax2))
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Calinski-Harabasz Index')
    ax2.legend(loc='upper right')

    # Plot the Davies-Bouldin index on the third subplot
    (sns.lineplot(x=range(2, 11), y=db_scores, color='green',
                  marker='o', label='Davies-Bouldin Index', ax=ax3))
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Davies-Bouldin Index')
    ax3.legend(loc='upper left')

    # Set the plot title
    plt.suptitle(
        'K-Means Clustering Metrics', fontsize=30)

    # Show the plots
    plt.show()


def run_kmeans(final_tsvd, ingredient_df, feature_names):
    """
    Run KMeans clustering with 2 clusters on the input data
    and generates a scatter plot of the clustered data points.
    """
    # create a KMeans object with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=10053, n_init=10)

    # fit the KMeans model to the standardized data
    kmeans.fit(final_tsvd)

    # get the cluster labels for each data point
    labels = kmeans.labels_

    # create a scatter plot of the clustered data points
    (sns.scatterplot(x=final_tsvd[:, 0], y=final_tsvd[:, 1],
                     hue=labels, palette='Reds', alpha=0.5))

    # set the plot title and labels
    plt.title('Final K-Means Clustering')
    plt.xlabel('SV 1')
    plt.ylabel('SV 2')

    # Save the plot as a PNG file
    plt.savefig("kmeans.png")

    # Create an HTML img tag to display the image
    img_tag = (f'<img src="kmeans.png" alt="kmeans"'
               'style="display:block; margin-left:auto;'
               'margin-right:auto; width:80%;">')

    # display the HTML <img> tag
    display(HTML(img_tag))

    plt.close()
    # Create a dataframe containing the clusters
    # and their corresponding data points
    df_clusters = pd.DataFrame(data=ingredient_df, columns=feature_names)
    df_clusters['Cluster'] = labels

    # Create a global variable containing the indexes of 
    # the data points in each cluster
    global cluster_indexes
    cluster_indexes = df_clusters.groupby(
        'Cluster').apply(lambda x: x.index.tolist())


def plot_dendrograms(final_tsvd):
    # Compute linkage matrices
    dendo_single = linkage(final_tsvd, method='single')
    dendo_complete = linkage(final_tsvd, method='complete')
    dendo_average = linkage(final_tsvd, method='average')
    dendo_ward = linkage(final_tsvd, method='ward')

    # Set up the plot with four subplots in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Plot the single method dendrogram
    dendrogram(dendo_single, ax=axs[0, 0], truncate_mode='level', p=20)
    axs[0, 0].set_title("Single Method")
    axs[0, 0].spines['bottom'].set_color('#CC7722')
    axs[0, 0].spines['top'].set_color('#CC7722')
    axs[0, 0].spines['right'].set_color('#CC7722')
    axs[0, 0].spines['left'].set_color('#CC7722')
    axs[0, 0].tick_params(axis='x', colors='#CC7722')
    axs[0, 0].tick_params(axis='y', colors='#CC7722')

    # Plot the complete method dendrogram
    dendrogram(dendo_complete, ax=axs[0, 1], truncate_mode='level', p=5)
    axs[0, 1].set_title("Complete Method")
    axs[0, 1].spines['bottom'].set_color('#CC7722')
    axs[0, 1].spines['top'].set_color('#CC7722')
    axs[0, 1].spines['right'].set_color('#CC7722')
    axs[0, 1].spines['left'].set_color('#CC7722')
    axs[0, 1].tick_params(axis='x', colors='#CC7722')
    axs[0, 1].tick_params(axis='y', colors='#CC7722')

    # Plot the average method dendrogram
    dendrogram(dendo_average, ax=axs[1, 0], truncate_mode='level', p=5)
    axs[1, 0].set_title("Average Method")
    axs[1, 0].spines['bottom'].set_color('#CC7722')
    axs[1, 0].spines['top'].set_color('#CC7722')
    axs[1, 0].spines['right'].set_color('#CC7722')
    axs[1, 0].spines['left'].set_color('#CC7722')
    axs[1, 0].tick_params(axis='x', colors='#CC7722')
    axs[1, 0].tick_params(axis='y', colors='#CC7722')

    # Plot the ward method dendrogram
    dendrogram(dendo_ward, ax=axs[1, 1], truncate_mode='level', p=2)
    axs[1, 1].set_title("Ward Method")
    axs[1, 1].spines['bottom'].set_color('#CC7722')
    axs[1, 1].spines['top'].set_color('#CC7722')
    axs[1, 1].spines['right'].set_color('#CC7722')
    axs[1, 1].spines['left'].set_color('#CC7722')
    axs[1, 1].tick_params(axis='x', colors='#CC7722')
    axs[1, 1].tick_params(axis='y', colors='#CC7722')

    # Set the plot layout
    fig.tight_layout(pad=2)
    
    plt.savefig("dendrograms.png")

    # Create an HTML img tag to display the image
    img_tag = (f'<img src="dendrograms.png" alt="dendrograms"'
               'style="display:block;margin-left:auto;'
               'margin-right:auto;width:80%;">')

    # Display the img tag in the Jupyter Notebook
    display(HTML(img_tag))
    plt.close()


def create_word_clouds(food, cluster_indexes):
    """
    This function creates word clouds from
    the food names in the different clusters.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    stop_words = (list(set(stopwords.words('english') + list(STOPWORDS))) +
                  ['i', 'ii', 'iii',  'recipe', 'chef', 'john',
                   'shrimp', 'salmon', 'salad', 'dip', 'cake', 'let'])

    # create word cloud for cluster 1
    from matplotlib.colors import ListedColormap
    df_new_name = food['Name'].iloc[cluster_indexes[0]].apply(
        lambda x: clean_recipe_text(x))
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'[a-z-]+',
                                       stop_words=stop_words,
                                       ngram_range=(1, 2),
                                       min_df=.001)
    corpus_name = tfidf_vectorizer.fit_transform(df_new_name)
    corpus_labels = tfidf_vectorizer.get_feature_names_out()
    corpus_data = corpus_name[corpus_name.sum(axis=1).nonzero()[0]].toarray()
    df_name2 = pd.DataFrame(corpus_data, columns=corpus_labels)
    series_summary = df_name2.sum().sort_values(ascending=False)
    cupcake_colors = (['#FCE6DC', '#FBE5D6', '#F9D2C6', '#F6B8AF', '#F09E96',
                       '#E88985', '#D76C6E', '#C1545A', '#AF3F47', '#9E2B35'])

    # create a ListedColormap object from the color palette
    cupcake_cmap = ListedColormap(cupcake_colors)

    # load the mask image
    mask = np.array(WordImage.open("cupcake.png"))

    # create the wordcloud with the cupcake color scheme
    wordcloud = (WordCloud(mask=mask,
                          background_color='white',
                          contour_width=2,
                          contour_color='orange',
                          stopwords=stop_words,
                          colormap=cupcake_cmap)
                 .generate_from_frequencies(series_summary))

    # display the wordcloud
    axes[0].imshow(wordcloud)
    axes[0].axis('off')
    axes[0].set_title('Sweet Treats', fontsize=20)

    # create word cloud for cluster 2
    df_new_name = food['Name'].iloc[cluster_indexes[1]].apply(
        lambda x: clean_recipe_text(x))
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'[a-z-]+',
                                       stop_words=stop_words,
                                       ngram_range=(1, 2),
                                       min_df=.001)
    corpus_name = tfidf_vectorizer.fit_transform(df_new_name)
    corpus_labels = tfidf_vectorizer.get_feature_names_out()
    corpus_data = corpus_name[corpus_name.sum(axis=1).nonzero()[0]].toarray()
    df_name2 = pd.DataFrame(corpus_data, columns=corpus_labels)
    series_summary = df_name2.sum().sort_values(ascending=False)

    chicken_colors = (['#FFA500', '#FFD700', '#FF6347', '#8B0000',
                       '#996515', '#C4AE66', '#F5DEB3', '#CD853F'])

    # create a ListedColormap object from the color palette
    chicken_cmap = ListedColormap(chicken_colors)
    mask = np.array(WordImage.open("chicken.png"))
    wordcloud = (WordCloud(mask=mask,
                          background_color='white',
                          contour_width=2,
                          contour_color='orange',
                          stopwords=stop_words,
                          colormap=chicken_cmap)
                 .generate_from_frequencies(series_summary))

    # display the word cloud
    axes[1].imshow(wordcloud)
    axes[1].axis('off')
    axes[1].set_title('Savory Eats', fontsize=20)

    # show the plots
    plt.show()


def display_top_10_foods(df_ratings, df_food):
    """
    This function returns the top 10 foods in terms of ratings. 
    """
    # Group the ratings by Food_ID and calculate the total number
    # of reviews and total rating for each food
    df_grouped = (df_ratings.groupby('Food_ID')['Rating']
                  .agg(['sum', 'count']).reset_index())
    (df_grouped.rename(columns={'sum': 'Total Rating',
                                'count': 'Number of Reviews'}, inplace=True))

    # Sort the data by number of reviews in descending order
    df_sorted = (df_grouped.sort_values(['Number of Reviews',
                                         'Total Rating', ], ascending=False))

    # Get the top 10 foods by number of reviews
    top_10 = df_sorted.head(10)

    # Define a list to hold the rows of the table
    rows = []

    # Define the number of rows and columns in the table
    num_rows = 5
    num_cols = 2

    # Loop over the rows of the table
    for i in range(num_rows):
        # Define a list to hold the cells in the current row
        cells = []

        # Loop over the cells in the current row
        for j in range(num_cols):
            # Compute the index of the current food in the top 10 list
            index = i * num_cols + j

            if index < len(top_10):
                # Get the food ID, name, sodium content,
                # total rating, and number of reviews for the current food
                food_id = top_10.iloc[index]['Food_ID']
                name = df_food.loc[df_food['Food_ID']
                                   == food_id, 'Name'].iloc[0]
                sodium = df_food.loc[df_food['Food_ID']
                                     == food_id, 'sodium_mg'].iloc[0]
                food_url = df_food.loc[df_food['Food_ID']
                                       == food_id, 'photo_url'].iloc[0]
                total_rating = top_10.iloc[index]['Total Rating']
                num_reviews = top_10.iloc[index]['Number of Reviews']

                # Define a cell for the current food with an image, name,
                # sodium content, total rating, and number of reviews
                cell = f'<td style="width: 50%; text-align: center;">'
                f'<img src="{food_url}" style="display: inline-block;'
                f'vertical-align: middle; max-width: 100%; max-height: 100%;">'
                f'<br>{name}<br>Sodium(mg): {sodium}<br>'
                f'Total Rating: {total_rating}<br>'
                f'Number of Reviews: {num_reviews}</td>'
            else:
                cell = '<td></td>'

            # Add the cell to the list of cells in the current row
            cells.append(cell)

        # Join the cells into a row and add it to the list of rows
        row = f'<tr>{"".join(cells)}</tr>'
        rows.append(row)

    # Join the rows into a table
    table = f'<table style="width: 100%;">{"".join(rows)}</table>'

    # Add a label to the table
    label = f'<h3>Top 10 Foods with the Most Number of Reviews</h3>'

    # Display the table
    display(HTML(label + table))


def display_top_foods(df_ratings, df_food):
    """
    This function returns the top 10 foods in terms of number of ratings. 
    """
    # Group the ratings by Food_ID and calculate the total number of reviews
    # and total rating for each food
    df_grouped = df_ratings.groupby('Food_ID')['Rating'].agg(
        ['sum', 'count']).reset_index()
    df_grouped.rename(columns={'sum': 'Total Rating',
                      'count': 'Number of Reviews'}, inplace=True)

    # Sort the data by number of reviews in descending order
    df_sorted = df_grouped.sort_values(
        ['Number of Reviews', 'Total Rating', ], ascending=False)

    # Get the top 10 foods by number of reviews
    top_10 = df_sorted.head(10)

    # Define a list to hold the rows of the table
    rows = []

    # Define the number of rows and columns in the table
    num_rows = 5
    num_cols = 2

    # Loop over the rows of the table
    for i in range(num_rows):
        # Define a list to hold the cells in the current row
        cells = []

        # Loop over the cells in the current row
        for j in range(num_cols):
            # Compute the index of the current food in the top 10 list
            index = i * num_cols + j

            if index < len(top_10):
                food_id = top_10.iloc[index]['Food_ID']
                name = df_food.loc[df_food['Food_ID']
                                   == food_id, 'Name'].iloc[0]
                sodium = df_food.loc[df_food['Food_ID']
                                     == food_id, 'sodium_mg'].iloc[0]
                food_url = df_food.loc[df_food['Food_ID']
                                       == food_id, 'photo_url'].iloc[0]
                total_rating = top_10.iloc[index]['Total Rating']
                num_reviews = top_10.iloc[index]['Number of Reviews']

                cell = (f'<td style="width: 50%; text-align: center;">'
                        f'<img src="{food_url}" style="display: inline-block;'
                        f'vertical-align: middle; max-width: 100%;'
                        f'max-height: 100%;"><br>{name}<br>'
                        f'Sodium(mg): {sodium}<br>Total Rating: {total_rating}'
                        f'<br>Number of Reviews: {num_reviews}</td>')
            else:
                cell = '<td></td>'

            # Add the cell to the list of cells in the current row
            cells.append(cell)

        # Join the cells into a row and add it to the list of rows
        row = f'<tr>{"".join(cells)}</tr>'
        rows.append(row)

    # Join the rows into a table
    table = f'<table style="width: 100%;">{"".join(rows)}</table>'

    # Add a label to the table
    label = f'<h3>Top 10 Foods with the Most Number of Reviews</h3>'

    # Display the table
    display(HTML(label + table))


def display_top_10_sodium(df_food):
    """
    This function returns the top 10 foods with the
    highest number of sodium content (in mg). 
    """
    df_food = df_food[df_food.sodium_mg != "6 large muffins"]
    df_food.sodium_mg = df_food.sodium_mg.apply(pd.to_numeric)
    df_sorted = df_food.sort_values(by='sodium_mg', ascending=False)
    top_10 = df_sorted.head(10)

    # Define a list to hold the rows of the table
    rows = []

    # Define the number of rows and columns in the table
    num_rows = 5
    num_cols = 2

    # Loop over the rows of the table
    for i in range(num_rows):
        # Define a list to hold the cells in the current row
        cells = []

        # Loop over the cells in the current row
        for j in range(num_cols):
            # Compute the index of the current food in the top 10 list
            index = i * num_cols + j

            if index < len(top_10):
                food_id = top_10.iloc[index]['Food_ID']
                name = df_food.loc[df_food['Food_ID']
                                   == food_id, 'Name'].iloc[0]
                sodium = df_food.loc[df_food['Food_ID']
                                     == food_id, 'sodium_mg'].iloc[0]
                food_url = df_food.loc[df_food['Food_ID']
                                       == food_id, 'photo_url'].iloc[0]
                servings = top_10.iloc[index]['servings']

                cell = (f'<td style="width: 50%; text-align: center;">'
                        f'<img src="{food_url}" style="display: inline-block;'
                        f'vertical-align: middle; max-width: 100%;'
                        f'max-height: 100%;"><br>{name}<br>'
                        f'Sodium(mg): {sodium}<br>Servings: {servings}<br>'
                        f'Sodium Content Per Serving:'
                        f'{round(sodium/servings,2)}</td>')
            else:
                cell = '<td></td>'

            # Add the cell to the list of cells in the current row
            cells.append(cell)

        # Join the cells into a row and add it to the list of rows
        row = f'<tr>{"".join(cells)}</tr>'
        rows.append(row)

    # Join the rows into a table
    table = f'<table style="width: 100%;">{"".join(rows)}</table>'

    # Add a label to the table
    label = (f'<h3>Top 10 Foods with the'
             'Most Number of Sodium Content (in mg)</h3>')

    # Display the table
    display(HTML(label + table))


def display_top_10_sodium_serving(df_food):
    """
    This function returns the top 10 foods with the
    highest number of sodium content (in mg) per serving. 
    """
    df_food = df_food[df_food.sodium_mg != "6 large muffins"]
    df_food.sodium_mg = df_food.sodium_mg.apply(pd.to_numeric)
    df_food['sodium_serving'] = df_food.sodium_mg / df_food.servings
    df_sorted = df_food.sort_values(by='sodium_serving', ascending=False)

    top_10 = df_sorted.head(10)
    # Define a list to hold the rows of the table
    rows = []

    # Define the number of rows and columns in the table
    num_rows = 5
    num_cols = 2

    # Loop over the rows of the table
    for i in range(num_rows):
        # Define a list to hold the cells in the current row
        cells = []

        # Loop over the cells in the current row
        for j in range(num_cols):
            # Compute the index of the current food in the top 10 list
            index = i * num_cols + j

            if index < len(top_10):
                food_id = top_10.iloc[index]['Food_ID']
                name = df_food.loc[df_food['Food_ID']
                                   == food_id, 'Name'].iloc[0]
                sodium = df_food.loc[df_food['Food_ID']
                                     == food_id, 'sodium_mg'].iloc[0]
                food_url = df_food.loc[df_food['Food_ID']
                                       == food_id, 'photo_url'].iloc[0]
                servings = top_10.iloc[index]['servings']
                sodium_servings = top_10.iloc[index]['sodium_serving']

                cell = (f'<td style="width: 50%; text-align: center;">'
                        f'<img src="{food_url}" style="display: inline-block;'
                        f'vertical-align: middle; max-width: 100%;'
                        f'max-height: 100%;"><br>{name}<br>'
                        f'Sodium(mg): {sodium}<br>Servings: {servings}<br>'
                        f'Sodium Content Per Servings:'
                        f'{round(sodium_servings,2)}</td>')
            else:
                cell = '<td></td>'

            # Add the cell to the list of cells in the current row
            cells.append(cell)

        # Join the cells into a row and add it to the list of rows
        row = f'<tr>{"".join(cells)}</tr>'
        rows.append(row)

    # Join the rows into a table
    table = f'<table style="width: 100%;">{"".join(rows)}</table>'

    # Add a label to the table
    label = (f'<h3>Top 10 Foods with the Most Number of '
             'Sodium content (in mg) per Serving</h3>')

    # Display the table
    display(HTML(label + table))


def display_low_10_sodium_serving(df_food):
    """
    This function returns the top 10 foods with the least number of
    sodium content (in mg) per serving. 
    """
    df_food = df_food[df_food.sodium_mg != "6 large muffins"]
    df_food.sodium_mg = df_food.sodium_mg.apply(pd.to_numeric)
    df_food['sodium_serving'] = df_food.sodium_mg / df_food.servings
    df_sorted = df_food.sort_values(by='sodium_serving', ascending=True)

    bottom_10 = df_sorted.head(10)

    # Define a list to hold the rows of the table
    rows = []

    # Define the number of rows and columns in the table
    num_rows = 5
    num_cols = 2

    # Loop over the rows of the table
    for i in range(num_rows):
        # Define a list to hold the cells in the current row
        cells = []

        # Loop over the cells in the current row
        for j in range(num_cols):
            # Compute the index of the current food in the top 10 list
            index = i * num_cols + j

            if index < len(bottom_10):
                food_id = bottom_10.iloc[index]['Food_ID']
                name = df_food.loc[df_food['Food_ID']
                                   == food_id, 'Name'].iloc[0]
                sodium = df_food.loc[df_food['Food_ID']
                                     == food_id, 'sodium_mg'].iloc[0]
                food_url = df_food.loc[df_food['Food_ID']
                                       == food_id, 'photo_url'].iloc[0]
                servings = bottom_10.iloc[index]['servings']
                sodium_servings = bottom_10.iloc[index]['sodium_serving']

                cell = (f'<td style="width: 50%; text-align: center;">'
                        f'<img src="{food_url}" style="display:'
                        f'inline-block; vertical-align: middle;'
                        f'max-width: 100%; max-height: 100%;"><br>{name}<br>'
                        f'Sodium(mg): {sodium}<br>Servings: {servings}<br>'
                        f'Sodium Content Per Servings: '
                        f'{round(sodium_servings,2)}</td>')
            else:
                cell = '<td></td>'

            # Add the cell to the list of cells in the current row
            cells.append(cell)

        # Join the cells into a row and add it to the list of rows
        row = f'<tr>{"".join(cells)}</tr>'
        rows.append(row)

    # Join the rows into a table
    table = f'<table style="width: 100%;">{"".join(rows)}</table>'

    # Add a label to the table
    label = (f'<h3>Top 10 Foods with the Least Number of'
             'Sodium content (in mg) per serving</h3>')

    # Display the table
    display(HTML(label + table))


def food_recommendation1(food, ratings, Food_Name):
    """
    This function takes in a food name as input and returns a list of 3 
    recommended similar foods for the 1st cluster.
    """
    try:
        cluster0 = food.iloc[cluster_indexes[0]]

        ratings0 = ratings[ratings['Food_ID'].isin(cluster0['Food_ID'])]

        # Pivot the ratings data to create a user-item matrix
        dataset = ratings0.pivot(
            index='Food_ID', columns='User_ID', values='Rating')
        dataset.fillna(0, inplace=True)
        csr_dataset = csr_matrix(dataset.values)
        dataset.reset_index(inplace=True)

        # Train a nearest neighbors model on the ratings data
        model = NearestNeighbors(
            metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        model.fit(csr_dataset)

        n = 3
        FoodList = food[food['Name'].str.contains(Food_Name)]
        if len(FoodList):
            Foodi_index = dataset[dataset['Food_ID'] ==
                                  FoodList.iloc[0]['Food_ID']].index[0]

            distances, indices = model.kneighbors(
                csr_dataset[Foodi_index], n_neighbors=n+1)
            Food_indices = sorted(list(zip(indices.squeeze().tolist(
            ), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

            Recommendations = []
            for val in Food_indices:
                Foodi_index = val[0]
                i = food[food['Food_ID'] ==
                         dataset.iloc[Foodi_index]['Food_ID']].index
                if not i.empty:
                    food_name = food.iloc[i[0]]['Name']
                    food_sodium = food.iloc[i[0]]['sodium_mg']
                    food_image = (Image(
                        url=food.iloc[i[0]]['photo_url'],
                        width=150, height=150))
                    food_url = url = food.iloc[i[0]]['photo_url']
                    (Recommendations.append(
                        {'Name': food_name, 'Image': food_image,
                         'Sodium': food_sodium, 'Url': food_url}))

            # Return the list of recommended similar foods
            if Recommendations:
                # Create the rows for the HTML table
                rows = []
                for recommendation in Recommendations:
                    image = recommendation["Image"]
                    food_url = recommendation["Url"]
                    name = recommendation['Name']
                    sodium = recommendation['Sodium']
                    cell = (f'<td style="width: 33.33%; text-align: center;">'
                            f'<img src="{food_url}"'
                            f'style="display: inline-block;'
                            f'vertical-align: middle; max-width: 100%;'
                            f'max-height: 100%;"><br>{name}<br>'
                            f'Sodium(mg): {sodium}</td>')
                    rows.append(cell)
                rows = [f'<tr><td colspan="3" style="text-align:center;'
                        f'font-weight:bold;font-size:24px;">'
                        f'Top 3 Sweet Treats</td></tr>'] + rows[:3]

                # Create the table HTML code
                table = (f'<table style="width: 100%;">'
                         f'<tr>{"".join(rows)}</tr>'
                         f'</table>')

                # Display the table
                display(HTML(table))
            else:
                print("No Similar Foods.")
        else:
            print("No Similar Foods.")
    except:
        print("No Similar Foods.")


def food_recommendation2(food, ratings, Food_Name):
    """
    This function takes in a food name as input and returns a list of 3
    recommended similar foods for the 2nd cluster.
    """
    try:
        cluster0 = food.iloc[cluster_indexes[1]]

        ratings0 = ratings[ratings['Food_ID'].isin(cluster0['Food_ID'])]
        # Pivot the ratings data to create a user-item matrix
        dataset = ratings0.pivot(
            index='Food_ID', columns='User_ID', values='Rating')
        dataset.fillna(0, inplace=True)
        csr_dataset = csr_matrix(dataset.values)
        dataset.reset_index(inplace=True)
        # Train a nearest neighbors model on the ratings data
        model = NearestNeighbors(
            metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        model.fit(csr_dataset)

        n = 3
        FoodList = food[food['Name'] == Food_Name]
        if len(FoodList):
            Foodi_index = dataset[dataset['Food_ID'] ==
                                  FoodList.iloc[0]['Food_ID']].index[0]

            distances, indices = model.kneighbors(
                csr_dataset[Foodi_index], n_neighbors=n+1)
            Food_indices = sorted(list(zip(indices.squeeze().tolist(
            ), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

            Recommendations = []
            for val in Food_indices:
                Foodi_index = val[0]
                i = food[food['Food_ID'] ==
                         dataset.iloc[Foodi_index]['Food_ID']].index
                if not i.empty:
                    food_name = food.iloc[i[0]]['Name']
                    food_sodium = food.iloc[i[0]]['sodium_mg']
                    food_image = (Image(
                        url=food.iloc[i[0]]['photo_url'],
                        width=150, height=150))
                    food_url = url = food.iloc[i[0]]['photo_url']
                    (Recommendations.append(
                        {'Name': food_name, 'Image': food_image,
                         'Sodium': food_sodium, 'Url': food_url}))

            # Return the list of recommended similar foods
            if Recommendations:
                rows = []
                for recommendation in Recommendations:
                    image = recommendation["Image"]
                    food_url = recommendation["Url"]
                    name = recommendation['Name']
                    sodium = recommendation['Sodium']
                    cell = (f'<td style="width: 33.33%; text-align: center;">'
                            f'<img src="{food_url}" style="display: '
                            f'inline-block; vertical-align: middle; '
                            f'max-width: 100%; max-height: 100%;"><br>{name}'
                            f'<br>Sodium(mg): {sodium}</td>')
                    
                    rows.append(cell)
                rows = [
                    f'<tr><td colspan="3" style="text-align:center; '
                    f'font-weight:bold;font-size:24px;">'
                    f'Top 3 Savory Eats</td></tr>'] + rows[:3]

                # Create the table HTML code
                table = (f'<table style="width: 100%;"><tr>'
                         f'{"".join(rows)}</tr></table>')

                # Display the table
                display(HTML(table))
            else:
                print("No Similar Foods.")
        else:
            print("No Similar Foods.")
    except:
        print("No Similar Foods.")