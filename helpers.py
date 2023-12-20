import requests
import numpy as np
from bs4 import BeautifulSoup
import ast
from imdb import Cinemagoer
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import re
import string
from itertools import combinations
from collections import Counter, defaultdict
from flair.models import SequenceTagger
from flair.data import Sentence
from fuzzywuzzy import fuzz, process
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from SPARQLWrapper import SPARQLWrapper, JSON
import time
from nltk.corpus import stopwords
from sklearn.cluster import KMeans, DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import io
from scipy import stats
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tqdm import tqdm


######################################### DOWNLOADING NLTK PACKAGEs ####################################################

nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

######################################### COMPLEMENTING DATASETS #######################################################


def wikipedia_query(save_file=True, save_path='DATA/', filename='wiki_queries.csv'):
    """
    Retrieves IMDB, freebase ID  and revenue of movies on wikipedia. If the query crashes, the request is made again after
    5s. for a maximal of 10 tries.
    :param save_file: boolean: whether to save the created dataframe in a csv file, default = True
    :param save_path: string: path where the data will be saved, default = 'DATA/'
    :param filename: string: name of the file to save, default = 'wiki_queries.csv'
    :return: a dataframe containing IMDB ID, freebase ID and revenue
    """
    # Call the wikidata query service
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # Create the query
    sparql.setQuery("""
    SELECT ?work ?imdb_id ?freebase_id ?revenue
    WHERE
    {
      ?work wdt:P31/wdt:P279* wd:Q11424.
      ?work wdt:P345 ?imdb_id.
      
      OPTIONAL {?work wdt:P2142 ?revenue.}
      OPTIONAL {?work wdt:P646 ?freebase_id.}

      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    """)

    # Set the return format to JSON
    sparql.setReturnFormat(JSON)

    max_retries = 10
    retry_delay = 5

    for attempt in range(max_retries):
        print(f'Attempt number {attempt + 1}')

        try:
            # Execute the query and convert results to a DataFrame
            results = sparql.query().convert()
            print('Wikipedia query successful')
            bindings = results['results']['bindings']

            # Extracting IMDb, Wikipedia, Freebase IDs, and labels
            data = []
            for binding in bindings:
                row = {
                    'work': binding['work']['value'] if 'work' in binding else None,
                    'IMDB_ID': binding['imdb_id']['value'] if 'imdb_id' in binding else None,
                    'freebase_ID': binding['freebase_id']['value'] if 'freebase_id' in binding else None,
                    'box_office_revenue': binding['revenue']['value'] if 'revenue' in binding else None,
                }
                data.append(row)

            # Create a DataFrame
            wiki_df = pd.DataFrame(data)

            # remove duplicates
            wiki_df_filtered = wiki_df.drop_duplicates('IMDB_ID', keep='first')

            if save_file:
                wiki_df_filtered.to_csv(save_path + filename, index=False)
                print(f'file {save_path + filename} saved')

            return wiki_df_filtered

        except Exception as e:
            print(f"An error occurred: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Exiting.")


def get_history_timeline(save_path='DATA/', file_name='timeline.csv'):
    """
    Creates a historical timeline of events related to women's rights in the US and save it in a csv.
    Gets the dates and events descriptions from https://www.history.com/topics/womens-history/womens-history-us-timeline
    :param save_path: string: where to save the resulting csv file
    :param file_name: string: name of the csv file
    """
    url = 'https://www.history.com/topics/womens-history/womens-history-us-timeline'
    response = requests.get(url)
    timeline = pd.DataFrame(columns=['Date', 'Event'])
    dates = []
    descriptions = []

    if response.status_code == 200:
        print('Successfully accessed ' + url)

        # Use BeautifulSoup to parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all <p> tags = tags that contain the dates and events descriptions
        events = soup.find_all('p')

        # only iterate on paragraphs that correspond to events
        for event in events[6:-15]:
            # paragraphs that contains a date in the title and do not start by READ MORE
            if 'READ MORE' not in event.text and event.strong is not None:
                description = event.text
                date = event.strong.text
                # remove ':' and select only the last characters to get the year
                dates.append(int(event.strong.text.replace(':', '')[-5:]))
                # remove the date from the description
                descriptions.append(description.replace(date, '').replace(':', ''))

        timeline = pd.DataFrame({'Date': dates, 'Event': descriptions})
    else:
        print('could not access ' + url)

    timeline.to_csv(save_path + file_name, index=False)
    print(f'Timeline saved in {save_path + file_name}')


def filter_titles_IDs(dataset):
    """
    Filters a dataset of titles from IMDB by keeping only movies that came out between 2010 and 2022. It also removes
    adult movies and movies with unknown start year.
    :param dataset: panda dataframe: dataset to filter
    :return: list of string: the IMDB ids corresponding to the movies that are left after filtering
    """

    filtered_dataset = dataset[(dataset['titleType'] == 'tvMovies') | (dataset['titleType'] == 'movie')]
    filtered_dataset = filtered_dataset[(filtered_dataset['isAdult'] == 0) & (filtered_dataset['startYear'] != '\\N')]
    # filter years
    filtered_dataset['startYear'] = pd.to_numeric(filtered_dataset['startYear'])
    filtered_dataset = filtered_dataset[(filtered_dataset['startYear'] < 2023) & (filtered_dataset['startYear'] > 2010)]

    # get IDs
    movie_IDs = filtered_dataset['tconst'].drop_duplicates()

    return movie_IDs


def get_movie_info(toget, list_to_complete, movie, imdb_id):
    """
    Gets the 'toget' (e.g. plot, title, countries, ...) information of a specific movie (Cinemagoer object) and appends
    it to list_to_complete. If 'toget' is not found, it raises an error and appends None to the list.
    :param toget: string: information we want to be retrieved from IMDB
                can be: 'plot', 'title', 'countries', 'year', 'genres', 'languages' or 'cast'
    :param list_to_complete: list to which 'toget' is appended
    :param movie: cinemagoer movie object: movie we want to retrieve information about
    :param imdb_id: string: IMDB id of the movie we want to retrieve information about
    :return: list with the information about the movie appended or 'None' if it was not available on IMDB
    """
    try:
        list_to_complete.append(movie[toget])
    except KeyError:
        print(f"{toget} not available for movie {imdb_id}")
        list_to_complete.append(None)

    return list_to_complete


def get_IMDB_movies_data(IMDB_ids, save_path='DATA/', file_name='IMDB_movies_2010-2022.csv'):
    """
    Use cinemagoer to get plots, titles, countries, year, genres and languages from movies with the ids in IMDB_ids
    Creates a dataframe with these information and saves it as a csv
    :param IMDB_ids: list of string: IMDB identifier of the movies we want to get data on
    :param file_name: string: name of the file to save
    :param save_path: string: path to which the file will be saved
    """

    # create an instance of the cinemagoer class
    ia = Cinemagoer()

    # data we want to get from IMDB
    IMDB_IDs = []
    plots = []
    titles = []
    countries = []
    years = []
    genres = []
    languages = []
    i = 1

    for id in IMDB_ids:
        try:
            # get movie from IMDB
            movie = ia.get_movie(id[2:])  # the first 2 characters of the ID need to be removed
        except Exception as e:
            print(f"An error occurred for movie {id}: {e}")
        try:
            title = movie['title']
            print(f'loaded movie {i}: {title}')
            titles.append(movie['title'])
        except KeyError:
            print(f"Title not available for movie {id}")
            titles.append(None)

        plots = get_movie_info('plot', plots, movie, id)
        years = get_movie_info('year', years, movie, id)
        countries = get_movie_info('countries', countries, movie, id)
        genres = get_movie_info('genres', genres, movie, id)
        languages = get_movie_info('languages', languages, movie, id)
        IMDB_IDs.append(id)
        i += 1

    # create data frame with the scraped data
    IMDB_data = pd.DataFrame(
        {'IMDB_ID': IMDB_IDs, 'name': titles, 'release_date': years, 'languages': languages, 'countries': countries,
         'genre': genres, 'plot_summary': plots})

    # save it
    IMDB_data.to_csv(save_path + file_name, index=False)


def filter_IMDB_movie_dataset(data):
    """
    Filter the IMDB movie dataset: Remove from the data rows with unknown countries or languages. Remove rows that do
    not have United States as one of their countries and English or American Sign as one of their languages.
    Remove duplicated movies if any.
    :param data: panda dataframe: dataframe to filter
    :return: the filtered dataframe
    """
    # remove movies that still have an unknown country or unknown language
    data = data[pd.notna(data['countries'])]
    data = data[pd.notna(data['languages'])]
    # keep only US movies
    IMDB_data_us = data[data['countries'].str.contains('United States', case=False)].copy()
    IMDB_data_us_en = IMDB_data_us[
        IMDB_data_us['languages'].str.contains('English' or 'American Sign', case=False)].copy()
    # keep only movies from 2010-2022
    IMDB_data_us_en = IMDB_data_us_en[(IMDB_data_us_en['release_date'] < 2023)]
    # drop duplicates movie if any
    IMDB_data_us_en = IMDB_data_us_en.drop_duplicates(subset='IMDB_ID', keep='first')

    return IMDB_data_us_en


def clean_IMDB_character_dataset(dataframe, IMDB_ids):
    """
    Clean the IMDB character dataset: removes all jobs other than actresses and actors, keep only movies that are in
    IMDB_ids, add the gender in the dataset, drops job and category columns, converts the character name in a string
    :param dataframe: pandas dataframe: dataframe to clean, should contain columns 'nconst' (character IMDB ID),
    tconst (movie IMDB ID), characters, 'category', 'job' and 'ordering'
    :param IMDB_ids: list of string: IMDB movie ids to clean
    :return: a pandas dataframe
    """

    # only keep actors and actresses
    data = dataframe[(dataframe['category'] == 'actor') | (dataframe['category'] == 'actress')]

    # get only characters from movies used to complete the movie database
    characters_data = data[data['tconst'].isin(IMDB_ids)]

    # add gender of actors/actresses
    characters_data.loc[characters_data['category'] == 'actor', 'actor_gender'] = 'M'
    characters_data.loc[characters_data['category'] != 'actor', 'actor_gender'] = 'F'

    # drop useless columns
    characters_data = characters_data.drop(columns=['job', 'category'])

    # convert into a list
    characters_data['characters'] = characters_data['characters'].apply(
        lambda x: ast.literal_eval(x) if x != '\\N' else x)
    # Extract strings from lists
    characters_data['characters'] = characters_data['characters'].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

    return characters_data


def remove_duplicated_columns(dataframe, columns_to_remove, col_to_keep='_y', col_to_delete='_x'):
    """
    Merge columns that are duplicated when performing an outer merge (col_name_x and col_name_y) by keeping
    col_name + col_to_keep if both columns do not contain NaNs or if col_name + col_to_delete contains NaN
    and keeping col_name + col_to_delete otherwise.
    :param dataframe: panda dataframe
    :param columns_to_remove: list of string: name of the columns that were duplicated and need to be merged
    :param col_to_keep: string: '_x' or '_y', indicates which column will be kept in the dataframe and will be used to
    fill the NaNs in the other column
    :param col_to_delete: string: '_x' or '_y' indicates which column will be deleted from the dataframe
    :return: panda dataframe
    """
    for col_name in columns_to_remove:
        mask = dataframe[col_name + col_to_keep].isna()
        dataframe.loc[mask, col_name + col_to_keep] = dataframe.loc[mask, col_name + col_to_delete]
        dataframe = dataframe.drop(columns=col_name + col_to_delete)
        dataframe = dataframe.rename(columns={col_name + col_to_keep: col_name})

    return dataframe


def merge_datasets_characters(characters_data, actors_data, movie_data):
    """
    Completes the character data with information from the actor data (left join). Then merges the resulting dataset with
    information about the movie (left join with movie_data). Computes the age of the actors the year of the release
    and renames columns tconst, nconst, characters, primaryName and birthYear into IMDB_ID, actor_IMDB_ID, character_name,
    actor_name and actor_birthday respectively.
    :param characters_data: pandas dataframe: should at least contain columns 'nconst' (character IMDB ID),
    'tconst' (movie IMDB ID), 'characters' and 'ordering'
    :param actors_data: pandas dataframe: should at least contain columns 'nconst' (character IMDB ID),
    'primaryName' (actor name) and 'birthYear'
    :param movie_data: pandas dataframe: should at least contain columns: 'genre', 'plot_summary', 'IMDB_ID'
    :return: merged and formatted pandas dataframe
    """

    characters_data_merged = pd.merge(characters_data, actors_data, on='nconst', how='left').copy()

    # remove columns we do not need in the character dataset
    characters_movies_data = movie_data.drop(columns=['genre', 'plot_summary', 'freebase_ID', 'wikipedia_ID'])
    characters_data_merged = characters_data_merged.rename(columns={'tconst': 'IMDB_ID'})

    characters_data_final = pd.merge(characters_data_merged, characters_movies_data, on='IMDB_ID', how='left').copy()

    # change columns name
    characters_data_final = characters_data_final.rename(
        columns={'nconst': 'actor_IMDB_ID', 'characters': 'character_name',
                 'primaryName': 'actor_name', 'birthYear': 'actor_birthday'})

    # compute actor age the year of the release
    characters_data_final.loc[characters_data_final['actor_birthday'] == '\\N', 'actor_birthday'] = None
    characters_data_final['actor_birthday'] = (characters_data_final['actor_birthday']).astype(float)
    characters_data_final['actor_age'] = (
            characters_data_final['release_date'] - (characters_data_final['actor_birthday']).astype(float))

    # drop actor_birthday column
    characters_data_final = characters_data_final.drop(columns='actor_birthday')

    return characters_data_final.drop(columns='ordering')


def name_to_lowercase(dataframe, column_name):
    """
    Changes the content of column_name to lower cases
    :param dataframe: panda dataframe
    :param column_name: string: name of the column that needs to be put in lower case
    :return: panda dataframe with 'column_name' content in lower case
    """
    return [c.lower() if pd.notna(c) else c for c in dataframe[column_name]]


def get_popularity_index(data, threshold=10):
    """
    Creates a popularity index for movies in the dataset. It is done by dividing the box-office revenue of each movie
    for a year by the average of the 3 highest grossing movies the same year. This operation is only done on years that
    have more than 10% of box-office values.
    :param data: panda dataframe: data on used to compute the popularity index. Must contain the columns: 'release_date',
    'box_office_revenue' and 'IMDB_ID'
    :param threshold: int: minimal percentage of box-offices needed in one year in order to compute the popularity index
    for this year.
    :return:
    """

    # select years on which to compute popularity index
    data_per_year = data.groupby('release_date').count()
    fraction_box_office_per_year = data_per_year['box_office_revenue'] * 100 / data_per_year['IMDB_ID']
    box_office_years = fraction_box_office_per_year[fraction_box_office_per_year > threshold].index

    # get average of 3 biggest box-office for each year
    top_three_per_year = data.groupby('release_date')['box_office_revenue'].nlargest(3)
    max_per_year = top_three_per_year.groupby('release_date').mean()

    # compute popularity index
    pop_index = []
    for i in range(data.shape[0]):
        year = data['release_date'][i]
        if year in box_office_years:
            pop_index.append(data['box_office_revenue'][i] / max_per_year[year])
        else:
            pop_index.append(None)

    return pop_index


######################################### MAIN CHARACTER ANALYSIS ######################################################

def select_elligible_movies_for_main_char_analysis(movie_metadata, character_metadata):
    """
    Selects movies that are elligible for the main character analysis
    :param movie_metadata: panda dataframe
    :param character_metadata: panda dataframe
    :return elligible_movies: panda dataframe
    """
    # Removing movies without summaries
    elligible_movies = movie_metadata[movie_metadata['plot_summary'].notna()]

    # Removing movies with no listed characters
    movies_with_no_listed_char_wiki_ID = []
    movies_with_no_listed_char_imdb_ID = []
    for wiki_id, imdb_id in zip(elligible_movies['wikipedia_ID'], elligible_movies['IMDB_ID']):

        if not pd.isnull(wiki_id) and pd.isnull(imdb_id):
            if character_metadata.loc[character_metadata['wikipedia_ID'] == wiki_id].empty:
                movies_with_no_listed_char_wiki_ID.append(wiki_id)
            continue

        if pd.isnull(wiki_id) and not pd.isnull(imdb_id):
            if character_metadata.loc[character_metadata['IMDB_ID'] == imdb_id].empty:
                movies_with_no_listed_char_imdb_ID.append(imdb_id)
            continue

        if not pd.isnull(wiki_id) and not pd.isnull(imdb_id):
            no_char_on_wikipedia = character_metadata.loc[character_metadata['wikipedia_ID'] == wiki_id].empty
            no_char_on_imdb = character_metadata.loc[character_metadata['IMDB_ID'] == imdb_id].empty

            if no_char_on_wikipedia and no_char_on_imdb:
                movies_with_no_listed_char_wiki_ID.append(wiki_id)
                movies_with_no_listed_char_imdb_ID.append(imdb_id)

    elligible_movies = elligible_movies[~elligible_movies['wikipedia_ID'].isin(movies_with_no_listed_char_wiki_ID)]
    elligible_movies = elligible_movies[~elligible_movies['IMDB_ID'].isin(movies_with_no_listed_char_imdb_ID)]

    id_types = ['wikipedia_ID', 'IMDB_ID']
    for id_type in id_types:
        # Removing movies with characters having name listed as nan
        movies_with_nan_characters_ID = pd.Series(
            character_metadata[character_metadata['character_name'].isna()][id_type].unique())
        movies_with_nan_characters_ID = movies_with_nan_characters_ID.dropna().tolist()
        elligible_movies = elligible_movies[~elligible_movies[id_type].isin(movies_with_nan_characters_ID)]

        # Removing movies with characters having gender listed as nan
        movies_with_nan_genders_ID = pd.Series(
            character_metadata[character_metadata['actor_gender'].isna()][id_type].unique())
        movies_with_nan_genders_ID = movies_with_nan_genders_ID.dropna().tolist()
        elligible_movies = elligible_movies[~elligible_movies[id_type].isin(movies_with_nan_genders_ID)]

    return elligible_movies


def extract_main_characters(summary: str, nb_sentences=5):
    """
    Extracts the most mentioned characters in a movie summary
    :param summary: string
    :param nb_sentences: integer
    :return main_characters: list of strings
    """
    tagger = SequenceTagger.load('ner')

    # Extracting and tagging first nb_sentences from summary
    sentences = sent_tokenize(summary)
    tagged_sentences = [Sentence(sent) for sent in sentences[:nb_sentences]]
    tagger.predict(tagged_sentences)

    # Extracting all names from the tagged sentences
    entities = [entity for sent in tagged_sentences for entity in sent.to_dict(tag_type='ner')['entities']]
    names = [entity['text'] for entity in entities if entity['labels'][0]['value'] == 'PER']

    # Removing punctuation
    names = [name.translate(str.maketrans('', '', string.punctuation)) for name in names]

    names_numbered = Counter(names).most_common()

    characters = defaultdict(int)

    for name, count in names_numbered:
        found = False
        standardized_name = name.lower()

        # Adding up number of counts if over 50% match
        for existing_name in characters:
            if fuzz.ratio(standardized_name, existing_name) > 50:
                characters[existing_name] += count
                found = True
                break

        # Adding name to character list if unique
        if not found:
            characters[standardized_name] += count

    # Converting from dictionary to ordered list
    ordered_characters = sorted(characters.items(), key=lambda x: x[1], reverse=True)

    ordered_characters = Counter(characters).most_common()
    main_characters = [name for name, count in ordered_characters[:3]]

    return main_characters


def find_main_characters_genders(movie_row, characters_df):
    """
    Finds the gender of each character by matching character names to character metadata
    :param movie_row: pandas dataframe (one row)
    :param characters_df: pandas dataframe
    :return genders: list of strings
    """
    IMDB_ID_character_list = characters_df.loc[characters_df['IMDB_ID'] == movie_row['IMDB_ID']]
    wikipedia_ID_character_list = characters_df.loc[characters_df['wikipedia_ID'] == movie_row['wikipedia_ID']]
    selected_character_metadata = pd.concat([IMDB_ID_character_list, wikipedia_ID_character_list], ignore_index=True)

    genders = []
    for name in movie_row['main characters']:
        confidence = 0
        if selected_character_metadata['character_name'].any():
            closest_character, confidence, score = process.extractOne(name,
                                                                      selected_character_metadata['character_name'])

        if confidence > 50:
            gender = selected_character_metadata.loc[
                selected_character_metadata['character_name'] == closest_character, 'actor_gender'].values[0]
            genders.append(gender)

    return genders


def calculate_gender_ratio(genders: list):
    """
    Calculates the female to male gender ratio in a list of strings
    containing ['M'] and ['F']
    :param genders: list
    :return ratio: float
    """

    if len(genders) == 0:
        return pd.NA

    nb_females = sum(1 for gender in genders if gender == 'F')

    return nb_females / len(genders)


def create_data_plot_gender_ratio(movies):
    """
    Create plotly go.Scatter objects that can be used to plot the average main character gender ratio of movies per
    decade and the 95% confidence interval
    :param movies: pandas dataframe: contains information on the movies we wish to plot.
    Must contain the columns "main_character_genders" and "decade".
    :return: plot_average, plot_lower_bound, plot_upper_bound
    """

    # compute main character gender ratio average
    movies['gender_ratio'] = movies['main character genders'].apply(calculate_gender_ratio)
    movies = movies.dropna(subset=['gender_ratio'])
    ratio_data = movies[["decade", "gender_ratio"]]
    # average per decade
    ratio_means = ratio_data.groupby('decade').mean().reset_index()
    ratio_means['gender_ratio'] = pd.to_numeric(ratio_means['gender_ratio'], errors='coerce')
    # compute 95% confidence interval
    ratio_ci = ratio_data.groupby('decade')['gender_ratio']. \
        apply(stats.sem).mul(stats.t.ppf(0.975, ratio_data.groupby('decade').size() - 1)).reset_index(name='conf')

    # Create average line
    plot_average = go.Scatter(
        name='Decade Average',
        x=ratio_means['decade'],
        y=ratio_means['gender_ratio'],
        mode='lines',
        marker_color=px.colors.qualitative.Plotly[0]
    )
    # create 95% CI lines
    plot_lower_bound = go.Scatter(
        x=ratio_means['decade'],
        y=ratio_means['gender_ratio'] - ratio_ci['conf'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=True,
        name='95% Confidence Interval',
        fillcolor='rgba(173, 216, 230, 0.3)',
        fill='tonexty')
    plot_upper_bound = go.Scatter(
        x=ratio_means['decade'],
        y=ratio_means['gender_ratio'] + ratio_ci['conf'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False,
        name='95% Confidence Interval',
        fillcolor='rgba(173, 216, 230, 0.3)',
        fill='tonexty')

    return plot_average, plot_lower_bound, plot_upper_bound


def plot_gender_ratio_plotly(movies, save_fig=True, show_png=True, folder='', file_name="main_char_ratio.html"):
    """
    Creates an interactive plot of the average main character gender ratio of movies per decade and the 95% confidence
    interval using the plotly library
    :param movies: pandas dataframe: contains information on the movies we wish to plot.
    Must contain the columns "gender_ratio", "main_character_genders" and "decade".
    :param save_fig: bool: if True, will save the interactive plot in a .html file, default = True
    :param show_png: bool: if True will display the plot as a png instead of as an interactive plot, default = True
    :param folder: string: name of the folder to save the figure in, default=''
    :param file_name: string: name of the file to save the figure in, default='main_char_ratio.html'
    """
    plot_average, plot_lower_bound, plot_upper_bound = create_data_plot_gender_ratio(movies)

    # Create a Plotly figure
    fig = go.Figure([plot_average, plot_lower_bound, plot_upper_bound])

    fig.update_layout(
        xaxis_title='Movie release decade',
        yaxis_title='Female/Male ratio',
        title='Gender ratio in main characters over time',
        hovermode="x")

    if save_fig:
        fig.write_html(folder + file_name, auto_open=True)
    if show_png:
        fig.show('png')
    else:
        fig.show()


def subplot_gender_ratio(movies, idx, subplot):
    """
    Creates a subplot of the average main character gender ratio of movies per decade and the 95% confidence
    interval using the plotly library
    :param movies: pandas dataframe: contains information on the movies we wish to plot.
    Must contain the columns "gender_ratio", "main_character_genders" and "decade".
    :param idx: int: Where to plot the subplot in a 2x3 grid. The location is (idx%2+1, idx%3+1)
    :param subplot: plotly.subplots object: plot to which the subplot will be added
    """

    plot_average, plot_lower_bound, plot_upper_bound = create_data_plot_gender_ratio(movies)
    plots = [plot_average, plot_lower_bound, plot_upper_bound]

    # show legend only one
    if idx != 1:
        plot_lower_bound.update(dict(showlegend=False))
        plot_average.update(dict(showlegend=False))

    for plot in plots:
        subplot.append_trace(plot, row=idx % 2 + 1, col=idx % 3 + 1)
    subplot.update_yaxes(range=[0, 1], row=idx % 2 + 1, col=idx % 3 + 1)


def random_movies_per_year(group):
    """
    Selects 20 movies at random
    :param group: pandas dataframe
    """
    return group.sample(n=min(20, len(group)), random_state=10)


################################## GENRES PREPROCESSING ################################################################

def no_genres_in_list(genres, df):
    """
    Filters out movies from a dataframe that do not belong to any of the specified genres.
    :param genres: list: List of genres to be considered for filtering.
    :param df: pandas DataFrame: Dataframe containing movie information. 
               It should have columns 'name' for movie names and 'genre' for movie genres.
    :return: list: List of movie IDs (from the 'name' column) that belong to at least one of the specified genres.
    """
    ID_no_accepted_genre = []
    for index, row in df.iterrows():
        ID = row['name']
        genre = row['genre']
        if all(element in genres for element in genre) and genre != []:
            ID_no_accepted_genre.append(ID)
    print("Number of movies : ", len(ID_no_accepted_genre))
    return ID_no_accepted_genre


################################## PERSONAS PREPROCESSING ##############################################################

def extract_words(df, id_col, char_name_col, to_extract):
    """
    Extracts verbs, adjectives, and nouns from textual data in a dataframe.
    :param df: pandas dataframe: df that contains the textual data to analyze in a column.
    :param id_col: str: column name containing identifier for each text entry (here each movie has an ID and is associated with a summary).
    :param char_name_col: str: column name containing character names.
    :param to_extract: str: column name containing text to be analyzed.
    :return: Lists of verbs, adjectives, nouns, and named entity chunks for each movie.
    """
    tokens = pd.Series()
    tagged_tokens = []
    chunks_array = []
    verbs_list = []
    adjs_list = []
    nouns_list = []
    stop_words = set(stopwords.words('english'))
    # Adding tags for verbs, adjectives, and nouns. These are Penn Treebank tags, the full list can be found here:
    #https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    adj_tags = ['JJ', 'JJR', 'JJS']
    noun_tags = ['NN',
                 'NNS']  # We do not take NNPs and NNP because they are names and we do not want to include names in our analysis

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Movies"):
        verbs = []
        adjs = []
        nouns = []
        text = row[to_extract]
        associated_w_text = row[char_name_col]
        movie_id = row[id_col]

        if type(text) == str:  # To only keep movies with a summary (ignoring NaN)
            token = [word for word in nltk.word_tokenize(text) if word.lower() not in stop_words]  # Removing stopwords
            tokens[associated_w_text] = token
            tagged_tokens.append((movie_id, associated_w_text, nltk.pos_tag(token)))

    for movie_id, associated_w_text, tagged_token in tqdm(tagged_tokens, desc="Processing Tokens", leave=False):
        chunks_array.append((movie_id, associated_w_text, nltk.ne_chunk(tagged_token)))

        verbs = []
        adjs = []
        nouns = []

        # Categorize
        for word, pos_tag in tagged_token:
            if pos_tag in verb_tags:
                verbs.append(word)
            elif pos_tag in adj_tags:
                adjs.append(word)
            elif pos_tag in noun_tags:
                nouns.append(word)

        verbs_list.append((movie_id, associated_w_text, verbs))
        adjs_list.append((movie_id, associated_w_text, adjs))
        nouns_list.append((movie_id, associated_w_text, nouns))

    # Returns lists of all verbs, adjectives, and nouns for each movie and raw chunks for each movie
    return verbs_list, adjs_list, nouns_list, chunks_array


def find_characters_genders_for_all_movies(movies_df, characters_df): 
    """
    Finds the gender of characters for all movies in a dataframe
    :param movies_df: pandas dataframe with movie information
    :param characters_df: pandas dataframe with character information
    :return result_df: pandas dataframe with IMDb ID, summary, and characters' genders for all movies
    """
    all_results = []

    for _, movie_row in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Processing Movies"):
        # Check if the 'plot_summary' is a non-NaN value
        if pd.notna(movie_row['plot_summary']):
            IMDB_ID_character_list = characters_df.loc[characters_df['IMDB_ID'] == movie_row['IMDB_ID']]

            genders = []
            characters = []
            summary_words = [word for word in word_tokenize(movie_row['plot_summary'].lower()) if
                             word.isalnum()]  # Tokenize and exclude non-alphanumeric characters

            # Set to keep track of already matched characters for the current movie and summary
            matched_characters = set()

            for word in summary_words:
                closest_character, confidence, score = process.extractOne(word,
                                                                          IMDB_ID_character_list['character_name'])

                if confidence > 50 and closest_character:  # Check confidence and non-empty character
                    # Check if the character has already been matched for the current movie and summary
                    if closest_character not in matched_characters:
                        gender = IMDB_ID_character_list.loc[
                            IMDB_ID_character_list['character_name'] == closest_character, 'actor_gender'].values
                        if len(gender) > 0:
                            characters.append(closest_character)
                            genders.append(gender[0])

                        # Add the matched character to the set for the current movie and summary
                        matched_characters.add(closest_character)

            result_df = pd.DataFrame({
                'IMDB_ID': [movie_row['IMDB_ID']] * len(characters),
                'plot_summary': [movie_row['plot_summary']] * len(characters),
                'character_name': characters,
                'gender': genders
            })

            all_results.append(result_df)

    result_df = pd.concat(all_results, ignore_index=True)
    return result_df


def extract_context_strings(result_df):
    """
    Extracts context strings for each character in the DataFrame
    :param result_df: pandas dataframe with IMDb ID, summary, characters, and genders
    :return result_df_with_context: pandas dataframe with IMDb ID, summary, characters, genders, and context strings
    """
    result_df_with_context = result_df.copy()
    stop_words = set(stopwords.words('english'))
    # remove the stopwords to not extract them
    result_df_with_context['plot_summary'] = result_df_with_context['plot_summary'].apply(
        lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
    result_df_with_context['associated_words'] = ""

    for _, row in tqdm(result_df_with_context.iterrows(), total=len(result_df_with_context),
                       desc="Extracting Context Strings"):
        character = row['character_name']
        summary_words = [word for word in word_tokenize(row['plot_summary']) if word.isalnum()]

        # Use fuzzy matching to find occurrences of the character name in the summary (non exact matches will also be found)
        occurrences = [i for i, word in enumerate(summary_words) if process.extractOne(word, [character])[1] > 50]

        # Extract context strings for each occurrence, 3 words before and 3 words after
        context_strings = []
        for occurrence in occurrences:
            start_index = max(0, occurrence - 3)
            end_index = min(len(summary_words), occurrence + 4)

            # Exclude the matched word from the context string
            matched_word = summary_words[occurrence]
            context_string = " ".join(summary_words[start_index:end_index])
            context_string = context_string.replace(matched_word, "")  # Remove the matched word (character name)
            context_strings.append(context_string.strip())  # Strip leading and trailing spaces

        # Concatenate context strings if there are multiple occurrences for the same character name
        if len(context_strings) > 1:
            context_string = " ".join(context_strings)
        elif len(context_strings) == 1:
            context_string = context_strings[0]
        else:
            context_string = ""

        result_df_with_context.at[_, 'associated_words'] = context_string

    return result_df_with_context


def create_gender_dictionaries(df):
    """
    Constructs dictionaries categorizing words by gender, decade, and category (Verbs, Adjectives and Nouns).
    :param df: pandas dataframe: contains data on characters and their associated words.
    :return: 2 dictionaries (one for male and one for female) containing word lists categorized by decade and category.
    """
    # Initialize dictionaries for male and female characters
    male_dict = {}
    female_dict = {}

    # Iterate through the dataframe and populate dictionaries
    for index, row in df.iterrows():
        gender = row['actor_gender']
        decade = row['decade'] # we want the create a dictionnary per decade to later analyze

        # Check if the gender is male and handle empty lists (characters don't necessarily have words of the 3 cat. associated to them (they may have verbs but no adjectives or nouns for example)
        if gender == 'M':
            if decade not in male_dict:
                male_dict[decade] = {'Verbs': [], 'Adjectives': [], 'Nouns': []} # create a dict per decade

            male_dict[decade]['Verbs'].extend(row['Verbs']) if row['Verbs'] else None
            male_dict[decade]['Adjectives'].extend(row['Adjectives']) if row['Adjectives'] else None
            male_dict[decade]['Nouns'].extend(row['Nouns']) if row['Nouns'] else None

        # Check if the gender is female and handle empty lists the same way as for men
        elif gender == 'F':
            if decade not in female_dict:
                female_dict[decade] = {'Verbs': [], 'Adjectives': [], 'Nouns': []}

            female_dict[decade]['Verbs'].extend(row['Verbs']) if row['Verbs'] else None
            female_dict[decade]['Adjectives'].extend(row['Adjectives']) if row['Adjectives'] else None
            female_dict[decade]['Nouns'].extend(row['Nouns']) if row['Nouns'] else None

    return male_dict, female_dict

def calculate_word_frequencies(decades_dict):
    """
    Computes the frequency of words across different categories and decades.
    :param decades_dict: Dictionary with words by decade and category associated to a specific gender
    :return: Dictionary: word frequencies categorized by decade and category.
    """
    # Initialize a dictionary to store frequencies for each decade
    all_frequencies = {}

    # Iterate through the decades and calculate word frequencies
    for decade, dictionary in decades_dict.items():
        frequencies = {'Verbs': {}, 'Adjectives': {}, 'Nouns': {}}

        for category, words in dictionary.items():
            total_words = len(words)
            word_counter = Counter(words)
            frequencies[category] = {word: count / total_words for word, count in word_counter.items()}

        all_frequencies[decade] = frequencies

    return all_frequencies

def stem_words_with_progress(word_list):
    """
    Stems words in a list using the Porter stemming algorithm with a progress bar to follow the run.
    :param word_list: List[str]: list of words to be stemmed.
    :return: List[str]: stemmed words.
    """
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in word_list:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def subtract_frequencies(frequencies_1, frequencies_2):
    """
    Computes the normalized difference in word frequencies between two frequency dictionaries to obtain normalized relative frequencies.
    :param frequencies_1: Dictionary: word frequencies categorized by decades and category
    :param frequencies_2: Dictionary: word frequencies categorized by decades and category
    :return: Dictionary: subtracted frequencies and unique words for each input dictionary.
    """
    subtracted_frequencies = {}
    words_only_in_freq1 = {}
    words_only_in_freq2 = {}

    for decade in frequencies_1.keys():
        subtracted_frequencies[decade] = {'Verbs': {}, 'Adjectives': {}, 'Nouns': {}}
        words_only_in_freq1[decade] = {'Verbs': {}, 'Adjectives': {}, 'Nouns': {}}
        words_only_in_freq2[decade] = {'Verbs': {}, 'Adjectives': {}, 'Nouns': {}}

        for category in frequencies_1[decade].keys():
            common_words = set(frequencies_1[decade][category].keys()) & set(frequencies_2[decade][category].keys())
            
            # find the common words between the 2 dictionnaries and substract their frequencies
            for word in common_words:
                freq_1 = frequencies_1[decade][category].get(word, 0)
                freq_2 = frequencies_2[decade][category].get(word, 0)
                subtracted_frequencies[decade][category][word] = freq_1 - freq_2

            # keep track of the words that are only in one of the two dictionnaries
            remaining_words_1 = set(frequencies_1[decade][category].keys()) - common_words
            remaining_words_2 = set(frequencies_2[decade][category].keys()) - common_words

            for word in remaining_words_1:
                subtracted_frequencies[decade][category][word] = frequencies_1[decade][category][word]
                words_only_in_freq1[decade][category][word] = frequencies_1[decade][category][word]

            for word in remaining_words_2:
                subtracted_frequencies[decade][category][word] = -frequencies_2[decade][category][word]
                words_only_in_freq2[decade][category][word] = frequencies_2[decade][category][word]

    return subtracted_frequencies, words_only_in_freq1, words_only_in_freq2


def plot_rel_freq_per_decade(relative_frequencies, categories):
    """
    Plots relative word frequencies per decade for given categories.
    :param relative_frequencies: Dictionary: relative word frequencies categorized by decade and category.
    :param categories: List[str]: list of word categories (e.g., Verbs, Adjectives, Nouns).
    """
    # Create subplots for each category
    decades = relative_frequencies.keys()
    num_categories = len(categories)

    fig, axs = plt.subplots(len(decades), num_categories, figsize=(5 * num_categories, 3 * len(decades)))

    # Iterate through each decade
    for i, decade in enumerate(decades):
        for j, category in enumerate(categories):
            # Get the top words and their frequencies
            top_words_positive = [word for _, word in sorted(
                zip(relative_frequencies[decade][category].values(), relative_frequencies[decade][category].keys()),
                key=lambda x: x[0],
                reverse=True
            )[:5]]

            top_words_negative = [word for _, word in sorted(
                zip(relative_frequencies[decade][category].values(), relative_frequencies[decade][category].keys()),
                key=lambda x: x[0]
            )[:5]]

            top_words = top_words_positive + top_words_negative
            top_frequencies = [relative_frequencies[decade][category][word] for word in top_words]

            # Plot the bar chart for each category
            bars = axs[i, j].bar(top_words, top_frequencies,
                                 color=['skyblue' if freq >= 0 else 'pink' for freq in top_frequencies])
            axs[i, j].set_title(f'{category} {decade}')
            axs[i, j].set_ylabel('Relative frequencies: Male - Female')

            # Rotate x-axis tick labels to the words can be easily read
            axs[i, j].set_xticklabels(top_words, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def plot_top_words_per_decade(frequencies, categories, col="blue"):
    """
    Plots top words per decade for given categories in a dictionnary.
    :param frequencies: Dictionary: word frequencies categorized by decade and category.
    :param categories: List[str]: list of word categories (e.g., Verbs, Adjectives, Nouns).
    :param col: str: color for the plot bars (default is blue).
    """
    # Create subplots for each category
    decades = frequencies.keys()
    num_categories = len(categories)

    fig, axs = plt.subplots(len(decades), num_categories, figsize=(5 * num_categories, 3 * len(decades)))

    # Iterate through each decade
    for i, decade in enumerate(decades):
        for j, category in enumerate(categories):
            # Get the top words and their frequencies
            top_words = [word for _, word in sorted(
                zip(frequencies[decade][category].values(), frequencies[decade][category].keys()),
                key=lambda x: x[0],
                reverse=True
            )[:5]]

            top_frequencies = [frequencies[decade][category][word] for word in top_words]

            # Plot the bar chart for each category
            bars = axs[i, j].bar(top_words, top_frequencies, color=col)
            axs[i, j].set_title(f'{category} {decade}')
            axs[i, j].set_ylabel('Absolute Frequency Difference')

            # Rotate x-axis tick labels to the words can be easily read
            axs[i, j].set_xticklabels(top_words, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def subset_df(df, genre):
    """
    Subsets a dataframe based on a given genre to only keep the rows that are associated to the genre.
    :param df: pandas dataframe: contains information on movies.
    :param genre: str: genre to subset by.
    :return: pandas dataframe: subset of the original dataframe filtered by genre.
    """
    df_subset_genre = df[df['genre'].str.len() > 0]
    df_subset_genre = df_subset_genre[df_subset_genre['genre'].apply(lambda x: genre in x)]
    return df_subset_genre


def plot_by_genre(genre, df):
    """
    Plots word frequencies per decade and category (Verbs, Adjectives and Nouns) for a specific genre.
    :param genre: str: genre to analyze and plot.
    :param df: pandas dataframe: contains information on movies and characters.
    """
    # use previously defined functions to create plots with associated words for each gender in genres we are interested in
    df_subset_genre = subset_df(df, genre)
    male_dict_genre, female_dict_genre = create_gender_dictionaries(df_subset_genre)
    male_frequencies_per_decade_genre = calculate_word_frequencies(male_dict_genre)
    female_frequencies_per_decade_genre = calculate_word_frequencies(female_dict_genre)
    relative_frequencies_genre, unique_male_genre, unique_female_genre = subtract_frequencies(
        male_frequencies_per_decade_genre, female_frequencies_per_decade_genre)
    plot_rel_freq_per_decade(relative_frequencies_genre, ['Verbs', 'Adjectives', 'Nouns'])


######################################## CLUSTERING OF STEREOTYPICAL MOVIES ############################################

def compute_difference_mean_ages(data):
    """
    Computes the difference of the mean age of female and the mean age of male for each movie in "data"
    :param data: panda dataframe: contains data on characters, must contain at least the columns: actor_gender,
    IMDB_ID and actor_age
    :return: a panda dataframe containing for each movie its IMDB ID, the mean age of actresses, the mean age of actors
    and the difference of these means
    """

    # separate male and female characters
    female = data[data['actor_gender'] == 'F'].dropna(subset='actor_age')
    male = data[data['actor_gender'] == 'M'].dropna(subset='actor_age')
    female_per_movie = female.groupby('IMDB_ID')
    male_per_movie = male.groupby('IMDB_ID')

    # compute the mean ages for both genders
    mean_age_F = female_per_movie['actor_age'].mean()
    mean_age_M = male_per_movie['actor_age'].mean()
    df_mean_age_F = pd.DataFrame({'IMDB_ID': mean_age_F.index, 'mean_age_female': mean_age_F.values})
    df_mean_age_M = pd.DataFrame({'IMDB_ID': mean_age_M.index, 'mean_age_male': mean_age_M.values})

    # create the final dataframe
    mean_ages = pd.merge(df_mean_age_F, df_mean_age_M, on='IMDB_ID', how='outer')
    mean_ages['difference_mean_ages'] = mean_ages['mean_age_female'] - mean_ages['mean_age_male']

    # removes incoherent data if present (negative ages)
    mean_ages = mean_ages[(mean_ages['mean_age_female'] > 0) & (mean_ages['mean_age_male'] > 0)]

    return mean_ages


def plot_sse(features_X, start=2, end=11):
    """
    Plots the sum of the square error of the k-means clustering algorithm for values of k comprised in [start, end]
    :param features_X: panda dataframe: features on which to perform k means clustering
    :param start: int: start value of k to try
    :param end: int: end value of k to try
    """
    sse = []
    for k in range(start, end):
        kmeans = KMeans(n_clusters=k, random_state=10, n_init=10).fit(features_X)
        sse.append({"k": k, "sse": kmeans.inertia_})

    sse = pd.DataFrame(sse)
    # Plot the data
    plt.plot(sse.k, sse.sse)
    plt.xlabel("K")
    plt.ylabel("Sum of Squared Errors")
    plt.title('Sum of squared errors of k-means clustering for different values of k')
    plt.show()


def plot_kmeans_2d(data, labels, columns):
    """
    Visualise a clustering in 2D by plotting the two columns "columns" of "data" and color coding the dots with the labels.
    :param data: panda dataframe: the data that was clustered, must contain the columns "columns"
    :param labels: numpy array of size data.shape[0]: contains the cluster number of each data point in data
    :param columns: list of string, length = 2: contains the name of the two columns of data that we want to visualise.
    The first element of the list will be plotted on the x-axis and the second on the y-axis.
    """
    plt.scatter(data[columns[0]], data[columns[1]], c=labels, alpha=0.6, s=20)
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.title('K-means clustering')
    plt.show()


def plot_kmeans_3d(data, labels, columns):
    """
    Visualise a clustering in 3D by plotting the 3 columns "columns" of "data" and color coding the dots with the labels.
    :param data: panda dataframe: the data that was clustered, must contain the columns "columns"
    :param labels: numpy array of size data.shape[0]: contains the cluster number of each data point in data
    :param columns: list of string, length = 3: contains the name of the three columns of data that we want to visualise.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(data[columns[0]])
    y = np.array(data[columns[1]])
    z = np.array(data[columns[2]])
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_zlabel(columns[2])

    ax.scatter(x, y, z, marker="s", c=labels)

    plt.show()


def bootstrap_ci_stereotypical_movies(data, num_iterations=1000, alpha=0.05):
    """
    Computes the confidence interval of the proportion of stereotypical and not-stereotypical movies per decade using
    bootstrap.
    :param data: panda dataframe: movie data used to compute the CIs, must contain the columns IMDB_ID, decade
    and cluster_index. A movie with cluster index of 0 is considered as non-stereotypical. A movie with cluster index
    of 1 is considered as stereotypical.
    :param num_iterations: int: the number of re-sampling done for the bootstrap, default = 1000.
    :param alpha: float: the level of significance of the confidence interval, default = 0.05.
    :return: CI lower bound for stereotypical movies, CI upper bound for stereotypical movies,
    CI lower bound for non-stereotypical movies, CI upper bound for non-stereotypical movies
    """

    fraction_stereo_lower = []
    fraction_stereo_upper = []
    fraction_not_stereo_lower = []
    fraction_not_stereo_upper = []

    for decade, decade_data in data.groupby('decade'):
        n = len(decade_data)
        fraction_stereo = []
        fraction_not_stereo = []

        for i in range(num_iterations):
            bootstrap_sample_indices = np.random.choice(n, size=n, replace=True)
            bootstrap_sample = decade_data.iloc[bootstrap_sample_indices]

            not_stereotypical_ids = bootstrap_sample[(bootstrap_sample['cluster_index'] == 0)]['IMDB_ID']
            stereotypical_ids = bootstrap_sample[(bootstrap_sample['cluster_index'] == 1)]['IMDB_ID']

            fraction_stereo.append(len(stereotypical_ids) / n)
            fraction_not_stereo.append(len(not_stereotypical_ids) / n)

        fraction_not_stereo = np.array(fraction_not_stereo)
        fraction_stereo = np.array(fraction_stereo)

        lower_bound_stereo = np.percentile(fraction_stereo, (alpha / 2) * 100)
        upper_bound_stereo = np.percentile(fraction_stereo, (1 - alpha / 2) * 100)
        lower_bound_not_stereo = np.percentile(fraction_not_stereo, (alpha / 2) * 100)
        upper_bound_not_stereo = np.percentile(fraction_not_stereo, (1 - alpha / 2) * 100)

        fraction_stereo_lower.append(lower_bound_stereo)
        fraction_stereo_upper.append(upper_bound_stereo)
        fraction_not_stereo_lower.append(lower_bound_not_stereo)
        fraction_not_stereo_upper.append(upper_bound_not_stereo)

    return (
        np.array(fraction_stereo_lower),
        np.array(fraction_stereo_upper),
        np.array(fraction_not_stereo_lower),
        np.array(fraction_not_stereo_upper)
    )


########################## PLOT PROPORTIONS AND AGES ###################################################################
def create_plot_male_female(character_data):
    """
    Create plotly go.Bar objects that can be used to plot the average female and male proportions in movies per decade.
    :param character_data: pandas dataframe: contains information on the characters we wish to plot.
    Must contain the columns "actor_gender" and "decade".
    :return: plot_male, plot_female
    """
    # compute proportions per decade
    character_data_grouped_by_decade = character_data.groupby(character_data['decade'])
    proportion_female = character_data_grouped_by_decade['actor_gender'].apply(lambda x: (x == 'F').sum() / len(x))
    proportion_male = 1 - proportion_female

    # Create a DataFrame
    data_for_plotting = pd.DataFrame({
        'Male': proportion_male,
        'Female': proportion_female})

    # Create traces for male and female bars
    plot_male = go.Bar(x=data_for_plotting.index, y=data_for_plotting['Male'], name='Male actors',
                       marker=dict(color='mediumturquoise'))
    plot_female = go.Bar(x=data_for_plotting.index, y=data_for_plotting['Female'], name='Female actors',
                         marker=dict(color='hotpink'))

    return plot_male, plot_female


def subplot_proportion_men_women_per_decade(character_data, idx, subplot):
    """
    Creates an interactive subplot of the average female and male proportions in movies per decade using the plotly library
    :param character_data: pandas dataframe: contains information on the characters we wish to plot.
    Must contain the columns "actor_gender" and "decade".
    :param idx: int: Where to plot the subplot in a 2x3 grid. The location is (idx%2+1, idx%3+1).
    :param subplot: plotly.subplots object: plot to which the subplot will be added
    """
    plot_male, plot_female = create_plot_male_female(character_data)

    # show legend only once
    if idx != 1:
        plot_male.update(dict(showlegend=False))
        plot_female.update(dict(showlegend=False))

    subplot.append_trace(plot_female, row=idx % 2 + 1, col=idx % 3 + 1)
    subplot.append_trace(plot_male, row=idx % 2 + 1, col=idx % 3 + 1)


def plot_proportion_men_women_per_decade(character_data, save_fig=True, show_fig_png=True, folder='',
                                         file_name="Proportion of Men vs Women in Movies per Decade.html"):
    """
    Displays an interactive graph of the average female and male proportions in movies per decade using the plotly library
    :param character_data: pandas dataframe: pandas dataframe: contains information on the characters we wish to plot.
    Must contain the columns "actor_gender" and "decade".
    :param save_fig: bool: if true, the interactive plot will be saved in an HTML file, default = True
    :param show_fig_png: bool: if true, the fig is displayed as a PNG (used for github display), else the figure is displayed
    as an interactive graph, default = True
    :param folder: string: name of the folder to save the figure in, default=''
    :param file_name: string: name of the file to save the figure in,
    default='Proportion of Men vs Women in Movies per Decade.html'
    """

    plot_male, plot_female = create_plot_male_female(character_data)

    # Create layout
    layout = go.Layout(
        title='Proportion of Men vs Women in Movies per Decade',
        xaxis=dict(title='Decade'),
        yaxis=dict(title='Proportion of total actors'),
        barmode='stack')

    # Create figure
    fig = go.Figure(data=[plot_female, plot_male], layout=layout)

    if save_fig:
        fig.write_html(folder + file_name, auto_open=True)
    if show_fig_png:
        fig.show("png")
    else:
        fig.show()


def create_data_proportion_percentages_plot(character_data):
    """
    Computes the proportion of movies across decades that have less that 25% of women in them, between 25-50%,
    between 50-75% and more than 75%.
    :param character_data: pandas dataframe: contains information on the characters we wish to plot. Must contain
    the columns "decade", as well as a "name" for the movies and "actor gender".
    :return: [below_25_per_decade, quarter_50_per_decade, half_75_per_decade, above_75_per_decade] and unique_decades
    """

    # group by decade and by movie
    characters_grouped_by_decade_and_movie = character_data.groupby(['decade', 'name'])
    # Calculate the proportion of men and women per year per movie
    proportion_female_per_movie = \
        characters_grouped_by_decade_and_movie['actor_gender'].apply(lambda x: (x == 'F').sum() / len(x))

    # Convert the Series to a DataFrame for easier data handling in later analysis
    female_proportions = proportion_female_per_movie.reset_index()
    female_proportions.columns = ['decade', 'name', 'proportion_female']

    # Lists to store in which category each movie belongs to per year
    below_25_per_decade = []
    quarter_50_per_decade = []
    half_75_per_decade = []
    above_75_per_decade = []

    # Iterate through each year in the index of female_proportions
    for decade, proportions in female_proportions.groupby('decade'):
        # Count movies in different categories for the current decade
        below_25 = len(proportions[proportions['proportion_female'] <= 0.25])
        quarter_50 = len(proportions[(proportions['proportion_female'] > 0.25) &
                                     (proportions['proportion_female'] <= 0.5)])
        half_75 = len(proportions[(proportions['proportion_female'] > 0.5) &
                                  (proportions['proportion_female'] <= 0.75)])
        above_75 = len(proportions[proportions['proportion_female'] > 0.75])

        # Calculate proportions for each category
        total_movies = len(proportions)
        proportion_below_25 = below_25 / total_movies
        proportion_quarter_50 = quarter_50 / total_movies
        proportion_half_75 = half_75 / total_movies
        proportion_above_75 = above_75 / total_movies

        # Append proportions to respective lists
        below_25_per_decade.append(proportion_below_25)
        quarter_50_per_decade.append(proportion_quarter_50)
        half_75_per_decade.append(proportion_half_75)
        above_75_per_decade.append(proportion_above_75)

    # Extract unique decades
    unique_decades = female_proportions['decade'].unique()
    data = [below_25_per_decade, quarter_50_per_decade, half_75_per_decade, above_75_per_decade]

    return data, unique_decades


def subplot_proportion_movies_different_percentages_women(character_data, idx, subplot):
    """
    Creates an interactive subplot of the proportion of movies across decades that have less that 25% of women in them,
    between 25-50%, between 50-75% and more than 75%.
    :param character_data:  pandas dataframe: contains information on the characters we wish to plot. Must contain
    the columns "decade", as well as a "name" for the movies and "actor gender".
    :param idx: int: Where to plot the subplot in a 2x3 grid. The location is (idx%2+1, idx%3+1).
    :param subplot: plotly.subplots object: plot to which the subplot will be added
    """

    categories = ['0-25%', '25-50%', '50-75%', '75-100%']
    data, unique_decades = create_data_proportion_percentages_plot(character_data)

    # create all the traces
    for i, category in enumerate(categories):
        trace = go.Scatter(
            x=unique_decades,
            y=data[i],
            mode='lines',
            stackgroup='one',
            name=category,
            marker_color=px.colors.qualitative.Plotly[i]
        )
        # show legend only once
        if idx != 0:
            trace.update(dict(showlegend=False))
        subplot.append_trace(trace, row=idx % 2 + 1, col=idx % 3 + 1)


def plot_proportion_movies_different_percentages_women(character_data, save_fig=True, show_fig_png=True, folder='',
                                                       file_name="Proportion of Movies with Different Percentages of Female Actors.html"):
    """
    Displays an interactive graph of the proportion of movies across decades that have less that 25% of women in them,
    between 25-50%, between 50-75% and more than 75% using the plotly library
    :param character_data:  pandas dataframe: contains information on the characters we wish to plot. Must contain
    the columns "decade", as well as a "name" for the movies and "actor gender".
    :param save_fig: bool: if true, the interactive plot will be saved in an HTML file, default = True
    :param show_fig_png: bool: if true, the fig is displayed as a PNG (used for github display), else the figure is
    displayed as an interactive graph, default = True
    :param folder: string: name of the folder to save the figure in, default=''
    :param file_name: string: name of the file to save the figure in,
    default="Proportion of Movies with Different Percentages of Female Actors.html"
    """

    # Create traces for stack plot
    traces = []
    categories = ['0-25%', '25-50%', '50-75%', '75-100%']
    data, unique_decades = create_data_proportion_percentages_plot(character_data)
    for i, category in enumerate(categories):
        trace = go.Scatter(
            x=unique_decades,
            y=data[i],
            mode='lines',
            stackgroup='one',
            name=category
        )
        traces.append(trace)

    # Create layout
    layout = go.Layout(
        title='Proportion of Movies with Different Percentages of Female Actors',
        xaxis=dict(title='Decade'),
        yaxis=dict(title='Proportion of movies in a given decade'),
        legend=dict(x=1.05, y=1, traceorder='normal', orientation='v')
    )

    # Create figure
    fig = go.Figure(data=traces, layout=layout)

    if save_fig:
        fig.write_html(folder + file_name, auto_open=True)
    if show_fig_png:
        fig.show("png")
    else:
        fig.show()


def create_data_age_plots(character_data):
    """
    Create plotly go.Scatter objects that can be used to plot the average age of female and male actors in movies per
    decade and their respective 95% CI.
    :param character_data: pandas dataframe: contains information on the characters we wish to plot.
    Must contain the columns "actor_age", "actor_gender" and "decade".
    :return: plot_men, men_upper_bound, men_lower_bound, plot_women, women_upper_bound, women_lower_bound
    """

    # drop the actors with NaN as age but also actors with negative ages (which appeared in 1910)
    character_age = character_data.dropna(subset=['actor_age']).copy()
    character_age = character_age[character_age['actor_age'] >= 0]  # remove actors with a negative age
    # Group by decade and gender
    characters_age_by_decade_gender = character_age.groupby(['decade', 'actor_gender'])

    # Calculate average age and standard deviation of age for each gender per year
    gender_age_stats_per_decade = characters_age_by_decade_gender['actor_age'].agg(['mean', 'sem'])

    # Reset the index to make the grouped columns accessible for further analysis or visualization
    gender_age_stats_per_decade.reset_index(inplace=True)

    # Filter data for men and women
    men_data = gender_age_stats_per_decade[gender_age_stats_per_decade['actor_gender'] == 'M'].copy()
    women_data = gender_age_stats_per_decade[gender_age_stats_per_decade['actor_gender'] == 'F'].copy()

    # Calculate 95% confidence interval for men
    men_data['lower_bound'] = men_data['mean'] - 1.96 * men_data['sem']
    men_data['upper_bound'] = men_data['mean'] + 1.96 * men_data['sem']

    # Calculate 95% confidence interval for women
    women_data['lower_bound'] = women_data['mean'] - 1.96 * women_data['sem']
    women_data['upper_bound'] = women_data['mean'] + 1.96 * women_data['sem']

    # create the plotly objects
    plot_men = go.Scatter(
        name='Men Mean Age',
        x=men_data['decade'],
        y=men_data['mean'],
        mode='lines',
        line=dict(color='mediumturquoise'),
    )
    men_upper_bound = go.Scatter(
        name='Upper Bound',
        x=men_data['decade'],
        y=men_data['upper_bound'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    )
    men_lower_bound = go.Scatter(
        name='Lower Bound',
        x=men_data['decade'],
        y=men_data['lower_bound'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(173, 216, 230, 0.3)',
        fill='tonexty',
        showlegend=False
    )
    plot_women = go.Scatter(
        name='Women Mean Age',
        x=women_data['decade'],
        y=women_data['mean'],
        mode='lines',
        line=dict(color='hotpink'),
    )
    women_upper_bound = go.Scatter(
        name='Upper Bound',
        x=women_data['decade'],
        y=women_data['upper_bound'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    )
    women_lower_bound = go.Scatter(
        name='Lower Bound',
        x=women_data['decade'],
        y=women_data['lower_bound'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(255, 182, 193, 0.3)',
        fill='tonexty',
        showlegend=False
    )

    return plot_men, men_upper_bound, men_lower_bound, plot_women, women_upper_bound, women_lower_bound


def subplot_average_age_actor_across_decades(character_data, idx, subplot):
    """
    Creates an interactive subplot of the average age of female and male actors across decades along with their 95% CI.
    :param character_data: pandas dataframe: contains information on the characters we wish to plot.
    Must contain the columns "actor_age", "actor_gender" and "decade".
    :param idx: int: Where to plot the subplot in a 2x3 grid. The location is (idx%2+1, idx%3+1).
    :param subplot: plotly.subplots object: plot to which the subplot will be added
    """

    plot_men, men_upper_bound, men_lower_bound, plot_women, women_upper_bound, women_lower_bound = \
        create_data_age_plots(character_data)

    # show legend only once
    if idx != 1:
        plot_women.update(dict(showlegend=False))
        plot_men.update((dict(showlegend=False)))

    plots = [plot_men, men_upper_bound, men_lower_bound, plot_women, women_upper_bound, women_lower_bound]

    # append all traces to the main subplot
    for plot in plots:
        subplot.append_trace(plot, row=idx % 2 + 1, col=idx % 3 + 1)

    subplot.update_yaxes(range=[20, 55], row=idx % 2 + 1, col=idx % 3 + 1)


def plot_average_age_actor_across_decades(character_data, save_fig=True, show_fig_png=True, folder='',
                                          file_name="Average age of actor per decade per gender with 95% CI.html"):
    """
    Displays an interactive plot of the average age of female and male actors across decades along with their 95% CI.
    :param character_data: pandas dataframe: contains information on the characters we wish to plot.
    Must contain the columns "actor_age", "actor_gender" and "decade".
    :param save_fig: bool: if true, the interactive plot will be saved in an HTML file, default = True
    :param show_fig_png: bool: if true, the fig is displayed as a PNG (used for github display), else the figure is displayed
    as an interactive graph, default = True
    :param folder: string: name of the folder to save the figure in, default=''
    :param file_name: string: name of the file to save the figure in,
    default="Average age of actor per decade per gender with 95% CI.html"
    """

    plot_men, men_upper_bound, men_lower_bound, plot_women, women_upper_bound, women_lower_bound = \
        create_data_age_plots(character_data)

    # plot the plot with plotly to have an interactive plot
    fig = go.Figure([plot_men, men_upper_bound, men_lower_bound, plot_women, women_upper_bound, women_lower_bound])

    # Create figure
    fig.update_layout(
        yaxis_title='Age',
        xaxis_title='Decade',
        title='Average age of actors per decade per gender with 95% CI',
        hovermode="x",
        legend=dict(x=0, y=1, traceorder='normal', orientation='h')
    )

    # Save the Html to open in a browser tab
    if save_fig:
        fig.write_html(folder +file_name, auto_open=True)
    if show_fig_png:
        fig.show("png")
    else:
        fig.show()
