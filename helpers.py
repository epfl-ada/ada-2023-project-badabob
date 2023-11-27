import requests
from bs4 import BeautifulSoup
import ast
from imdb import Cinemagoer
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
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

######################################### COMPLEMENTING DATASETS #######################################################


def wikipedia_query(save_file=True, save_path='DATA/', filename='wiki_queries.csv'):
    """
    Retrives IMDB, freebase ID  and title of movies on wikipedia. If the query crashes, the request is made again after
    5s. for a maximal of 10 tries.
    :param save_file: boolean: whether to save the created dataframe in a csv file, default = True
    :param save_path: string: path where the data will be saved, default = 'DATA/'
    :param filename: string: name of the file to save, default = 'wiki_queries.csv'
    :return: a dataframe containing IMDB ID, freebase ID and title
    """
    # Call the wikidata query service
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # Create the query
    sparql.setQuery("""
    SELECT ?work ?imdb_id ?freebase_id ?label
    WHERE
    {
      ?work wdt:P31/wdt:P279* wd:Q11424.
      ?work wdt:P345 ?imdb_id.
      ?work wdt:P646 ?freebase_id.

      OPTIONAL {
        ?work rdfs:label ?label.
      }

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
                    'label': binding['label']['value'] if 'label' in binding else None,
                }
                data.append(row)

            # Create a DataFrame
            wiki_df = pd.DataFrame(data)

            # remove duplicates
            wiki_df_filtered = wiki_df.drop_duplicates('IMDB_ID')
            wiki_df_filtered = wiki_df_filtered.drop_duplicates('freebase_ID')

            if save_file:
                wiki_df_filtered.to_csv(save_path + filename, index=False)
                print(f'file {save_path + filename} saved')

            return wiki_df

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

    timeline.to_csv(save_path+file_name, index=False)
    print(f'Timeline saved in {save_path+file_name}')


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
            movie = ia.get_movie(id[2:]) # the first 2 characters of the ID need to be removed
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
    IMDB_data.to_csv(save_path+file_name, index=False)


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
    IMDB_data_us_en = IMDB_data_us[IMDB_data_us['languages'].str.contains('English' or 'American Sign', case=False)].copy()
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
    characters_data['characters'] = characters_data['characters'].apply(lambda x: ast.literal_eval(x) if x != '\\N' else x)
    # Extract strings from lists
    characters_data['characters'] = characters_data['characters'].apply(lambda x: x[0]if isinstance(x, list) and len(x) > 0 else x)

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


######################################### MAIN CHARACTER ANALYSIS ######################################################

def select_elligible_movies_for_main_char_analysis(movie_metadata,character_metadata):
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
    for wiki_id,imdb_id in zip(elligible_movies['wikipedia_ID'],elligible_movies['IMDB_ID']):
        
        if not pd.isnull(wiki_id) and pd.isnull(imdb_id):
            if character_metadata.loc[character_metadata['wikipedia_ID']==wiki_id].empty:
                movies_with_no_listed_char_wiki_ID.append(wiki_id)
            continue

        if pd.isnull(wiki_id) and not pd.isnull(imdb_id):
            if character_metadata.loc[character_metadata['IMDB_ID']==imdb_id].empty:
                movies_with_no_listed_char_imdb_ID.append(imdb_id)
            continue

        if not pd.isnull(wiki_id) and not pd.isnull(imdb_id):
            no_char_on_wikipedia = character_metadata.loc[character_metadata['wikipedia_ID']==wiki_id].empty
            no_char_on_imdb = character_metadata.loc[character_metadata['IMDB_ID']==imdb_id].empty

            if no_char_on_wikipedia and no_char_on_imdb:
                movies_with_no_listed_char_wiki_ID.append(wiki_id)
                movies_with_no_listed_char_imdb_ID.append(imdb_id)

    elligible_movies = elligible_movies[~elligible_movies['wikipedia_ID'].isin(movies_with_no_listed_char_wiki_ID)]
    elligible_movies = elligible_movies[~elligible_movies['IMDB_ID'].isin(movies_with_no_listed_char_imdb_ID)]

    id_types = ['wikipedia_ID','IMDB_ID']
    for id_type in id_types:

        # Removing movies with characters having name listed as nan
        movies_with_nan_characters_ID = pd.Series(character_metadata[character_metadata['character_name'].isna()][id_type].unique())
        movies_with_nan_characters_ID = movies_with_nan_characters_ID.dropna().tolist()
        elligible_movies = elligible_movies[~elligible_movies[id_type].isin(movies_with_nan_characters_ID)]

        # Removing movies with characters having gender listed as nan
        movies_with_nan_genders_ID = pd.Series(character_metadata[character_metadata['actor_gender'].isna()][id_type].unique())
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
    selected_character_metadata = pd.concat([IMDB_ID_character_list,wikipedia_ID_character_list],ignore_index=True)

    genders = []
    for name in movie_row['main characters']:
        confidence = 0
        if selected_character_metadata['character_name'].any():
            closest_character, confidence, score = process.extractOne(name, selected_character_metadata['character_name'])
            
        if confidence > 50:
            gender = selected_character_metadata.loc[selected_character_metadata['character_name'] == closest_character, 'actor_gender'].values[0]
            genders.append(gender)

    return genders


def calculate_gender_ratio(genders:list):
    """
    Calculates the female to male gender ratio in a list of strings
    containing ['M'] and ['F']
    :param genders: list of strings
    :return ratio: float
    """
    flat_list = [gender for sublist in genders for gender in sublist]
    genders = [gender for gender in flat_list if gender in ['F', 'M']]
    nb_females = sum(1 for gender in genders if gender == 'F')
    if len(genders) == 0:
        return float('nan')
    return nb_females/len(genders)


def plot_gender_ratio(gender_list_per_year):
    """
    Plots the female to male gender ratio across time in years
    :param gender_list_per_year: pandas dataframe
    """
    plt.figure(figsize=(10, 6))
    sns.set(style='whitegrid')
    sns.lineplot(x=gender_list_per_year['release_date'], y=gender_list_per_year['gender_ratio'], linewidth=2.5, alpha=0.8)
    sns.regplot(x=gender_list_per_year['release_date'], y=gender_list_per_year['gender_ratio'], scatter=False, color='red',label="Linear Regression, bootstrap, 95% Conf. Int.",ci=95)

    plt.xlabel('Movie release year')
    plt.ylabel('Female/Male ratio')
    plt.title('Gender ratio in main characters over time')
    plt.legend()
    plt.show()

    regression_results = linregress(gender_list_per_year['release_date'], gender_list_per_year['gender_ratio'])
    print(regression_results)


def random_movies_per_year(group):
    """
    Selects 20 movies at random
    :param group: pandas dataframe
    """
    return group.sample(n=min(20, len(group)), random_state=10)

################################## GENRES PREPROCESSING ################################################################


def no_genres_in_list(genres, df):
    ID_no_accepted_genre=[]
    for index,row in df.iterrows():
        ID=row['name']
        genre=row['genre']
        if all(element in genres for element in genre) and genre!=[]:
            ID_no_accepted_genre.append(ID)
    print("Number of movies : ",len(ID_no_accepted_genre))
    return ID_no_accepted_genre


