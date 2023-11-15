import pandas as pd
import requests
from bs4 import BeautifulSoup
from imdb import Cinemagoer
import ast
import numpy as np


######################################### COMPLEMENTING DATASETS #######################################################

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
    i = 0

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
    # drop duplicates movie if any
    IMDB_data_us_en = IMDB_data_us_en.drop_duplicates(subset='IMDB_ID', keep='first')

    return IMDB_data_us_en


def filter_IMDB_character_dataset(dataframe, IMDB_ids):
    """
    Filters the IMDB character dataset: removes all jobs other than actresses and actors,
    :param dataframe:
    :param IMDB_ids:
    :return:
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


def merge_datasets_characters(characters_data, actors_data, movie_data):
    characters_data_merged = pd.merge(characters_data, actors_data, on='nconst', how='left').copy()
    # remove columns we do not need in the character dataset
    characters_movies_data = movie_data.drop(columns=['languages', 'countries', 'genre', 'plot_summary'])
    characters_movies_data = characters_movies_data.rename(columns={'IMDB_ID': 'tconst'})
    characters_data_final = pd.merge(characters_data_merged, characters_movies_data, on='tconst', how='left').copy()
    # change name so that they are the same as our other dataset
    characters_data_final = characters_data_final.rename(
        columns={'tconst': 'IMDB_ID', 'nconst': 'actor_IMDB_ID', 'characters': 'character_name',
                 'primaryName': 'actor_name', 'birthYear': 'actor_birthday'})
    # compute actor age the year of the release
    characters_data_final.loc[characters_data_final['actor_birthday'] == '\\N', 'actor_birthday'] = np.nan
    characters_data_final['actor_birthday'] = (characters_data_final['actor_birthday']).astype(float)
    # Calculate age in years as an integer
    characters_data_final['actor_age'] = (
                characters_data_final['release_date'] - (characters_data_final['actor_birthday']).astype(float))

    return characters_data_final.drop(columns='ordering')


def remove_duplicated_columns(dataframe, columns_to_remove):
    # when merges columns _x and _y are created
    for col_name in columns_to_remove:
        mask = dataframe[col_name + '_x'].isna()
        dataframe.loc[mask, col_name + '_x'] = dataframe.loc[mask, col_name + '_y']
        dataframe = dataframe.drop(columns=col_name + '_y')
        dataframe = dataframe.rename(columns={col_name + '_x': col_name})

    return dataframe


def name_to_lowercase(dataframe, column_name):
    return [c.lower() if pd.notna(c) else c for c in dataframe[column_name]]


