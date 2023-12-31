# Hollywood Unveiled: Rewriting Women's Stories on the Silver Screen

The perception and role of women in the US society has greatly evolved during the past centuries. Since the beginning of the 20th century,
women were granted a myriad of new rights. Simultaneously, their perception has also evolved, shifting away from mere housewives. Women can now vote, work without their husband's approval and some are even leading big companies, such as Mary Barra who sits at the head of General Motors. 
To this day still, women keep fighting for their right to be treated fully equally with respect to men. 
Movies are often seen as a reflection of society, moving along with its ebbs and flows.
But has this specific social movement bled into the entertainment industry? Has the depiction of women progressed as important milestones in the Feminism movement were reached such as the Equal Pay Act? Or on the contrary, do women stereotypes still prevail in Hollywood? 

## Table of Contents

1. [Datastory](#datastory)
2. [Research Questions](#research-questions)
3. [Additional Datasets](#additional-datasets)
4. [Methods](#methods)
5. [Proposed Timeline](#proposed-timeline)
6. [Organization Within the Team](#organization-within-the-team)
7. [Questions for TAs](#questions-for-tas)

## Datastory
Uncover the evolution of women representation in movies and how history influenced it by using this [link](https://anabeldorado.github.io/badabob-datastory.github.io/) to our datastory.

## Directory Organisation
* DATA: Folder containing intermediate data output of the notebook M3_DataCreation&Exploration. These outputs were saved and are provided here to avoid having to re-run time consuming functions.
* M2.ipynb: Milestone 2 jupyter notebook.
* M3_Analysis.ipynb: Milestone 3 jupyter notebook that contains our data analysis. The data used in this notebook is created in M3_DataCreation&Exploration.ipynb.
* M3_DataCreation&Exploration.ipynb: Milestone 3 jupyter notebook that contains the code that created the data used in M3_Analysis.ipynb.
* helpers.py: Contains helper functions used in our notebooks.

## Research questions
The main question we aim to adress is: how has women's representation evolved over time in movies and does it relate to historical events linked to women's history in the United States?
The following questions will help us to address this: 
* How has the proportion of female actors in movies evolved over time? 
* Is the movie industry giving more important and leading roles to women?
* What are the characteristics of stereotypical movies?
* Which vocabulary is used when describing women and men and does this evolve through time?
* Which movie genres are more affected?

## Additional datasets

### A) Data from 2010 to 2022

The [CMU dataset](https://www.cs.cmu.edu/~ark/personas/) only contains movies until 2012. We decided to complete our
dataset until 2022. As a first step, we used the following 
[IMDB non-comercial datasets](https://datasets.imdbws.com/):


1) **title.principal.tsv**: contains information on the crew for each movie such as the job of the person and the 
name of the character played if it applies.
2) **names.basics.tsv**: contains information on movie crews such as birth years and professions.
3) **title.akas.tsv**: contains multiple titles for each movie and information on each title such as language and country 
in which it was used, and whether it is the original title of the movie.
4) **title.basic.tsv**: contains information on movies such as the release date, the primary title and the genres.

Datasets 1. and 2. were used to complete information on the characters. Genders were found by finding out whether the 
job of the crew members were 'actor' or 'actress'. The age of each actor when starring in the movie could also be
completed using the actor's birth year and the release year of each movie.

Some important information is still missing from these datasets such as plot summaries, countries of production
and languages. We thus used dataset 3. and 4. to collect IDs of movies that had a release date between 2010 and 2023 and 
for which the american movie title is the same as the original title. We then used these IDs to directly retrieve 
all the needed information on movies from  [IMDB](https://www.imdb.com/) using the 
[Cinemagoer](https://github.com/cinemagoer/cinemagoer) Python package.

### B) Data before 2010
Movies in the [CMU dataset](https://www.cs.cmu.edu/~ark/personas/) did not contain their IMDB IDs. In order to have a common ID
between the provided dataset and the one retrieved from IMDB, we queried from [wikipedia](https://www.wikipedia.org/)
the IMDB IDs corresponding to the freebase IDs of movies in the [CMU dataset](https://www.cs.cmu.edu/~ark/personas/). We also 
queried at the same time the box-office revenues of movies, since we had only little data on these in the original dataset.
Finally, more than 3000 movies were deleted from the [CMU dataset](https://www.cs.cmu.edu/~ark/personas/) since they had 
unknown languages. We found the language of these movies on again IMDB using [Cinemagoer](https://github.com/cinemagoer/cinemagoer).

## Methods

**What factors do you plan to use for clustering? I don't see how it is relevant to gender bias either. Could be explained more.**

1. **CMU dataset exploration and pre-processing**\
Preliminary exploration of the dataset to get familiar with it. Pre-processing according to our research questions
(e.g only keeping movies from the United States that are in English). 

2. **Data completion**\
Completion of the dataset up until 2022 using IMDb to obtain the final movie dataset and the final character dataset.

3. **Preliminary analysis**\
Analysis of the proportion of women in the movie industry per year.  
Analysis of the age distribution of men vs women and the average age of both gender per year and comparison of the distributions using a Student t-test.

4. **Analysis of words linked to each genre**\
Extraction of the most attributed adjectives, words and verbs per year to men and women.
Analyse the evolution of words attributed to men and women over time.
Study how varied the words associated to each gender are.

5. **Main Characters** \
Extract main characters and their gender from movie summaries to determine the proportion of women in leading roles and its evolution.\

6. **Regression for representation**\
Regression analysis to predict when equal representation of women and men in movies will theoretically be reached according to the trends of the last century.

7. **Genre Analysis**\
Re-do previous analysis (3 to 6) on data split by genre to see whether women's representation vary between genres.

8. **Clustering of stereotypical movies**\
Cluster movies according to their proportion of women, proportion of women main character and age differences between genders in order to identify more stereotypical movies.
Analysis of genre and evolution through time of most and less stereotypical movies.

## Proposed timeline

* 17/11/2023 - Preliminary data processing, data completion, first analysis of movies, characters and genres, implementation of natural language processing algorithm + ***Deliver Milestone 2***
* 24/11/2023 - Preliminary analysis and obtain personas for a larger set of characters 
* 01/12/2023 - Personas analysis and implement regression for representation  + ***Deliver Homework 2***
* 08/12/2023 - Main vs Secondary character analysis and analysis by genre
* 15/12/2023 - Characteristics of stereotypical women characters
* 22/12/2023 - Data Story writing and web page design + ***Deliver Milestone 3***

## Organization within the team

<table class="tg">
  <tr>
    <th>Team Member</th>
    <th>Tasks</th>
  </tr>
  <tr>
    <td>Julian Bär </td>
    <td>Natural Language Processing for Main characters identification<br>
        Analysis of main characters evolution<br>
        Web page design and datastory writting
    </td>
  </tr>
  <tr>
    <td>Lucille Niederhauser </td>
    <td>Data completion<br>
        Time Line Analysis<br>
        Clustering analysis<br>
        Interactive plots generation
    </td>
  </tr>
  <tr>
    <td>Victoria Rivet</td>
    <td>Gender and personas preliminary analysis<br>
        Natural Language Processing to get words associated to characters<br>
        Analysis of the associated words by gender and genres<br>
        Code cleaning
    </td>
  </tr>
  <tr>
    <td>Anabel Salazar</td>
    <td>Data preprocessing and filtering<br>
        Statistical analysis of actors' gender<br>
        Interactive plots generation<br>
        Web page design and datastory writting
    </td>
  </tr>
  <tr>
    <td>Maximilian Wettstein</td>
    <td>Movie genres analysis<br>
        Natural Language Processing to get words associated to characters<br>
        Analysis of the associated words by gender and genres<br>
        Code cleaning
    </td>
  </tr>
</table>
