A MODIFIER !!! ceci est un premier jet

# Hollywood Unveiled: Rewriting Women's Stories on the Silver Screen


The perception and role of women in the US society has greatly evolved during the past centuries. Indeed, between the beginning of the 20th century and the beginning of the 21st century,
women were granted a myriad of new rights. Simultaneously, their perception has also evolved, shifting away from mere housewives. Women can now vote, work without the approval of their husband and some are even leading big companies, such as Mary Barra who sits at the head of General Motors. 
To this day still, women keep fighting for their right to be treated fully equally with respect to their men counterparts. 
Movies are often seen as a reflection of society, moving along with its ebbs and flows.
But has this specific social movement bled into the entertainment industry? Has the portrayal of women evolved along the last
century? Has their depiction progressed as important milestones in the Feminism movement were reached ? Or on the contrary,
do women stereotypes still prevail in Hollywood ? 
Simply put, is the entertainment industry a mirror of women's evolution in society?

## Table of Contents

1. [Research Questions](#research-questions)
2. [Additional Datasets](#additional-datasets)
3. [Methods](#methods)
4. [Proposed Timeline](#proposed-timeline)
5. [Organization Within the Team](#organization-within-the-team)
6. [Questions for TAs](#questions-for-tas)

## Research questions
**--> Research Questions: A list of research questions you would like to address during the project.**  
How has the proportion of female actors in movies evolved over time? 
Is the movie industry giving more important and leading roles to women? Do they have their own storylines or are just love interests?
What types of adjectives are used in synopsis when describing women and to which character personas are women mostly attributed?
Which movie genres are more affected?
Do higher-grossing movies rely more on gender stereotypes than others?
This information will enable us to understand how gender stereotypes in movies fluctuate with time and whether this representation is affected by major women’s rights events in the US (e.g. right to vote, etc. found on Wikipedia). 

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
More than 3000 movies were deleted from the [CMU dataset](https://www.cs.cmu.edu/~ark/personas/) since they had 
unknown languages. One of our next step would be to find the language of these movies on IMDB using 
[Cinemagoer](https://github.com/cinemagoer/cinemagoer). This can be done easily since Cinemagoer allows to research
movies not only with IMDB IDs but also with movie titles.

Another important variable that we need to complete is the personnas. We will not use another dataset to do so but run the 
pipeline described in [Bamman et al., 2013](https://www.cs.cmu.edu/~dbamman/pubs/pdf/bamman+oconnor+smith.acl13.pdf)
using the [code of the authors](https://github.com/dbamman/ACL2013_Personas). This is expected to take a
long time, so we will run it on a random subset of plot summaries for each year to reduce computational time.

## Methods

1. **CMU dataset exploration and pre-processing**\
Preliminary exploration of the dataset to get familiar with it. Pre-processing according to our research questions
(e.g only keeping movies from the United States that are in English). 

2. **Data completion**\
Completion of the dataset up until 2022 using IMDb to obtain the final movie dataset and the final character dataset.

3. **Acquisition of personnas for all characters**\
Running the personnas algorithm on more of the characters in our dataset to obtain a meaningful representation
of the personnas over time. 

4. **Preliminary analysis**\
Analysis of the proportion of women in the movie industrie per year.  
Analysis of the age distribution of men vs women and the average age of both gender per year and comparison of the distributions using a Student t-test.

5. **Personnas analysis**\
Extraction of the most attributed personna per year for men and women.
Analyse the evolution of the personnas attributed to men and women over time.

6. **Main vs Secondary Characters** \
Determine the proportion of women in leading roles and its evolution.\
TO BE COMPLETED BY JULIAN

7. **Regression for representation**\
Regression analysis to predict when equal representation of women and men in movies will theoretically be reached according to the trends of the last century.

8. **Genre Analysis**\
Analysis of which personnas are more present in different genres using an ANOVA test.
Re-do previous analysis on data split by genre to see whether women's representation vary between genres

9. **Analyze Box office revenue**\
Look at the correlation between successful movies (high box office revenue) and women
representation using Pearson correlation coefficient or linear regression.

10. **Characteristics of stereotypical women characters**\
Use clustering methods to find common factors that make a character fall into stereotypical personnas.

## Proposed timeline

* 17/11/2023 - Preliminary data processing, data completion, first analysis of movies, characters and genres, implementation of natural language processing algorithm + ***Deliver Milestone 2***
* 24/11/2023 - Preliminary analysis and obtain personnas for a larger set of characters and 
* 01/12/2023 - Personnas analysis and implement regression for representation  + ***Deliver Homework 2***
* 08/12/2023 - Main vs Secondary character analysis and analysis by genre
* 15/12/2023 - Data Story writing, web page design and characteristics of stereotypical women characters
* 22/12/2023 - ***Deliver Milestone 3***

## Organization within the team

<table class="tg">
  <tr>
    <th>Team Member</th>
    <th>Tasks</th>
  </tr>
  <tr>
    <td>Julian Bär</td>
    <td>Natural Language Processing for Main/Secondary characters identification<br>
        statistical and clustering analysis of Main/Secondary character evolution<br>
        Web page design
    </td>
  </tr>
  <tr>
    <td>Lucille Niederhauser</td>
    <td>Data completion<br>
        Time Line Analysis<br>
        New personnas computation and analysis
    </td>
  </tr>
  <tr>
    <td>Victoria Rivet</td>
    <td>Gender and personnas preliminary analysis<br>
        Regression analysis of actors' gender<br>
        Correlation of the results with historical events<br>
        Text for datastory
    </td>
  </tr>
  <tr>
    <td>Anabel Salazar</td>
    <td>Data preprocessing and filtering<br>
        Statistical analysis of actors' gender<br>
        Analysis of women's features (other than gender)<br>
        Text for datastory
    </td>
  </tr>
  <tr>
    <td>Maximilian Wettstein</td>
    <td>Movie genres primary analysis and clustering<br>
        Clustering analysis<br>
        Generate interacting visualizations for datastory
    </td>
  </tr>
</table>

## Questions for TAs

1. Is it enough to deal with the increasing number of movies every year by normalizing by the number of movies we have in a 
given year? 
