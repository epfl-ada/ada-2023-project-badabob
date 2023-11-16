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
**--> Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.**

## Methods

1. **CMU dataset exploration and pre-processing**\
First we explored the dataset to get familiar with it. We then pre-processed it according to our research questions.
For example only keeping movies from the United States that were in English. 

2. **Data completion**\
Given that the CMU dataset was created in 2012, we completed it up until 2020 using IMD. This allowed us to obtain the final
movie dataset as well as the final character dataset.

3. **Acquisition of personnas for all characters**\
Run the algorithm of personnas on the entirety of the characters in our dataset to obtain a more meaningful representation
of the personnas over time. 

4. **Preliminary analysis**\
First, analyze the age distribution of men vs women using a Student t-test
Then, analyze the personnas attributed to women and extract the most attributed personna per year for men and women

5. **Genre Analysis**\
Analyze which personnas are more present in different genres using an ANOVA test.

6. **Main vs Secondary Characters** \
Explore how the proportion of women in leading roles has evolved\
TO BE COMPLETED BY JULIAN
7. **Regression for representation**\
When can we expect equal representation of women and men in movies ? Using a regression analysis
we will predict, according to the trends of the last century, when this point will be reached. 
8. **Analyze Box office revenue**\
Look at the correlation between successful movies (high box office revenue) and women
representation using ???
9. **Characteristics of Stereotypical women characters**\
Find characteristics of characters that are stereotypical and non-stereotypical women. 
Using a PCA to find the main factors that make a character fall into one of these categories. 


## Proposed timeline

* 17/11/2023 - Preliminary data processing, data completion, preliminary analysis of movies, characters and genres, implementation of natural language processing algorithm + ***Deliver Milestone 2***
* 24/11/2023 - Obtain personnas for a larger set of characters, and implement statistical analysis on actors' gender and movie genres 
* 01/12/2023 - Implement a regression analysis as well as clustering methods + ***Deliver Homework 2***
* 08/12/2023 - Extract results for each research question 
* 15/12/2023 - Data Story writing and web page design
* 22/12/2023 - ***Deliver Milestone 3***

## Organization within the team
**--> Organization within the team: A list of internal milestones up until project Milestone P3.**


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
