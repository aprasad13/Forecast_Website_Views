# Wikipedia Traffic Forecast
### Introduction

This project focuses on the problem of forecasting the future values of multiple time series, as it has always been one of the most challenging problems in the field. More specifically, this aim on the problem of forecasting future web traffic for approximately 145,000 Wikipedia articles.

As this is a forecasting problem. The project shows the performance of following forecasting Models:
- SARIMA
- Facebook Prophet

### Data Description

The training dataset consists of approximately 145k time series. Each of these time series represent a number of daily views of a different Wikipedia article, starting from July, 1st, 2015 up until December 31st, 2016.

train_*.csv - contains traffic data. This a csv file where each row corresponds to a particular article and each column correspond to a particular date. Some entries are missing data. The page names contain the Wikipedia project (e.g. en.wikipedia.org), type of access (e.g. desktop) and type of agent (e.g. spider). In other words, each article name has the following format: 'name_project_access_agent' (e.g. 'AKB48_zh.wikipedia.org_all-access_spider').

Click on the link to access the [train data](https://drive.google.com/file/d/1tx_PNttgS-76EXMCSntWI1r4Rv1MLJfa/view?usp=sharing)

### Methodology Used
#### Missing values

There are about 8% of missing values in this data set, which is not trivial.

#### Data transformation and helper functions
These are the tools we need for a visual examinination of arbitrary individual time series data.

##### Article names and metadata
As it was said that each article name has the a format: 'name_project_access_agent'. To make the training data easier to handle we split it into two part: the article information (from the Page column) and the time series data (tdates) from the date columns. We briefly separate the article information into data from wikipedia, wikimedia, and mediawiki due to the different formatting of the Page names. After that, we rejoin all article information into a common data set (tpages).

##### Time series extraction
In order to plot the time series data we use a helper function that allows us to extract the time series for a specified row number. In addition, with the help of the extractor tool we define a function that re-connects the Page information to the corresponding time series and plots this curve according to our specification on article name, access type, and agent for all the available languages.

#### Summary parameter extraction
In this section we will have a more global look at the population parameters of our training time series data. Also here, we will start with the wikipedia data. The idea behind this approach is to probe the parameter space of the time series information along certain key metrics and to identify extreme observations that could break our forecasting strategies. We took the help of mean, standard deviation, amplitude, and a the slope of a naive linear fit of every article.

#### Individual observations with extreme parameters
Based on the overview parameters (seen in the previous section) we can focus our attention on those articles for which the time series parameters are at the extremes of the parameter space. We found the articles which are extreme in terms of following parameters:

- Large linear slope
- High standard deviations
- Large variability amplitudes
- High average views





