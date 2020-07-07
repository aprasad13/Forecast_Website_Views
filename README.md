# Wikipedia Traffic Forecast
### Introduction

This project focuses on the problem of forecasting the future values of multiple time series, as it has always been one of the most challenging problems in the field. More specifically, this aim on the problem of forecasting future web traffic for approximately 145,000 Wikipedia articles.

As this is a forecasting problem. The project shows the performance of following forecasting Models:
- SARIMA
- Facebook Prophet

### Data Description

The training dataset consists of approximately 145k time series. Each of these time series represent a number of daily views of a different Wikipedia article, starting from July, 1st, 2015 up until December 31st, 2016.

train_*.csv - contains traffic data. This a csv file where each row corresponds to a particular article and each column correspond to a particular date. Some entries are missing data. The page names contain the Wikipedia project (e.g. en.wikipedia.org), type of access (e.g. desktop) and type of agent (e.g. spider). In other words, each article name has the following format: 'name_project_access_agent' (e.g. 'AKB48_zh.wikipedia.org_all-access_spider').

Click on the link to access the train data
https://render.githubusercontent.com/view/ipynb?commit=193ae58417e2abf88b973c874072ad65f98d0228&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6170726173616431332f57696b6970656469615f547261666669635f466f7265636173742f313933616535383431376532616266383862393733633837343037326164363566393864303232382f436f64655f576562566965775f56332e6970796e62&nwo=aprasad13%2FWikipedia_Traffic_Forecast&path=Code_WebView_V3.ipynb&repository_id=277751425&repository_type=Repository#Time-series-extraction
