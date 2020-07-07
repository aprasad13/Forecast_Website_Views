from sklearn.model_selection import train_test_split
import copy 
import datetime
import statistics
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import random
import math
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

#------------------------------------------------------------------------------------------------------------------------------------
# Load data
df_train1 = pd.read_csv('/Users/amanprasad/Documents/Kaggle/Web_Traffic_Time_Series_Forecasting/web-traffic-time-series-forecasting/train_1.csv')
df_train1.shape
#(145063, 551)

df_key1 = pd.read_csv('/Users/amanprasad/Documents/Kaggle/Web_Traffic_Time_Series_Forecasting/web-traffic-time-series-forecasting/key_1.csv')
df_key1.shape
#(8703780, 2)

#df_train2 = pd.read_csv('/Users/amanprasad/Documents/Kaggle/Web_Traffic_Time_Series_Forecasting/web-traffic-time-series-forecasting/train_2.csv')
#df_train2.shape
#(145063, 804)

#df_key2 = pd.read_csv('/Users/amanprasad/Documents/Kaggle/Web_Traffic_Time_Series_Forecasting/web-traffic-time-series-forecasting/key_2.csv')
#df_key2.shape
#(8993906, 2)

#train_2= Each of these time series represent a number of daily views of a different Wikipedia article, starting from July, 1st, 2015 
#up until September 1st, 2017

#key_2= based on predictions of daily views between September 13th, 2017 and November 13th, 2017

train=copy.deepcopy(df_train1) 
key=copy.deepcopy(df_key1)

# Missing values in train
null_val=null_val_desc(train)
null_val['missing_count'].sum()/(train.shape[0]*train.shape[1])*100
# percentage of data missing = 6.025301136102712

#------------------------------------------------------------------------------------------------------------------------------------
#Data transformation and helper functions
#Article names and metadata

#To make the training data easier to handle we split it into two part: the article information (from the Page column) and the 
#time series data (tdates) from the date columns. We briefly separate the article information into data from wikipedia, wikimedia, 
#and mediawiki due to the different formatting of the Page names. After that, we rejoin all article information into a common 
#data set (tpages).

tdates=train.loc[:,~train.columns.isin(['Page'])]

foo=pd.DataFrame(data= train.loc[:,'Page'],columns=['Page'])
foo['rowname'] = foo.index
foo = foo[['rowname', 'Page']]

mediawiki=foo[foo['Page'].str.contains("mediawiki")]
wikimedia=foo[foo['Page'].str.contains("wikimedia")]

wikipedia=foo[~foo['Page'].str.contains("wikimedia")]
wikipedia=wikipedia[~wikipedia['Page'].str.contains("mediawiki")]

#-----------------------------------------------------
# creating wikipedia dataframe
new = wikipedia["Page"].str.split(".wikipedia.org_", expand = True) 
article=new.iloc[:,0].str[:-3]
locale=new.iloc[:,0].str[-2:]
new1 = new.iloc[:,1].str.split("_", expand = True)

# combining all
wikipedia['article']=article
wikipedia['locale']=locale
wikipedia['access']=new1.iloc[:,0]
wikipedia['agent']=new1.iloc[:,1]

# remove Page column from wikipedia
wikipedia.drop(['Page'],axis=1,inplace=True)

#-----------------------------------------------------
# creating wikimedia dataframe
new = wikimedia["Page"].str.split("_commons.wikimedia.org_", expand = True) 
new1 = new.iloc[:,1].str.split("_", expand = True)
wikimedia['article']=new.iloc[:,0]
wikimedia['access']=new1.iloc[:,0]
wikimedia['agent']=new1.iloc[:,1]
wikimedia['locale']='wikmed'

# remove Page column from wikimedia
wikimedia.drop(['Page'],axis=1,inplace=True)

#-----------------------------------------------------
# creating mediawiki dataframe
new = mediawiki["Page"].str.split("_www.mediawiki.org_", expand = True) 
new1 = new.iloc[:,1].str.split("_", expand = True)
mediawiki['article']=new.iloc[:,0]
mediawiki['access']=new1.iloc[:,0]
mediawiki['agent']=new1.iloc[:,1]
mediawiki['locale']='medwik'

# remove Page column from mediawiki
mediawiki.drop(['Page'],axis=1,inplace=True)

#-----------------------------------------------------
# combining wikipedia, wikimedia, mediawiki
frames = [wikipedia, wikimedia, mediawiki]
tpages = pd.concat(frames)
tpages.reset_index(drop=True,inplace=True)
tpages.shape
#(145063, 5)
train.shape
#(145063, 804)

#------------------------------------------------------------------------------------------------------------------------------------
#Time series extraction
#In order to plot the time series data we use a helper function that allows us to extract the time series for a specified row number. 
#(The normalised version is to facilitate the comparision between multiple time series curves, to correct for large differences 
#in view count.)

def extract_ts(rownr):
    df=pd.DataFrame(data=tdates.iloc[rownr,:])
    df=df.reset_index()
    df=df.rename(columns={"index": "dates", rownr: "views"})
    df['dates']= pd.to_datetime(df['dates'],format='%Y-%m-%d')
    return df

extract_ts(4)


def extract_ts_nrm(rownr):
    df=pd.DataFrame(data=tdates.iloc[rownr,:])
    df=df.reset_index()
    df=df.rename(columns={"index": "dates", rownr: "views"})
    df['dates']= pd.to_datetime(df['dates'],format='%Y-%m-%d')
    df['views']=df['views']/df['views'].mean()
    return df

#-----------------------------------------------------------------------------
# A custom-made plotting function allows us to visualise each time series and extract its meta data:

def plot_rownr(rownr):
    art=tpages.loc[tpages['rowname'] == rownr, 'article'].iloc[0]
    loc=tpages.loc[tpages['rowname'] == rownr, 'locale'].iloc[0]
    acc=tpages.loc[tpages['rowname'] == rownr, 'access'].iloc[0]
    b=extract_ts(rownr)
    b.set_index('dates',inplace=True)
    b['views: 30 Day Mean'] = b['views'].rolling(window=30).mean()
    title=('{a}-{l}-{ac}'.format(a=art,l=loc ,ac=acc))
    ax=b[['views','views: 30 Day Mean']].plot(title=title).autoscale(axis='x',tight=True)    

plot_rownr(1)

    
def plot_rownr_log(rownr):
    art=tpages.loc[tpages['rowname'] == rownr, 'article'].iloc[0]
    loc=tpages.loc[tpages['rowname'] == rownr, 'locale'].iloc[0]
    acc=tpages.loc[tpages['rowname'] == rownr, 'access'].iloc[0]
    b=extract_ts_nrm(rownr)
    b.set_index('dates',inplace=True)
    b['log(views)']=np.log(b['views'])
    b['log(views): 30 Day Mean'] = b['log(views)'].rolling(window=30).mean()
    title=('{a}-{l}-{ac}'.format(a=art,l=loc ,ac=acc))
    ax=b[['log(views)','log(views): 30 Day Mean']].plot(title=title).autoscale(axis='x',tight=True)

plot_rownr_log(1)

def plot_rownr_zoom(rownr, start, end):
    art=tpages.loc[tpages['rowname'] == rownr, 'article'].iloc[0]
    loc=tpages.loc[tpages['rowname'] == rownr, 'locale'].iloc[0]
    acc=tpages.loc[tpages['rowname'] == rownr, 'access'].iloc[0]
    b=extract_ts(rownr)
    b.set_index('dates',inplace=True)
    b['views: 30 Day Mean'] = b['views'].rolling(window=30).mean()
    title=('{a}-{l}-{ac}'.format(a=art,l=loc ,ac=acc))
    b[['views','views: 30 Day Mean']].plot(title=title,xlim=[start,end])


plot_rownr_zoom(1,'2015-07-01','2015-12-05')

#In addition, with the help of the extractor tool we define a function that re-connects the Page information to the corresponding 
#time series and plots this curve according to our specification on article name, access type, and agent for all the available languages:

def plot_names(art, acc, ag):
    pick=tpages[(tpages['article']==art)&(tpages['access']==acc)&(tpages['agent']==ag)]
    pick_nr=pick['rowname']
    pick_loc=pick['locale']
    foo=extract_ts(pick.index[0])
    foo['loc']=pick_loc[pick.index[0]]
    
    for i in range(1,len(pick.index)):
        foo1=extract_ts(pick_nr.iloc[i])
        foo1['loc']=pick_loc.iloc[i]
        foo=pd.concat([foo,foo1])

    sns.lineplot(x='dates',y='views',data=foo,hue='loc').set_title(art+'-'+acc+'-'+ag)
    return foo['loc'].unique()

plot_names("The_Beatles", "mobile-web", "all-agents")

#------------------------------------------
def plot_names_nrm(art, acc, ag):
    pick=tpages[(tpages['article']==art)&(tpages['access']==acc)&(tpages['agent']==ag)]
    pick_nr=pick['rowname']
    pick_loc=pick['locale']
    foo=extract_ts_nrm(pick.index[0])
    foo['loc']=pick_loc[pick.index[0]]
    
    for i in range(1,len(pick.index)):
        foo1=extract_ts_nrm(pick.index[i])
        foo1['loc']=pick_loc[pick.index[i]]
        foo=pd.concat([foo,foo1])
        
    foo['log(views)']=np.log(foo['views'])
    sns.lineplot(x='dates',y='log(views)',data=foo,hue='loc').set_title(art+'-'+acc+'-'+ag)
    return foo['loc'].unique()

plot_names_nrm("The_Beatles", "all-access", "all-agents")

#These are the tools we need for a visual examinination of arbitrary individual time series data. In the following, we will use 
#them to illustrate specific observations that are of particular interest.

#------------------------------------------------------------------------------------------------------------------------------------
#Summary parameter extraction

#In the next step we will have a more global look at the population parameters of our training time series data. Also here, we 
#will start with the wikipedia data. The idea behind this approach is to probe the parameter space of the time series information 
#along certain key metrics and to identify extreme observations that could break our forecasting strategies.  
    
#Projects data overview
#Before diving into the time series data let’s have a look how the different meta-parameters are distributed:

sns.countplot('agent',data=tpages)
plt.show()

sns.countplot('access',data=tpages)
plt.show()

chart1=sns.countplot('locale',data=tpages)
chart1.set_xticklabels(chart1.get_xticklabels(), rotation=45)
plt.show()

#We find that our wikipedia data includes 7 languages: German, English, Spanish, French, Japanese, Russian, and Chinese. 
#All of those are more frequent than the mediawiki and wikimedia pages. Mobile sites are slightly more frequent than desktop ones.

#-----------------------------------------------------------------------------
#Basic time series parameters
#We start with a basic set of parameters: mean, standard deviation, amplitude, and a the slope of a naive linear fit. This is our extraction function:

def params_ts1(rownr):
    foo=tdates.iloc[rownr,:]
    foo=foo.reset_index()
    foo=foo.rename(columns={"index": "dates", rownr: "views"})
    foo['dates']= pd.to_datetime(foo['dates'],format='%Y-%m-%d')
    
    foo['1']=foo['views']
    foo=foo[['dates','1','views']]
    
    #foo['views']=foo['views'].fillna(0)
    
    min_view=np.min(foo['views'])
    max_view=np.max(foo['views'])
    mean_view=np.mean(foo['views'])
    med_view = np.median(foo['views'])
    sd_view = statistics.stdev(foo['views'])
    
    dd_vv=copy.deepcopy(foo)
    dd_vv['views']=dd_vv['views'].fillna(0)
    dd=pd.DataFrame(dd_vv['dates'])
    vv=pd.DataFrame(dd_vv['views'])
    
    sd_error=np.array(standard_error_coef_linear_reg(dd,vv))
    slope=sd_error[0][1][0]
    
    data = {'rowname':rownr,'min_view':[min_view], 'max_view':[max_view],'mean_view':[mean_view],'med_view':[med_view],'sd_view':[sd_view],'slope':[slope]} 
    # Create DataFrame 
    df = pd.DataFrame(data)
    ind=pd.Series([rownr])
    df.set_index(ind,inplace=True)
    return df

#And here we run it. (Note, that in this kernel version I’m currently using a sub-sample of the data for reasons of runtime. 
#My extractor function is not very elegant, yet, and exceeds the kernel runtime for the complete data set.)
tpages.index
foo=copy.deepcopy(tpages)
a3=params_ts1(3333)
for i in foo['rowname']:
    print(i)
    a4=params_ts1(i)
    a3=pd.concat([a3,a4])


params_all=copy.deepcopy(a3)

params_all = params_all.loc[~params_all.index.duplicated(keep='last')]

params=copy.deepcopy(params_all)
params.describe()

#-----------------------------------------------------------------------------
#removing values (important section)
# droppping all the rows related to articles which has no views ever
article_allZeroView_index=params[params['mean_view']==0].index
article_allZeroView_rowname=params[params['mean_view']==0][['rowname']]

params.drop(labels=None, axis=0, index=article_allZeroView_index, columns=None, level=None, inplace=True, errors='raise')

params[(params['mean_view']<0)|(params['sd_view']<0)|(params['min_view']<0)|(params['max_view']<0)|(params['med_view']<0)]
# so no negative value

params[params['sd_view']==0].index
params[params['max_view']==0].index

missing_value(params)

mean_view_missing_index=params[params['mean_view'].isnull()].index
params.drop(labels=None, axis=0, index=mean_view_missing_index, columns=None, level=None, inplace=True, errors='raise')

params.reset_index(drop=True,inplace=True)
# missing value row/wesite wise
miss_val_tdates=missing_value_rowWise(tdates)
#-----------------------------------------------------------------------------
# printing the start and end dates of an article

sd_view_miss_rowname=params[params['sd_view'].isnull()]['rowname']

article_startDate=[]
rown_1=[]
for j in range(tdates.shape[0]):
    for i in range(0,tdates.shape[1]):
        if (math.isnan(tdates.iloc[j,i])==False):
            rown_1.append(j)
            article_startDate.append(tdates.columns[i])
            break


article_endDate=[]
rown_2=[]
for j in range(tdates.shape[0]):
    for i in reversed(range(tdates.shape[1])):
        if (math.isnan(tdates.iloc[j,i])==False):
            rown_2.append(j)
            article_endDate.append(tdates.columns[i])
            break


data={'rowname':rown_1,'article_startDate':article_startDate,'article_endDate':article_endDate}
article_start_end_Date=pd.DataFrame(data)

# exporting article_start_end_Date into csv
article_start_end_Date.to_csv(r'/Users/amanprasad/Documents/Kaggle/Web_Traffic_Time_Series_Forecasting/Files_WebViews/article_start_end_Date.csv', index = False)

#------------------------------------------------------------------------------------------------------------------------------------
#Overview visualisations
#Let’s explore the parameter space we’ve built. (The global shape of the distributions should not be affected by the sampling.) 
#First we plot the histograms of our main parameters:

# removing any row which contains null
#params=params.dropna()
#params.isnull().sum()

ax=sns.distplot(np.log(params['mean_view']),bins=100)
ax.set(title='mean_view distribution plot')
plt.show()

ax=sns.distplot(np.log(params['max_view']),bins=100)
ax.set(title='max_view distribution plot')
plt.show()

#--------------------------------
#Coefficient of Variation (CV)
#If you know nothing about the data other than the mean, one way to interpret the relative magnitude of the standard deviation 
#is to divide it by the mean. This is called the coefficient of variation. For example, if the mean is 80 and standard deviation 
#is 12, the cv = 12/80 = .15 or 15%.
#If the standard deviation is .20 and the mean is .50, then the cv = .20/.50 = .4 or 40%. So knowing nothing else about the data, 
#the CV helps us see that even a lower standard deviation doesn't mean less variable data.
#Hence it helps a lot with understanding relative variability.

ax=sns.distplot(np.log(params['sd_view']/params['mean_view']),bins=100)
ax.set(title='sd_view/mean_view distribution plot = Coefficient of Variation (CV)')
plt.show()
#---------------------------------

ax=sns.distplot((params['slope']),bins=100)
ax.set(title='slope distribution plot')
plt.show()

#We find:
#The distribution of average views is clearly bimodal, with peaks around 3 and 8 views. Something similar is true for 
#the number of maximum views.

#The distribution of standard deviations (divided by the mean) is skewed toward higher values with larger numbers of spikes or 
#stronger variability trends. Those will be the observations that are more challenging to forecast.

#The slope distribution is resonably symmetric and centred notably above zero.

#-----------------------------------------------------------------------------
#Let’s split it up by locale and focus on the densities:
params1=copy.deepcopy(params)

par_page = pd.merge(params1,tpages,on='rowname', how='left')
par_page['log_mean_view']=np.log(par_page['mean_view'])
par_page['log_max_view']=np.log(par_page['max_view'])
par_page['log_sd_view']=np.log(par_page['sd_view'])


# function for kde plot with hue
def kde_plot_with_hue(dataframe,column_to_plot,hue):
    #return dataframe[column_to_plot]
    gr = dataframe.groupby(hue)[column_to_plot]
    for label, arr in gr:
        #sns.set(color_codes=True)
        a=sns.kdeplot(arr, label=label, shade=True)
    a.set(title='KDE plot: {v}'.format(v=column_to_plot),xlabel=column_to_plot,ylabel='density')
    
kde_plot_with_hue(par_page,'log_mean_view','locale')
kde_plot_with_hue(par_page,'log_max_view','locale')
kde_plot_with_hue(par_page,'log_sd_view','locale')

#We find:
#The chinese pages (zh) are slightly but notably different from the rest. The have lower mean and max views and also 
#less variation. Their slope distribution is broader, but also shifted more towards positive values compared to the other curves.

#The peak in max views in earlier stage is most pronounced in the french pages (fr).

#The english pages (en) have the highest mean and maximum views, which is not surprising.

#-----------------------------------------------------------------------------
#Next, we will examine binned 2-d histograms.

above_max=par_page['max_view']-par_page['mean_view']
plt.scatter(np.log(above_max),par_page['log_mean_view'])
plt.xlabel("max views above mean")
plt.ylabel("mean views")
plt.show()

#We find:
#There is a clear correlation between mean views and maximum views. Also here we find again the two cluster peaks we had 
#identified in the individual histograms. A couple of outliers and outlier groups are noticeable.

#Let’s zoom into the upper right corner (the numbers in parentheses are the row numbers):
'''
plt.scatter(par_page['above_max'],par_page['log_mean_view'])

for line in range(0,par_page.shape[0]):
     plt.text(par_page.above_max[line], par_page.log_mean_view[line], par_page.article[line])

plt.xlim([13, 15])
plt.ylim([11, 15])
plt.show()
'''

#Another question: Does the (assumed) linear change in views depend on the total number of views?

plt.scatter(par_page['slope'],np.log(par_page['mean_view']))
plt.xlabel("linear slope relative to slope error")
plt.ylabel("log(mean_views)")
plt.show()

#We find that articles with higher average view-count have more variability in their linear trends.
#However, this might be due to our slope normalisation which will decrease the effective slope for low view counts. 
#It should not, however, affect the observation that the slopes of low-view articles are on average slightly higher than those 
#of high-view articles. Such an effect could be caused by viewing spikes, of course, but I would expect those to be randomly 
#distributed.

#Note: here slope= coeff/(standard Error) which is basically t statistics
#If 95% of the t distribution is closer to the mean than the t-value on the coefficient you are looking at, then you have 
#a P value of 5%. This is also reffered to a significance level of 5%. The P value is the probability of seeing a result as extreme 
#as the one you are getting (a t value as large as yours) in a collection of random data in which the variable had no effect. 
#A P of 5% or less is the generally accepted point at which to reject the null hypothesis. With a P value of 5% (or .05) there 
#is only a 5% chance that results you are seeing would have come up in a random distribution, so you can say with a 95% probability 
#of being correct that the variable is having some effect, assuming your model is specified correctly.

#------------------------------------------------------------------------------------------------------------------------------------
#Individual observations with extreme parameters

#Based on the overview parameters we can focus our attention on those articles for which the time series parameters are at the 
#extremes of the parameter space.

#Large linear slope

#Those are the observations with the highest slope values. (In the sample this will be different, but in the full wikipedia 
#data set the top 10 have rownames 91728, 55587, 108341, 70772, 95367, 18357, 95229, 116150, 94975, 77292).
slope_sort=params1.sort_values(by=['slope'],ascending=False).head()

#Let’s have a look at the time series data of the top 4 articles:
plot_rownr(91727)
plot_rownr(55586)
plot_rownr(108340)
plot_rownr(70771)

#We find:

#Lot’s of love for Twenty One Pilots in Spain. Those rapid rises and wibbly-wobbly bits are going to be difficult to predict, 
#unless there’s a periodic modulation on top of the large-scale trend. Certaintly worth figuring out.

#We also see that our smoother is dealing rather well with most of the slower variability patterns and could be used to remove 
#the low-frequency structures for further analysis.

#-----------------------------------------------------------------------------
#Let’s compare the interest in Twenty One Pilots for the different countries, to see whether a prediction for one of them could 
#learn from the others:
plot_names_nrm("Twenty_One_Pilots", "all-access", "all-agents")

#Note, that those curves are normalised to mean views (each) and have a logarithmic y-axis to mitigate the effect of large spikes. 
#This chart is for relative trend comparison.

#We find:
#Germany and France show quite similar viewing behaviour. The English pages show less dramatic changes but end up

#With a purely time-series-forecast approach I think that the large spikes are close to impossible to predict. However, external 
#data could help a lot here.

#-----------------------------------------------------------------------------
#Those viewing numbers were going up, but which articles were going down?
slope_sort=params1.sort_values(by=['slope'],ascending=False).tail()

plot_rownr(95855)
plot_rownr(74114)
plot_rownr(8387)
plot_rownr(103658)

#The main page itself on mobile, and review articles on 2015 were the biggest losers.

#------------------------------------------------------------------------------------------------------------------------------------
#High standard deviations
#The top 10 wikipedia rows are 9774, 38573, 103123, 99322, 74114, 39180, 10403, 33644, 34257, and 26993. Bingo, anyone?

params1.sort_values(by=['sd_view'],ascending=False).head()

plot_rownr(9774)
plot_rownr(38573)
plot_rownr(103123)
plot_rownr(99322)

#Those are pretty strong spikes in the main page views, even if the baseline is around 1-10 million to begin with. They look 
#consistent though over different languages. Any ideas what could cause this?

#If we normalise standard deviation by mean we get a different set of results:
params_sd=copy.deepcopy(params1)
params_sd['sd_view/mean_view']=params1['sd_view']/params1['mean_view']
sd_sort=params_sd.sort_values(by=['sd_view/mean_view'],ascending=False).head()

plot_rownr(10031)
plot_rownr(38811)
plot_rownr(86904)
plot_rownr(102520)

#Those are very, very suspicious. They are essentially low baselines with single dates that have way higher view counts 
#These have to be errors in the data which can be dangerous for predictions 
#if they appear close to either end of the date window. In other cases, most smoothing methods should be able to deal with them.

#------------------------------------------------------------------------------------------------------------------------------------
#Large variability amplitudes
#The top amplitudes are the same as the top standard deviations, due to the spikey nature of the variability:
params_sd['amplitude']=params1['max_view']-params1['mean_view']
sd_sort=params_sd.sort_values(by=['amplitude'],ascending=False).head()

#------------------------------------------------------------------------------------------------------------------------------------
#High average views
#Those are the time series of the most popular pages, which we already identified as the main pages in the plots above:
params_sd.sort_values(by=['mean_view'],ascending=False).head()
plot_rownr(38573)
plot_rownr(9774)
plot_rownr(74114)
plot_rownr(139119)

#In addition to the spikes on the english main page there is a suprising amount of variability as exemplified by the 
#long-term structure in the German main page.

# doubt from where these value came
#What about other main pages, as identified in the zoom-in above?
plot_rownr_log(92205)
plot_rownr(116196)
plot_rownr_log(10403)
plot_rownr_log(33644)

#Here 3 of the 4 plots have a logarithmic y-axis to improve the clarity of visualising the time series’ with strong spikes. 
#We see that also those popular pages exhibit strong variability on various time scales.
#--------------------------------
#In summary: We have identified the time series’ with the highest variability according to basic criteria. We also found a few 
#time series sets with bogus values. These are the data sets that might pose the greatest challenge to our prediction algorithms.
#--------------------------------

#------------------------------------------------------------------------------------------------------------------------------------
#Short-term variability
#Before turning to forecasting methods, let’s have a closer look at the characteristic short-term variability that has become 
#evident in several of the plots already. Below, we plot a 2-months zoom into the “quiet” parts (i.e. no strong spikes) of 
#different time series:
plot_rownr_zoom(10403, "2016-10-01", "2016-12-01")
plot_rownr_zoom(9774, "2015-09-01", "2015-11-01")
plot_rownr_zoom(139119, "2016-10-01", "2016-12-01")
plot_rownr_zoom(110657, "2016-07-01", "2016-09-01")

#We see that the high-view-count time series on the 1st and 3rd plot that show a very regular periodicity that is strikingly similar 
#for both of them. A similar structure can be seen on the 2nd and 4th plot, although here it is partly distorted by a slight 
#upward trend (2nd plot) and/or variance caused by lower viewing numbers (4th plot).

#These plots provide evidence that there is variability on a weekly scale. The next figure will visualise this weekly behaviour 
#in a different way:

# get days- enter the dataframe- there should be a column 'dates
def weekdayName_views(dataframe):
    for i in dataframe.index:
        d=dataframe.loc[i,['dates']][0]
        week_num = d.weekday()
        weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
        week_num_asName = weekDays[week_num]
        dataframe.loc[i,['dates']]=week_num_asName
    return dataframe

foo11=extract_ts(10404)
weekdayName_views(foo11)

def weekDay_views(rownr,startDate,endDate):
    foo1=extract_ts(rownr)
    foo1=foo1[(foo1['dates']>startDate)&(foo1['dates']<endDate)]
    weekdayName_views(foo1)
    foo2=foo1.groupby(['dates']).mean()
    foo2['wday_views']=foo2['views']/np.mean(foo2['views'])
    foo2['id']=rownr
    foo2.drop(['views'],axis=1,inplace=True)
    foo2.reset_index(inplace=True)
    foo2['dates']=foo2['dates'].astype('category')
    foo2['dates']=foo2['dates'].cat.reorder_categories(["Monday", "Tuesday", "Wednesday","Thursday","Friday","Saturday","Sunday"], ordered=True)
    foo2=foo2.sort_values(by=['dates'])
    foo2.reset_index(drop=True,inplace=True)    
    return foo2

foo1=weekDay_views(10404,"2016-10-01","2016-12-01")
foo2=weekDay_views(9775, "2015-09-01","2015-11-01")
foo3=weekDay_views(139120, "2016-10-01","2016-12-01")
foo4=weekDay_views(110658,"2016-07-01","2016-09-01")

foo_all=[foo1,foo2,foo3,foo4]
foo=pd.concat(foo_all)
foo.reset_index(drop=True,inplace=True)
foo['id']=foo['id'].astype('category')

ax=sns.scatterplot(x="dates", y="wday_views", hue="id",data=foo)
ax.set(xlabel='Day Of Week',ylabel='Relative Average View')

#Here we average the variability in the previous plot over the day of the week and then overlay all four time series with 
#different colours on a relative scale. We see the clear trend toward lower viewing numbers on the weekend (Fri/Sat/Sun), and 
#also a declining trend as we approach the weekend. This gives us valuable information on the general type of variability 
#over the course of a week. In order to study this behaviour more in detail, we would need to average over a larger number of 
#time series.




























