# missing data count column wise
def null_val_desc(dataframe_name):
    null_all=dataframe_name.isnull().sum()
    null_all_desc=null_all[null_all>0].sort_values(ascending = False) 
    null_all_desc=pd.DataFrame(null_all_desc)
    null_all_desc = null_all_desc.reset_index()
    null_all_desc.columns = ['column', 'missing_count']
    return null_all_desc

#-------------------------------------------------------------------------------------------------------------
def standard_error_coef_linear_reg(X,y):
    import pandas as pd
    import numpy as np
    from sklearn import linear_model
    model = linear_model.LinearRegression()
    model.fit(X=X, y=y)
    
    #print("from scikit-learn")
    #print('Intercept = {I}'.format(I=model.intercept_))
    #print('Model Coefficient = {C}'.format(C=model.coef_))

    N = len(X)
    p = len(X.columns) + 1
    
    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = X.values
    
    beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y.values
    #print('Standard Error Intercept(1st) and Coefficients = {S}'.format(S=beta_hat))
    return [beta_hat,model.coef_]

#-------------------------------------------------------------------------------------------------------------
def kde_plot_with_hue(dataframe,column_to_plot,hue):
    #return dataframe[column_to_plot]
    gr = dataframe.groupby(hue)[column_to_plot]
    for label, arr in gr:
        a=sns.kdeplot(arr, label=label, shade=True)
    a.set(title='KDE plot: {v}'.format(v=column_to_plot),xlabel=column_to_plot,ylabel='density')
    


#-------------------------------------------------------------------------------------------------------------
# find missing values in columns

def missing_value(dataFrame):
    null_all=dataFrame.isnull().sum()
    
    null_all_desc=null_all[null_all>0].sort_values(ascending = False) 
    
    null_all_desc=pd.DataFrame(null_all_desc)
    null_all_desc = null_all_desc.reset_index()
    null_all_desc.columns = ['column', 'missing_count']
    return (null_all_desc)

#-------------------------------------------------------------------------------------------------------------
def missing_value_rowWise(dataframe):
    miss=[]
    rowname=[]
    for i in range(len(dataframe.index)) :
        miss.append(dataframe.iloc[i].isnull().sum())
        rowname.append(i)
    miss=np.array(miss)   
    data = {'rowname':rowname, 'count_missing_val':miss}
    # Create DataFrame 
    df = pd.DataFrame(data)
    return df
        
#-------------------------------------------------------------------------------------------------------------
# get days- enter the dataframe- there should be a column 'dates
def weekdayName_views(dataframe):
    for i in dataframe.index:
        d=dataframe.loc[i,['dates']][0]
        week_num = d.weekday()
        weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
        week_num_asName = weekDays[week_num]
        dataframe.loc[i,['dates']]=week_num_asName
    return dataframe

    

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

        
        
        
        
        
        
