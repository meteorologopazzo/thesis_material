def linregress_3D(y_array):
    import numpy as np
    # y_array is a 3-D array formatted like (time,lon,lat)
    # The purpose of this function is to do linear regression using time series of data over each (lon,lat) grid box with consideration of ignoring np.nan
    # Construct x_array indicating time indexes of y_array, namely the independent variable.
    x_array=np.empty(y_array.shape)
    for i in range(y_array.shape[0]): x_array[i,:,:]=i+1 # This would be fine if time series is not too long. Or we can use i+yr (e.g. 2019).
    x_array[np.isnan(y_array)]=np.nan
    
    # Compute the number of non-nan over each (lon,lat) grid box.
    n=np.sum(~np.isnan(x_array),axis=0)
    
    # Compute mean and standard deviation of time series of x_array and y_array over each (lon,lat) grid box.
    x_mean=np.nanmean(x_array,axis=0)
    y_mean=np.nanmean(y_array,axis=0)
    x_std=np.nanstd(x_array,axis=0)
    y_std=np.nanstd(y_array,axis=0)
    
    # Compute co-variance between time series of x_array and y_array over each (lon,lat) grid box.
    cov=np.nansum((x_array-x_mean)*(y_array-y_mean),axis=0)/n
    
    # Compute correlation coefficients between time series of x_array and y_array over each (lon,lat) grid box.
    cor=cov/(x_std*y_std)
    
    # Compute slope between time series of x_array and y_array over each (lon,lat) grid box.
    slope=cov/(x_std**2)
    
    # Compute intercept between time series of x_array and y_array over each (lon,lat) grid box.
    intercept=y_mean-x_mean*slope
    
    # Compute tstats, stderr, and p_val between time series of x_array and y_array over each (lon,lat) grid box.
    df = n-2
    t_value = np.abs(cor)*np.sqrt((df)/(1-corr_coeff**2))
    p_val = 2*(1 - stats.t.cdf(t_value,df=df))
    
    
    
    ####### OLD FORMULATION FOR T AND P VALUES
    #tstats=cor*np.sqrt(n-2)/np.sqrt(1-cor**2)
    stderr=slope/t_value   #tstats
    #from scipy.stats import t
    #p_val=t.sf(tstats,n-2)*2
    
    # Compute r_square and rmse between time series of x_array and y_array over each (lon,lat) grid box.
    # r_square also equals to cor**2 in 1-variable lineare regression analysis, which can be used for checking.
    r_square=np.nansum((slope*x_array+intercept-y_mean)**2,axis=0)/np.nansum((y_array-y_mean)**2,axis=0)
    rmse=np.sqrt(np.nansum((y_array-slope*x_array-intercept)**2,axis=0)/n)
    
    # Do further filteration if needed (e.g. We stipulate at least 3 data records are needed to do regression analysis) and return values
    n=n*1.0 # convert n from integer to float to enable later use of np.nan
    n[n<3]=np.nan
    slope[np.isnan(n)]=np.nan
    intercept[np.isnan(n)]=np.nan
    p_val[np.isnan(n)]=np.nan
    r_square[np.isnan(n)]=np.nan
    rmse[np.isnan(n)]=np.nan
    
    return n,slope,intercept,p_val,r_square,rmse








    ##### I WANT TO PROVIDE THE XARRAY MYSELF

def MY_point_regr(x_array,y_array):
    import numpy as np
    from scipy import stats
    
    # y_array is a 3-D array formatted like (time,lat,lon)
    # The purpose of this function is to do linear regression using time series of data over each (lon,lat) grid box with consideration of ignoring np.nan
    # Construct x_array indicating time indexes of y_array, namely the independent variable.
    #x_array=np.empty(y_array.shape)
    #for i in range(y_array.shape[0]): x_array[i,:,:]=i+1 # This would be fine if time series is not too long. Or we can use i+yr (e.g. 2019).
    #x_array[np.isnan(y_array)]=np.nan
    
    # Compute the number of non-nan over each (lat,lon) grid box.
    n=np.sum(~np.isnan(x_array),axis=0)
    
    # Compute mean and standard deviation of time series of x_array and y_array over each (lon,lat) grid box.
    x_mean=np.nanmean(x_array,axis=0)
    y_mean=np.nanmean(y_array,axis=0)
    x_std=np.nanstd(x_array,axis=0)
    y_std=np.nanstd(y_array,axis=0)
    
    # Compute co-variance between time series of x_array and y_array over each (lon,lat) grid box.
    cov=np.nansum((x_array-x_mean)*(y_array-y_mean),axis=0)/n
    
    # Compute correlation coefficients between time series of x_array and y_array over each (lon,lat) grid box.
    # IN FUNC_STATISTICS.PY I used this to compute correlation
    # the two methods return significantly different results
    # so much so that one point is considered pval<5% with one
    # and pval>5% with the other method
    # corr_coeff, trash = stats.spearmanr(x,y)
    # PEARSON COEFFICIENT
    cor=cov/(x_std*y_std)
    
    # SPEARMAN RANKS
    corr_spearman = np.zeros((546, 573))
    
    for i in range(546):
        for j in range(573):
            corr_spearman[i,j], trash = stats.spearmanr(x_array[:,i,j], y_array[:,i,j], axis=0)
    
    # Compute slope between time series of x_array and y_array over each (lon,lat) grid box.
    slope=cov/(x_std**2)
    
    # Compute intercept between time series of x_array and y_array over each (lon,lat) grid box.
    intercept=y_mean-x_mean*slope
    
    # Compute tstats, stderr, and p_val between time series of x_array and y_array over each (lon,lat) grid box.
    df = n-2
    t_value = np.abs(cor)*np.sqrt( (df)/(1-cor**2) )
    p_val = 2*(1 - stats.t.cdf(t_value,df=df))
    
    
    # compute pval with spearman correlation
    t_spear = np.abs(corr_spearman)*np.sqrt( (df)/(1-corr_spearman**2) )
    p_spear = 2*(1 - stats.t.cdf(t_spear,df=df))
    
    
    ############## OLD FORMULATION FOR T AND P VALUES
    #tstats=cor*np.sqrt(n-2)/np.sqrt(1-cor**2)
    stderr=slope/t_value   #tstats
    #from scipy.stats import t
    #p_val=t.sf(tstats,n-2)*2
    
    ##################
    
    # Compute r_square and rmse between time series of x_array and y_array over each (lon,lat) grid box.
    # r_square also equals to cor**2 in 1-variable lineare regression analysis, which can be used for checking.
    r_square=np.nansum((slope*x_array+intercept-y_mean)**2,axis=0)/np.nansum((y_array-y_mean)**2,axis=0)
    rmse=np.sqrt(np.nansum((y_array-slope*x_array-intercept)**2,axis=0)/n)
    
    # Do further filteration if needed (e.g. We stipulate at least 3 data records are needed to do regression analysis) and return values
    n=n*1.0 # convert n from integer to float to enable later use of np.nan
    n[n<3]=np.nan
    
    slope[np.isnan(n)]=np.nan
    intercept[np.isnan(n)]=np.nan
    p_val[np.isnan(n)]=np.nan
    r_square[np.isnan(n)]=np.nan
    rmse[np.isnan(n)]=np.nan
    
    
    
    return n, slope, intercept, cor, corr_spearman, p_val, p_spear, r_square, rmse
