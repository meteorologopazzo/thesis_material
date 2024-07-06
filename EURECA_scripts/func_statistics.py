import numpy as np
###############################
def rmse(x_pre,x_ref):
    return np.sqrt(np.nanmean((x_pre - x_ref)**2))

###############################
def bias(x_pre,x_ref):
    return np.nanmean(x_pre - x_ref)

###############################
# function to compute pdf 
def compute_pdf(xa):
    x = np.double(xa)
    x = x.flatten()
    #x = xa.values.flatten()
    x = x[~np.isnan(x)]
    x = x[~np.isinf(x)]
    nbin = np.sqrt(x.size).astype(int)
    pdf, xbin = np.histogram(x, bins=nbin, density=True)
    cbin = 0.5*(xbin[1:]+xbin[:-1])
    return cbin, pdf

###############################
# function to compute pdf 
def compute_pdf_nbin(xa, nbin):
    x = np.double(xa)
    x = x.flatten()
    x = x[~np.isnan(x)]; x = x[~np.isinf(x)]
    pdf, xbin = np.histogram(x, bins=nbin, density=True)
    cbin = 0.5*(xbin[1:]+xbin[:-1])
    return cbin, pdf

###############################
# function to compute linear regression, correlation coeff and p value
# TO BE USED WHEN FITTING PERCENTILES!
def slopes_r_p(x,y,std_y=None):
    from scipy import stats
    import numpy as np
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    linreg = stats.linregress(x,y)
    corr_coeff, trash = stats.spearmanr(x,y)
    df = np.size(x)-2   # degrees of freedom.
    t_value = np.abs(corr_coeff)*np.sqrt((df)/(1-corr_coeff**2))
    p_value = 2*(1 - stats.t.cdf(t_value,df=df))
    
    
    if std_y is not None:
        chisq = np.sum(  (y-linreg.slope*x-linreg.intercept)**2 / std_y**2 )
        chisq_cumul_right = 1. - stats.chi2.cdf(chisq, df=df)
        chisq_rid = chisq/df
    else:
        chisq, chisq_cumul_right, chisq_rid = 999., 999., 999.
        
        
    chi_square = {'chi2': chisq, 'chi2_cumulated': chisq_cumul_right, 'chi2_reduced': chisq_rid}
        
        
    return linreg, corr_coeff, p_value, chi_square


########### APPLY SUBSAMPLING FOR COMPARISON and draw statistics only on subsampled data
def slopes_r_p_onlysub(x, y, nt, nskip):
    from scipy import stats
    import numpy as np
    
    x = x[::nt,::nskip,::nskip]
    y = y[::nt,::nskip,::nskip]
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    linreg = stats.linregress(x,y)
    corr_coeff, trash = stats.spearmanr(x,y)
    
    df = np.size(x)-2
    mean_x = np.mean(x);  mean_x2 = np.mean(x**2); lever_arm = mean_x2-mean_x**2
    
    sigma_y = np.sqrt( np.sum(  (y-linreg.slope*x-linreg.intercept)**2 )/df  )
    sigma_slope = sigma_y/( np.sqrt(np.size(xx)*(lever_arm) ) )
    sigma_intercept = sigma_y*np.sqrt(mean_x2/( np.size(xx)*(lever_arm) ))
    
    sigmas = (sigma_slope, sigma_intercept)
    
    t_value_cannelli = linreg.slope/sigma_slope     # SOMETHING MISSING?
    p_value_cannelli = 2*(1 - stats.t.cdf(t_value_cannelli,df=df))

    t_value = np.abs(corr_coeff)*np.sqrt((df)/(1-corr_coeff**2))
    p_value = 2*(1 - stats.t.cdf(t_value,df=df))
    
    return linreg, corr_coeff, p_value, p_value_cannelli, sigmas


########### APPLY SUBSAMPLING FOR COMPARISON only when assessing fit quality
def slopes_r_p_mix(x, y, nt, nskip):
    from scipy import stats
    import numpy as np
    
    xx = x[::nt,::nskip,::nskip]
    yy = y[::nt,::nskip,::nskip]
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    xx = xx[~np.isnan(xx)]
    yy = yy[~np.isnan(yy)]
    
    linreg = stats.linregress(x,y)
    corr_coeff, trash = stats.spearmanr(x,y)
    
    df = np.size(xx)-2
    mean_x = np.mean(x);  mean_x2 = np.mean(x**2); lever_arm = mean_x2-mean_x**2
    
    sigma_y = np.sqrt( np.sum(  (y-linreg.slope*x-linreg.intercept)**2 )/df  )
    sigma_slope = sigma_y/( np.sqrt(np.size(xx)*(lever_arm) ) )
    sigma_intercept = sigma_y*np.sqrt(mean_x2/( np.size(xx)*(lever_arm) ))
    
    sigmas = (sigma_slope, sigma_intercept)
    
    t_value_cannelli = linreg.slope/sigma_slope     # SOMETHING MISSING?
    p_value_cannelli = 2*(1 - stats.t.cdf(t_value_cannelli,df=df))
    
    # to ADD: scipy.stats.chisquare(f_obs, f_exp=None) --> not working as expected
    # chisq = np.sum(  (y-linreg.slope*x-linreg.intercept)**2 / std_y**2 )
    # in realtà mi servirebbero le dev std delle diverse osservazioni y --> calcolo X2 solo per i percentili e sto contento
    

    t_value = np.abs(corr_coeff)*np.sqrt((df)/(1-corr_coeff**2))
    p_value = 2*(1 - stats.t.cdf(t_value,df=df))
    
    return linreg, corr_coeff, p_value, p_value_cannelli, sigmas



##### POINT-WISE REGRESSION OF Y WRT X
# MISSING: subsampling considering time corrrelations
def point_regression(x,y):
    
    # x,y formatted as : time-lat-lon
    
    import numpy as np
    from scipy import stats
    
    #xx = x[::nt,::nskip,::nskip]
    #yy = y[::nt,::nskip,::nskip]
    
    #x = x[~np.isnan(x)]
    #y = y[~np.isnan(y)]
    
    #xx = xx[~np.isnan(xx)]
    #yy = yy[~np.isnan(yy)]
    
    linreg = stats.linregress(x,y)
    corr_coeff, trash = stats.spearmanr(x,y, axis=0)
    
    df = np.size(xx)-2
    mean_x = np.mean(x);  mean_x2 = np.mean(x**2); lever_arm = mean_x2-mean_x**2
    
    sigma_y = np.sqrt( np.sum(  (y-linreg.slope*x-linreg.intercept)**2 )/df  )
    sigma_slope = sigma_y/( np.sqrt(np.size(xx)*(lever_arm) ) )
    sigma_intercept = sigma_y*np.sqrt(mean_x2/( np.size(xx)*(lever_arm) ))
    
    sigmas = (sigma_slope, sigma_intercept)
    
    t_value_cannelli = linreg.slope/sigma_slope     # SOMETHING MISSING?
    p_value_cannelli = 2*(1 - stats.t.cdf(t_value_cannelli,df=df))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


###############################
# compute slope, R, pvalue, RMSE, bias for each grid point 
def metrics_maps(a,b):
    from scipy import stats
    dims = (b.shape[1],b.shape[2])
    slope = np.zeros(dims); r = np.zeros(dims); p = np.zeros(dims)
    bias = np.zeros(dims); rmse = np.zeros(dims)
    for i in range(0,b.shape[1]):
        for j in range(0,b.shape[2]):
            if all(np.isnan(a[:,i,j])) or all(np.isnan(b[:,i,j])):
                slope[i,j] = np.nan; r[i,j] = np.nan; p[i,j] = np.nan
                bias[i,j] = np.nan; rmse[i,j] = np.nan
            else:
                x = a[:,i,j]; x = x[~np.isnan(x)]
                y = b[:,i,j]; y = y[~np.isnan(y)]
                linreg = stats.linregress(x,y)
                slope[i,j] = linreg.slope; r[i,j] = linreg.rvalue; p[i,j] = linreg.pvalue
                bias[i,j] = np.mean(y) - np.mean(x)
                rmse[i,j] = np.sqrt(np.mean((y-x)**2))
    return slope,r,p,bias,rmse

###############################
# compute slope, R, RMSE, bias for each time
def metrics_time(a,b):
    from scipy import stats
    slope = np.zeros(b.shape[0]); r = np.zeros(b.shape[0]); p = np.zeros(b.shape[0])
    bias = np.zeros(b.shape[0]); rmse = np.zeros(b.shape[0])
    for i in range(0,b.shape[0]):
            x = a[i]; x = x[~np.isnan(x)]
            y = b[i]; y = y[~np.isnan(y)]
            if (x.size == 0)|(y.size == 0):
                slope[i] = np.nan; r[i] = np.nan; p[i] = np.nan
                bias[i] = np.nan; rmse[i] = np.nan
            else:
                linreg = stats.linregress(x,y)
                slope[i] = linreg.slope; r[i] = linreg.rvalue; p[i] = linreg.pvalue
                bias[i] = np.mean(y) - np.mean(x)
                rmse[i] = np.sqrt(np.mean((y-x)**2))
    return slope,r,p,bias,rmse

###############################
# compute slope for each grid point 
def slope_maps(a,b):
    from scipy import stats
    dims = (b.shape[1],b.shape[2])
    slope = np.zeros(dims)
    for i in range(0,b.shape[1]):
        for j in range(0,b.shape[2]):
            if all(np.isnan(a[:,i,j])) or all(np.isnan(b[:,i,j])):
                slope[i,j] = np.nan
            else:
                x = a[:,i,j]; x = x[~np.isnan(x)]
                y = b[:,i,j]; y = y[~np.isnan(y)]
                linreg = stats.linregress(x,y)
                slope[i,j] = linreg.slope
    return slope

###############################
# compute slope for each time
def slope_time(a,b):
    from scipy import stats
    slope = np.zeros(b.shape[0])
    for i in range(0,b.shape[0]):
            x = a[i]; x = x[~np.isnan(x)]
            y = b[i]; y = y[~np.isnan(y)]
            if (x.size == 0)|(y.size == 0):
                slope[i] = np.nan
            else:
                linreg = stats.linregress(x,y)
                slope[i] = linreg.slope
    return slope
