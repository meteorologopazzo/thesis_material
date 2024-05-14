##### Percentile distribution - it returns the p-value of each bin
# 2 sided p-value = 2*(1-cdf(|tvalue|, dof)
def perc_distribution_pvalue_dof(control, variable, control_sub, var_sub, nbins, perc_step, popmean):
    
    from scipy import stats
    import numpy as np
    
    # memory alloc
    distribution_control = np.zeros(nbins)
    distribution = np.zeros(nbins)
    
    std_distribution = np.zeros(nbins)
    std_err_distribution = np.zeros(nbins)
    
    number_of_points = np.zeros(nbins)
    number_of_points_sub = np.zeros(nbins)
    
    percentiles = np.zeros(nbins+1)
    percentiles_sub = np.zeros(nbins+1)
    p_value = np.zeros(nbins)

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step)
        
        lower = np.percentile(control[~np.isnan(control)],pp)
        upper = np.percentile(control[~np.isnan(control)],pp+perc_step)
        percentiles[qq] = lower
        
        #  SUBSAMPLED STATISTICS
        lower_sub = np.percentile(control_sub[~np.isnan(control_sub)],pp)
        upper_sub = np.percentile(control_sub[~np.isnan(control_sub)],pp+perc_step)
        percentiles_sub[qq] = lower_sub
        
        
        # stats computation
        cond_mean_control = np.nanmean(control[(control>=lower)&(control<upper)])
        cond_mean = np.nanmean(variable[(control>=lower)&(control<upper)])
        cond_std = np.nanstd(variable[(control>=lower)&(control<upper)])
        
        distribution_control[qq] = cond_mean_control 
        distribution[qq] = cond_mean                          
        std_distribution[qq] = cond_std
                 
        number_of_points[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        number_of_points_sub[qq] = np.sum(~np.isnan(var_sub[(control_sub>=lower_sub)&(control_sub<upper_sub)]))
        
        std_err_distribution[qq] = std_distribution[qq]/np.sqrt(number_of_points_sub[qq])
        
        #################      CORRECT VERSION   ###################
        dof = number_of_points[qq]
        t_stat = (distribution[qq] - popmean)/std_err_distribution[qq]    #(std_distribution[qq]/np.sqrt(dof))
        p_value[qq] = 2*(1 - stats.t.cdf(np.abs(t_stat), df=dof))
        
    return distribution_control, distribution, std_distribution, std_err_distribution, number_of_points, p_value

####################################################################################




def fb_distribution_npoint_pvalue(control, variable, control_sub, var_sub, nbins, perc_step, popmean):
    
    from scipy import stats
    import numpy as np
    
    distribution_control_fb = np.zeros(nbins)
    distribution_fb = np.zeros(nbins)
    std_distribution_fb = np.zeros(nbins)
    std_err_distribution_fb = np.zeros(nbins)
    
    number_of_points_fb = np.zeros(nbins)
    number_of_points_fb_sub = np.zeros(nbins)
    
    bin_edges_fb = np.zeros(nbins+1)
    bin_edges_sub = np.zeros(nbins+1)
    p_value = np.zeros(nbins)
    
    # Bin width
    bw = (np.max(control[~np.isnan(control)])-np.min(control[~np.isnan(control)]))/nbins 
    
    # SUBSAMPLED  Bin width
    bw_sub = (np.max(control_sub[~np.isnan(control_sub)])-np.min(control_sub[~np.isnan(control_sub)]))/nbins 

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step) # Index starting from 0.
        
        lower = np.min(control[~np.isnan(control)])+qq*bw
        upper = lower + bw
        bin_edges_fb[qq] = lower
        
        # SUBSAMPLED STATISTICS
        lower_sub = np.min(control[~np.isnan(control)])+qq*bw_sub
        upper = lower_sub + bw_sub
        bin_edges_sub[qq] = lower_sub
        
        distribution_control_fb[qq] = np.nanmean(control[(control>=lower)&(control<upper)])
        distribution_fb[qq] = np.nanmean(variable[(control>=lower)&(control<upper)])
        std_distribution_fb[qq] = np.nanstd(variable[(control>=lower)&(control<upper)])
        
        number_of_points_fb[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        number_of_points_fb_sub[qq] = np.sum(~np.isnan(var_sub[(control_sub>=lower_sub)&(control_sub<upper_sub)]))
        
        std_err_distribution_fb[qq] = std_distribution_fb[qq]/np.sqrt(number_of_points_fb_sub[qq])
        
        #################   PVALUE   :   CORRECT VERSION   ###################
        dof = number_of_points_fb_sub[qq]
        t_stat = (distribution_fb[qq] - popmean)/std_err_distribution_fb[qq]     #(std_distribution_fb[qq]/np.sqrt(dof))
        p_value[qq] = 2*(1 - stats.t.cdf(np.abs(t_stat), df=dof))
        
        
    bin_edges_fb[-1] = upper
    bin_centers_fb = 0.5*(bin_edges_fb[1:]+bin_edges_fb[:-1])
    
    return distribution_control_fb, distribution_fb, std_distribution_fb, std_err_distribution_fb, number_of_points_fb, p_value

