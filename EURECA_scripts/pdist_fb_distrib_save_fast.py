# this is just to cut some code lines in the notebooks
# it only spits out data 

from plotdistr import perc_distribution_pvalue, fb_distribution_npoint_pvalue
from MY_plotdistr import perc_distribution_pvalue_dof, fb_distribution_npoint_pvalue_dof
#import pyinputplus as pyip

def distrib_2d(x, y, perc_step, nbins, popmean, perc_fixbin):
    #pyip.inputStr(allowRegexes=[r'perc', 'fixbin']
    allowed_perc = ['p', 'pdist', 'perc', 'percentile'] 
    allowed_fb = ['fb', 'fixbin', 'fixed_bin', 'bin']
    
    if perc_fixbin in allowed_perc:
        xx = x.copy(); control = xx.reshape(-1)
        yy = y.copy(); variable = yy.reshape(-1)

        ##### Perc bin distribution: pvalue
        distr_x, distr_y, std_y, stderr_y, npoints_y, pvalue_y = perc_distribution_pvalue(control, variable, nbins, perc_step, popmean)
        
    elif perc_fixbin in allowed_fb:
        xx = x.copy(); control = xx.reshape(-1)
        yy = y.copy(); variable = yy.reshape(-1)

        ##### Perc bin distribution: pvalue
        distr_x, distr_y, std_y, stderr_y, npoints_y, pvalue_y = fb_distribution_npoint_pvalue(control, variable, nbins, perc_step, popmean, dof=None)

    return distr_x, distr_y, std_y, stderr_y, npoints_y, pvalue_y

####################################################################à





def dist_3d_subsample(x,y,perc_step, nbins, popmean, nt, nttop, nskip, nskiptop, top, perc_fixbin):
    
    import numpy as np
    
    allowed_perc = ['p', 'pdist', 'perc', 'percentile'] 
    allowed_fb = ['fb', 'fixbin', 'fixed_bin', 'bin']
    
    dist_y = np.zeros((y.shape[1],nbins))
    std_y = np.zeros((y.shape[1],nbins))
    stderr_y = np.zeros((y.shape[1],nbins))
    pvalue_y_sub = np.zeros((y.shape[1],nbins))
    # pvalue_y = np.zeros_like(pvalue_y_sub)
    npoints_y = np.zeros_like(dist_y)
    #npoints_y_sub = np.zeros_like(npoints_y)
    
    if perc_fixbin in allowed_perc:
        for h in range(0,y.shape[1]):
            if h % 10 == 0:
                print(h)    
            xx = x.copy(); control = xx.reshape(-1)
            yy = y[:,h].copy(); variable = yy.reshape(-1)
            
            ##### Perc bin distribution: pvalue and stderr subsampled on Lcorr
            if h <= top:
                control_sub = x[::nt,::nskip,::nskip].copy();           control_sub = control_sub.reshape(-1)
                var_sub = y[::nt,h,::nskip,::nskip].copy();             var_sub = var_sub.reshape(-1)
            else:
                control_sub = x[::nttop,::nskiptop,::nskiptop].copy();  control_sub = control_sub.reshape(-1)
                var_sub = y[::nttop,h,::nskiptop,::nskiptop].copy();    var_sub = var_sub.reshape(-1)
            
            ##### Perc bin distribution: pvalue
            dist_x, dist_y[h], std_y[h], stderr_y[h], npoints_y[h], pvalue_y_sub[h] = perc_distribution_pvalue_dof(control, variable, control_sub, var_sub, nbins, perc_step, popmean)

            
    elif perc_fixbin in allowed_fb:    
        for h in range(0,y.shape[1]):
            if h % 10 == 0:
                print(h)    
            xx = x.copy(); control = xx.reshape(-1)
            yy = y[:,h].copy(); variable = yy.reshape(-1)
            
            
            
            if h <= top:
                control_sub = x[::nt,::nskip,::nskip].copy();           control_sub = control_sub.reshape(-1)
                var_sub = y[::nt,h,::nskip,::nskip].copy();             var_sub = var_sub.reshape(-1)
            else:
                control_sub = x[::nttop,::nskiptop,::nskiptop].copy();  control_sub = control_sub.reshape(-1)
                var_sub = y[::nttop,h,::nskiptop,::nskiptop].copy();    var_sub = var_sub.reshape(-1)
            
            ##### Perc bin distribution: pvalue
            dist_x, dist_y[h], std_y[h], stderr_y[h], npoints_y[h], pvalue_y_sub[h] = fb_distribution_npoint_pvalue_dof(control, variable, control_sub, var_sub, nbins, perc_step, popmean)

            
    return dist_x, dist_y, std_y, stderr_y, npoints_y, pvalue_y_sub

################################################################




import numpy as np
from scipy import stats

def perc_distribution_pvalue_twoContrs(control_1, control_2, variable, nbins, perc_step, popmean):
    # Initialize output arrays
    distribution = np.zeros((nbins, nbins))
    std_distribution = np.zeros((nbins, nbins))
    std_err_distribution = np.zeros((nbins, nbins))
    number_of_points = np.zeros((nbins, nbins))
    p_value = np.zeros((nbins, nbins))
    
    distribution_control_1 = np.zeros(nbins)
    distribution_control_2 = np.zeros(nbins)
    
    # Remove NaNs
    valid_mask = ~np.isnan(control_1) & ~np.isnan(control_2) & ~np.isnan(variable)
    c1, c2, var = control_1[valid_mask], control_2[valid_mask], variable[valid_mask]

    percentiles = np.arange(0, 100, perc_step)

    # Compute distribution_control_1
    for qq1, pp1 in enumerate(percentiles):
        lower_1 = np.percentile(c1, pp1)
        upper_1 = np.percentile(c1, pp1 + perc_step)
        mask_1 = (c1 >= lower_1) & (c1 < upper_1)
        if np.any(mask_1):
            distribution_control_1[qq1] = np.nanmean(c1[mask_1])
        else:
            distribution_control_1[qq1] = np.nan

    # Compute distribution_control_2
    for qq2, pp2 in enumerate(percentiles):
        lower_2 = np.percentile(c2, pp2)
        upper_2 = np.percentile(c2, pp2 + perc_step)
        mask_2 = (c2 >= lower_2) & (c2 < upper_2)
        if np.any(mask_2):
            distribution_control_2[qq2] = np.nanmean(c2[mask_2])
        else:
            distribution_control_2[qq2] = np.nan

    # Compute gridded stats
    for qq1, pp1 in enumerate(percentiles):
        lower_1 = np.percentile(c1, pp1)
        upper_1 = np.percentile(c1, pp1 + perc_step)
        mask_1 = (c1 >= lower_1) & (c1 < upper_1)

        for qq2, pp2 in enumerate(percentiles):
            lower_2 = np.percentile(c2, pp2)
            upper_2 = np.percentile(c2, pp2 + perc_step)
            mask_2 = (c2 >= lower_2) & (c2 < upper_2)

            bin_mask = mask_1 & mask_2
            bin_values = var[bin_mask]

            if bin_values.size > 0:
                cond_mean = np.nanmean(bin_values)
                cond_std = np.nanstd(bin_values)
                n_points = bin_values.size
                t_stat, p_val = stats.ttest_1samp(bin_values, popmean)

                distribution[qq1, qq2] = cond_mean
                std_distribution[qq1, qq2] = cond_std
                std_err_distribution[qq1, qq2] = cond_std / np.sqrt(n_points)
                number_of_points[qq1, qq2] = n_points
                p_value[qq1, qq2] = p_val
            else:
                distribution[qq1, qq2] = np.nan
                std_distribution[qq1, qq2] = np.nan
                std_err_distribution[qq1, qq2] = np.nan
                number_of_points[qq1, qq2] = 0
                p_value[qq1, qq2] = np.nan

    return distribution_control_1, distribution_control_2, distribution, std_distribution, std_err_distribution, number_of_points, p_value


def distrib_twoContrs(x1, x2, y, perc_step, nbins, popmean):
    xx1 = x1.copy(); control1 = xx1.reshape(-1)
    xx2 = x2.copy(); control2 = xx2.reshape(-1)
    yy = y.copy(); variable = yy.reshape(-1)

    ##### Perc bin distribution: pvalue
    distr_x1, distr_x2, distr_y, std_y, stderr_y, npoints_y, pvalue_y = perc_distribution_pvalue_twoContrs(control1, control2, variable, nbins, perc_step, popmean)

    return distr_x1, distr_x2, distr_y, std_y, stderr_y, npoints_y, pvalue_y





# this function allows to easily store a 
# number of variables into a common .npy file
# which must be specified in th args

def my_save_data(filename, varlist):
    import numpy as np
    import os 
    
    if os.path.exists(filename):
        print('file already exists - DELETING IT and creating it anew')
        os.remove(filename)
    else:
        print('creating new file and saving variables')
    
    with open(filename, 'wb') as f:
        for x in varlist:
            np.save(f, x)























