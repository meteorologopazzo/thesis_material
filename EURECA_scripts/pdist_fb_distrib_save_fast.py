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























