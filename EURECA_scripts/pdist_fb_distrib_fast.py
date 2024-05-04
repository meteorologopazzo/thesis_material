# this is just to cut some code lines in the notebooks
# it only spits out data 

from plotdistr import *
#import pyinputplus as pyip

def distrib_2d(x, y, perc_step, nbins, popmean, perc_fixbin):
    #pyip.inputStr(allowRegexes=[r'perc', 'fixbin']
    allowed_perc = ['p', 'pdist', 'perc', 'percentile'] 
    allowed_fb = ['fb', 'fixbin', 'fixed_bin', 'bin']
    
    if perc_fixbin in allowed_perc:
        xx = x.copy(); control = xx.reshape(-1)
        yy = y.copy(); variable = yy.reshape(-1)

        ##### Perc bin distribution: pvalue
        distr_x, distr_y, std_y, stderr_y, pvalue_y = perc_distribution_pvalue(control, variable, nbins, perc_step, popmean)
        
    elif perc_fixbin in allowed_fb:
        xx = x.copy(); control = xx.reshape(-1)
        yy = y.copy(); variable = yy.reshape(-1)

        ##### Perc bin distribution: pvalue
        distr_x, distr_y, std_y, stderr_y, pvalue_y = fb_distribution_npoint_pvalue(control, variable, nbins, perc_step, popmean)

    return distr_x, distr_y, std_y, stderr_y, pvalue_y
        
def dist_3d_subsample(x,y,perc_step, nbins, popmean, nt, nttop, nskip, nskiptop, top, perc_fixbin):
    
    allowed_perc = ['p', 'pdist', 'perc', 'percentile'] 
    allowed_fb = ['fb', 'fixbin', 'fixed_bin', 'bin']
    
    dist_y = np.zeros((y.shape[1],nbins))
    std_y = np.zeros((y.shape[1],nbins))
    stderr_y = np.zeros((y.shape[1],nbins))
    pvalue_y_sub = np.zeros((y.shape[1],nbins))
    pvalue_y = np.zeros_like(pvalue_y_sub)
    
    if perc_fixbin in allowed_perc:
        for h in range(0,y.shape[1]):
            if h % 10 == 0:
                print(h)    
            xx = x.copy(); control = xx.reshape(-1)
            yy = y[:,h].copy(); variable = yy.reshape(-1)

            ##### Perc bin distribution: pvalue
            dist_x, dist_y[h], std_y[h], stderr_y[h], pvalue_y[h] = perc_distribution_pvalue(control, variable, nbins, perc_step, popmean)

        #if subsample:   # add bool subsample to args
            ##### Perc bin distribution: pvalue subsampled on Lcorr
            if h <= top:
                xx = x[::nt,::nskip,::nskip].copy(); control = xx.reshape(-1)
                yy = y[::nt,h,::nskip,::nskip].copy(); variable = yy.reshape(-1)
            else:
                xx = x[::nttop,::nskiptop,::nskiptop].copy(); control = xx.reshape(-1)
                yy = y[::nttop,h,::nskiptop,::nskiptop].copy(); variable = yy.reshape(-1)
            pdist_control, pdist, pstd, pstderr, pvalue_y_sub[h] = perc_distribution_pvalue(control, variable, nbins, perc_step, popmean)

            
    elif perc_fixbin in allowed_fb: 
        for h in range(0,y.shape[1]):
            if h % 10 == 0:
                print(h)    
            xx = x.copy(); control = xx.reshape(-1)
            yy = y[:,h].copy(); variable = yy.reshape(-1)

            ##### Perc bin distribution: pvalue
            dist_x, dist_y[h], std_y[h], stderr_y[h], pvalue_y[h] = fb_distribution_npoint_pvalue(control, variable, nbins, perc_step, popmean)

        #if subsample:   # add bool subsample to args
            ##### Perc bin distribution: pvalue subsampled on Lcorr
            if h <= top:
                xx = x[::nt,::nskip,::nskip].copy(); control = xx.reshape(-1)
                yy = y[::nt,h,::nskip,::nskip].copy(); variable = yy.reshape(-1)
            else:
                xx = x[::nttop,::nskiptop,::nskiptop].copy(); control = xx.reshape(-1)
                yy = y[::nttop,h,::nskiptop,::nskiptop].copy(); variable = yy.reshape(-1)
            pdist_control, pdist, pstd, pstderr, pvalue_y_sub[h] = perc_distribution_pvalue(control, variable, nbins, perc_step, popmean)

        
            
    return dist_x, dist_y, std_y, stderr_y, pvalue_y