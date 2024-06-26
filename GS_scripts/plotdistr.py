# -----
# FUNCTIONS FOR PLOT DISTRIBUTION
# -----

##### Fixed bin distribution: fb
def fb_distribution(control, variable, nbins, theshold_n, perc_step):
    distribution_fb = np.zeros(nbins)
    std_distribution_fb = np.zeros(nbins)
    std_err_distribution_fb = np.zeros(nbins)
    distribution_control_fb = np.zeros(nbins)
    number_of_points_fb = np.zeros(nbins)
    bin_edges_fb = np.zeros(nbins+1)
    
    bw = (np.max(control[~np.isnan(control)])-np.min(control[~np.isnan(control)]))/nbins # Bin width

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step) # Index starting from 0.
        lower = np.min(control[~np.isnan(control)])+qq*bw
        upper = lower + bw
        bin_edges_fb[qq] = lower
        distribution_fb[qq] = np.nanmean(variable[(control>=lower)&(control<upper)])
        std_distribution_fb[qq] = np.nanstd(variable[(control>=lower)&(control<upper)])
        distribution_control_fb[qq] = np.nanmean(control[(control>=lower)&(control<upper)])
        number_of_points_fb[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution_fb[qq] = std_distribution_fb[qq]/np.sqrt(number_of_points_fb[qq])

    bin_edges_fb[-1] = upper
    bin_centers_fb = 0.5*(bin_edges_fb[1:]+bin_edges_fb[:-1])
    dist_control = distribution_control_fb[number_of_points_fb>theshold_n]
    dist_variable = distribution_fb[number_of_points_fb>theshold_n]
    std_err = std_err_distribution_fb[number_of_points_fb>theshold_n] 
    return dist_control, dist_variable, std_err

def fb_distribution_npoint(control, variable, nbins, perc_step):
    distribution_fb = np.zeros(nbins)
    std_distribution_fb = np.zeros(nbins)
    std_err_distribution_fb = np.zeros(nbins)
    distribution_control_fb = np.zeros(nbins)
    number_of_points_fb = np.zeros(nbins)
    bin_edges_fb = np.zeros(nbins+1)
    
    bw = (np.max(control[~np.isnan(control)])-np.min(control[~np.isnan(control)]))/nbins # Bin width

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step) # Index starting from 0.
        lower = np.min(control[~np.isnan(control)])+qq*bw
        upper = lower + bw
        bin_edges_fb[qq] = lower
        distribution_fb[qq] = np.nanmean(variable[(control>=lower)&(control<upper)])
        std_distribution_fb[qq] = np.nanstd(variable[(control>=lower)&(control<upper)])
        distribution_control_fb[qq] = np.nanmean(control[(control>=lower)&(control<upper)])
        number_of_points_fb[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution_fb[qq] = std_distribution_fb[qq]/np.sqrt(number_of_points_fb[qq])

    bin_edges_fb[-1] = upper
    bin_centers_fb = 0.5*(bin_edges_fb[1:]+bin_edges_fb[:-1])
    return distribution_control_fb, distribution_fb, std_err_distribution_fb, number_of_points_fb

def fb_distribution_npoint_pvalue(control, variable, nbins, perc_step, popmean):
    from scipy import stats
    distribution_fb = np.zeros(nbins)
    std_distribution_fb = np.zeros(nbins)
    std_err_distribution_fb = np.zeros(nbins)
    distribution_control_fb = np.zeros(nbins)
    number_of_points_fb = np.zeros(nbins)
    bin_edges_fb = np.zeros(nbins+1)
    p_value = np.zeros(nbins)
    
    bw = (np.max(control[~np.isnan(control)])-np.min(control[~np.isnan(control)]))/nbins # Bin width

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step) # Index starting from 0.
        lower = np.min(control[~np.isnan(control)])+qq*bw
        upper = lower + bw
        bin_edges_fb[qq] = lower
        distribution_fb[qq] = np.nanmean(variable[(control>=lower)&(control<upper)])
        std_distribution_fb[qq] = np.nanstd(variable[(control>=lower)&(control<upper)])
        distribution_control_fb[qq] = np.nanmean(control[(control>=lower)&(control<upper)])
        number_of_points_fb[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution_fb[qq] = std_distribution_fb[qq]/np.sqrt(number_of_points_fb[qq])
        
        t_stat, p_value[qq] = stats.ttest_1samp(variable[(control>=lower)&(control<upper)], popmean=popmean)

    bin_edges_fb[-1] = upper
    bin_centers_fb = 0.5*(bin_edges_fb[1:]+bin_edges_fb[:-1])
    return distribution_control_fb, distribution_fb, std_err_distribution_fb, number_of_points_fb, p_value

##### Percentile distribution
def perc_distribution(control, variable, nbins, perc_step):
    distribution = np.zeros(nbins)
    std_distribution = np.zeros(nbins)
    std_err_distribution = np.zeros(nbins)
    distribution_control = np.zeros(nbins)
    number_of_points = np.zeros(nbins)
    percentiles = np.zeros(nbins+1)

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step)
        lower = np.percentile(control[~np.isnan(control)],pp)
        upper = np.percentile(control[~np.isnan(control)],pp+perc_step)
        percentiles[qq] = lower
        cond_mean = np.nanmean(variable[(control>=lower)&(control<upper)])
        cond_std = np.nanstd(variable[(control>=lower)&(control<upper)])
        cond_mean_control = np.nanmean(control[(control>=lower)&(control<upper)])
        distribution[qq] = cond_mean#-mean
        std_distribution[qq] = cond_std
        distribution_control[qq] = cond_mean_control#-mean_control
        number_of_points[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution[qq] = std_distribution[qq]/np.sqrt(number_of_points[qq])
#         print(qq,number_of_points[qq])
    return distribution_control, distribution, std_distribution, std_err_distribution

##### Percentile distribution - it returns the number of points instead of stderr
def perc_distribution_npoint(control, variable, nbins, perc_step):
    distribution = np.zeros(nbins)
    std_distribution = np.zeros(nbins)
    std_err_distribution = np.zeros(nbins)
    distribution_control = np.zeros(nbins)
    number_of_points = np.zeros(nbins)
    percentiles = np.zeros(nbins+1)

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step)
        lower = np.percentile(control[~np.isnan(control)],pp)
        upper = np.percentile(control[~np.isnan(control)],pp+perc_step)
        percentiles[qq] = lower
        cond_mean = np.nanmean(variable[(control>=lower)&(control<upper)])
        cond_std = np.nanstd(variable[(control>=lower)&(control<upper)])
        cond_mean_control = np.nanmean(control[(control>=lower)&(control<upper)])
        distribution[qq] = cond_mean#-mean
        std_distribution[qq] = cond_std
        distribution_control[qq] = cond_mean_control#-mean_control
        number_of_points[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution[qq] = std_distribution[qq]/np.sqrt(number_of_points[qq])
    return distribution_control, distribution, std_distribution, number_of_points

##### Percentile distribution - it returns the p-value of each bin
# 2 sided pvalue from scipy
def perc_distribution_pvalue(control, variable, nbins, perc_step, popmean):
    from scipy import stats
    distribution = np.zeros(nbins)
    std_distribution = np.zeros(nbins)
    std_err_distribution = np.zeros(nbins)
    distribution_control = np.zeros(nbins)
    number_of_points = np.zeros(nbins)
    percentiles = np.zeros(nbins+1)
    p_value = np.zeros(nbins)

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step)
        lower = np.percentile(control[~np.isnan(control)],pp)
        upper = np.percentile(control[~np.isnan(control)],pp+perc_step)
        percentiles[qq] = lower
        cond_mean = np.nanmean(variable[(control>=lower)&(control<upper)])
        cond_std = np.nanstd(variable[(control>=lower)&(control<upper)])
        cond_mean_control = np.nanmean(control[(control>=lower)&(control<upper)])
        distribution[qq] = cond_mean#-mean
        std_distribution[qq] = cond_std
        distribution_control[qq] = cond_mean_control#-mean_control
        number_of_points[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution[qq] = std_distribution[qq]/np.sqrt(number_of_points[qq])
        
        t_stat, p_value[qq] = stats.ttest_1samp(variable[(control>=lower)&(control<upper)], popmean=popmean)
        
    return distribution_control, distribution, std_distribution, std_err_distribution, p_value


##### Percentile distribution - it returns the p-value of each bin
# 2 sided p-value = 2*(1-cdf(|tvalue|, dof)
def perc_distribution_pvalue_dof(control, variable, nbins, perc_step, popmean, df):
    from scipy import stats
    distribution = np.zeros(nbins)
    std_distribution = np.zeros(nbins)
    std_err_distribution = np.zeros(nbins)
    distribution_control = np.zeros(nbins)
    number_of_points = np.zeros(nbins)
    percentiles = np.zeros(nbins+1)
    p_value = np.zeros(nbins)

    for pp in range(0,100,perc_step):
        qq = int(pp/perc_step)
        lower = np.percentile(control[~np.isnan(control)],pp)
        upper = np.percentile(control[~np.isnan(control)],pp+perc_step)
        percentiles[qq] = lower
        cond_mean = np.nanmean(variable[(control>=lower)&(control<upper)])
        cond_std = np.nanstd(variable[(control>=lower)&(control<upper)])
        cond_mean_control = np.nanmean(control[(control>=lower)&(control<upper)])
        distribution[qq] = cond_mean#-mean
        std_distribution[qq] = cond_std
        distribution_control[qq] = cond_mean_control#-mean_control
        number_of_points[qq] = np.sum(~np.isnan(variable[(control>=lower)&(control<upper)]))
        std_err_distribution[qq] = std_distribution[qq]/np.sqrt(number_of_points[qq])
        
        t_stat, pv = stats.ttest_1samp(variable[(control>=lower)&(control<upper)], popmean=popmean)
        p_value[qq] = 2*(1 - stats.t.cdf(np.abs(t_stat), df=df))
#         print(df, p_value[qq], pv)
        
    return distribution_control, distribution, std_distribution, std_err_distribution, p_value

# ----------
# SCATTER and DENSITY PLOT - it returns figure
# ----------
import numpy as np
import matplotlib.pyplot as plt
def scatterplot_fit(X ,Y, fit, title, xlabel, ylabel, fig):
    xa = np.arange(np.nanmin(X),np.nanmax(X),0.1)
    plt.scatter(X, Y, s=0.2)
    plt.plot(xa, fit.slope*xa + fit.intercept, color='orange')
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    ff2 = "{:.2e}".format
    plt.annotate('y =' + str(ff2(fit.intercept)) + ' + ' + str(ff2(fit.slope)) + '*x', xy=(0.1, 0.9), \
                 xycoords='axes fraction', fontsize=12, color='orange')
    return fig

def scatterplot_fit_sigma(X ,Y, fit, sigma, title, xlabel, ylabel, fig):
    xa = np.arange(np.nanmin(X),np.nanmax(X),0.1)
    plt.scatter(X, Y, s=0.2)
    plt.plot(xa, fit.slope*xa + fit.intercept, color='orange')
    plt.title(title + str(sigma*3) +'km', fontsize=12)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    ff2 = "{:.2e}".format
    plt.annotate('y =' + str(ff2(fit.intercept)) + ' + ' + str(ff2(fit.slope)) + '*x', xy=(0.1, 0.9), \
                 xycoords='axes fraction', fontsize=12, color='orange')
    return fig

def hist2d(x,y, nbin, title, xlabel, ylabel, fig):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    H, xedges, yedges = np.histogram2d(x,y, bins=nbin)
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, H, cmap='Reds')
    plt.colorbar()
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    return fig

def hist2d_sigma(x,y,sigma, nbin, title, xlabel, ylabel, fig):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    H, xedges, yedges = np.histogram2d(x,y, bins=nbin)
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, H, cmap='Reds')
    plt.colorbar()
    plt.title(title + str(sigma*3) +'km', fontsize=12)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    return fig

# ----------
# BOXPLOT
# ----------
# data and labels are lists
def box_plot(data, labels, edge_color, fill_color, ax):
    medianprops = dict(linestyle='-', linewidth=1, color=edge_color)
    meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor=edge_color, markersize=4)
    flierprops = dict(marker='o', markeredgecolor=edge_color, markersize=6,
                  linestyle='none')
    
    bp = ax.boxplot(data, labels=labels, medianprops=medianprops, flierprops=flierprops, \
                    meanprops=meanpointprops, showmeans=True, patch_artist=True)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)       
        
    return bp

def box_plot_pos(data, offset, labels, edge_color, fill_color):
    medianprops = dict(linestyle='-', linewidth=1, color=edge_color)
    meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor=edge_color, markersize=4)
    flierprops = dict(marker='o', markeredgecolor=edge_color, markersize=6,
                  linestyle='none')
    
    pos = np.arange(len(labels))+offset
    
    bp = ax.boxplot(data, positions= pos, labels=labels, medianprops=medianprops, flierprops=flierprops, \
                    meanprops=meanpointprops, showmeans=True, patch_artist=True)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)       
        
    return bp