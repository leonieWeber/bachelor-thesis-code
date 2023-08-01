import calcium_signal_analysis as c
 
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
import scipy
import scikit_posthocs as sp
from scipy.stats import kruskal

folderpath_viewer = 'data/manual_step/ResultFiles_viewer/'
r2_folderpath = 'data/manual_step/fit_quality/'
tex_figures_base_path = 'result_figures/'


all_filepaths = c.get_all_filepaths(folderpath_viewer)
df_all = c.make_df_all(all_filepaths)

df_p_decay = c.make_df_p_values(df_all, 'tau_decay')
df_p_decay_fit = c.make_df_p_values(df_all, 'fit_decay_tau')
df_p_rise = c.make_df_p_values(df_all, 'tau_rise')
df_p_rise_fit = c.make_df_p_values(df_all, 'fit_rise_tau')

df_tau_times = c.make_df_tau_times(df_all)
df_std = pd.DataFrame(df_tau_times.groupby(['line']).std())
df_means = pd.DataFrame(df_tau_times.groupby(['line']).mean())
df_means_n = c.make_df_means_and_n_IDs(df_all)
df_coeff_variation = df_std/df_means
df_coeff_variation['sensor'] = ['GCaMP3','GCaMP5','GCaMP6s','GRAB']
df_coeff_variation = df_coeff_variation.reindex(['nbt_GRAB', 'HuC_GCaMP3', 'HuC_GCaMP6s', 'HuC_GCaMP5'])

df_fitquality = c.make_df_all_fitqualities(r2_folderpath)
df_r_squared = c.make_df_r_squared(df_fitquality)
df_means_r_squared = pd.DataFrame(df_r_squared.groupby(['line']).mean())



# Plot 1
c.make_swarm_with_box_plot(df_all, 'tau_decay', 'cell_ID', '37% decay time')
c.make_sig_annotations(df_p_decay, [6, 12, 13, 14, 15, 15])
plt.ylim(0,17.5)
plt.savefig(tex_figures_base_path+'tau_decay_man.pdf', bbox_inches='tight')
plt.clf()

# Plot 2
c.make_swarm_with_box_plot(df_all, 'fit_decay_tau', 'cell_ID', 'decay time constant')
c.make_sig_annotations(df_p_decay_fit, [7, 12.5, 13.5, 14.5, 15.5, 15.5])
plt.ylim(0,17.5)
plt.savefig(tex_figures_base_path+'tau_decay_fit.pdf',bbox_inches='tight')
plt.clf()

# Plot 3
c.make_swarm_with_box_plot(df_all, 'fit_rise_tau', 'cell_ID', 'rise time constant')
c.make_sig_annotations(df_p_rise_fit, [2.3, 2.6, 2.4, 2.6, 2.8, 3])
plt.ylim(0,2.8)
plt.savefig(tex_figures_base_path+'tau_rise_fit.pdf', bbox_inches='tight')
plt.clf()

# Plot 4
c.make_swarm_with_box_plot(df_all, 'tau_rise', 'cell_ID', '63% rise time')
c.make_sig_annotations(df_p_rise, [2, 2.3, 2.6, 2.9, 3.2, 2.6])
plt.ylim(0, 2.8)
plt.savefig(tex_figures_base_path+'tau_rise_man.pdf', bbox_inches='tight')
plt.clf()

# Plot 5
own_colors = ['firebrick','turquoise','cornflowerblue','orange','blue','darkviolet','limegreen', 'darkblue','red','violet','teal','green','chocolate']
c.make_swarm_with_box_plot(df_all, 'tau_decay', 'fish_ID', '37% decay times', own_colors)
c.make_sig_annotations(df_p_decay, [6, 12, 13, 14, 15, 15])
plt.savefig(tex_figures_base_path+'tau_decay_man_fishID.pdf', bbox_inches='tight')
plt.clf()

# Plot 6
c.make_swarm_with_box_plot(df_all, 'fit_decay_tau', 'fish_ID', 'decay time constant', own_colors )
c.make_sig_annotations(df_p_decay_fit, [7, 12.5, 13.5, 14.5, 15.5, 15.5])
plt.savefig(tex_figures_base_path+'tau_decay_fit_fishID.pdf', bbox_inches='tight')
plt.clf()

# Plot 7
c.make_CV_barplots(df_coeff_variation)
plt.savefig(tex_figures_base_path+'coefficient_of_variance_all.pdf', bbox_inches='tight')
plt.clf()
