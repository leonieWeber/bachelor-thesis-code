import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from scipy.stats import kruskal
import scikit_posthocs as sp

def make_df_for_one_recording(filepath):
    df_viewer = pd.read_csv(filepath)

    roi_nums = []
    for entry in df_viewer.roi:
        roi_num = entry[-1]
        roi_nums.append(roi_num)
    df_viewer['roi_num'] = roi_nums

    rec_ID = re.search(r'\d{8}.+_\d{1}', df_viewer.recording_name[0]).group(0)
    date = re.search(r'^\d{8}', df_viewer.recording_name[0]).group(0)
    fish_num = re.search(r'\d{3}-\d{3}', df_viewer.recording_name[0]).group(0)[0]
    roi_perFish = rec_ID[-1]
    line = re.search(r'_\D{2,3}_\D{4,5}\d*\D*_', rec_ID).group(0)[1:-1]

    df_viewer['roi_ID'] = rec_ID + '_' + df_viewer.roi_num
    df_viewer['line'] = re.search(r'_\D{2,3}_\D{4,5}\d*\D*_', rec_ID).group(0)[1:-1]
    df_viewer['event'] = df_viewer.event.astype(str)
    df_viewer['event_ID'] = df_viewer.roi_ID + '_' + df_viewer.event
    df_viewer['fish_ID'] = date + '_' + df_viewer.line + '_' + fish_num
    df_viewer['cell_ID'] = df_viewer.fish_ID + '_' + roi_perFish + '_' + df_viewer.roi_num
    df_viewer['sensor'] = line[4:]
    return df_viewer

def get_all_filepaths(folderpath):
    subfolder_paths = glob.glob(folderpath + '/*')
    all_filepaths = []
    for path in subfolder_paths:
        filepaths = glob.glob(path + '/*viewerResults.csv')
        all_filepaths.extend(filepaths)
    return all_filepaths

def make_df_all(all_filepaths):
    dfs_all = []
    for path in all_filepaths:
        df = make_df_for_one_recording(path)
        dfs_all.append(df)
    df_all = pd.concat(dfs_all)
    return df_all

def make_swarm_with_box_plot(df_all, y_values, subgroups, y_label, colors = 'bright'):
    ax = sns.swarmplot(data = df_all, x = 'sensor', y = y_values, hue = subgroups, palette = colors, size = 4.3, legend = None, zorder = 0)
    ax = sns.boxplot(data = df_all, x = 'sensor', y = y_values, boxprops={'facecolor':'None'}, showfliers = False)
    ax.set(xlabel = 'sensor', ylabel = y_label + ' [s]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.rcParams.update({'font.size': 12})


def kruskal_wallis_with_post_hoc(groups):
    """
    Perform a Monte Carlo post hoc test based on the Kruskal-Wallis test.

    Args:
        groups (list or array-like): List of groups, where each group is a 1D array of observations.
        monte_carlo (bool): Use monte carlo post hoc or tukey hsd post hoc
        n_iterations: Nubmer of monte carlo iterations

    Returns:
        pandas.DataFrame: DataFrame containing pairwise comparisons and corresponding p-values.
    """
    # Perform the Kruskal-Wallis test
    h, p_value = kruskal(*groups)

    # Perform Post Hoc Tests
    post_hoc = dict()
    post_hoc['dunn'] = sp.posthoc_dunn(groups, p_adjust='bonferroni')
    post_hoc['mann_whitney'] = sp.posthoc_mannwhitney(groups, p_adjust='bonferroni')
    post_hoc['conover'] = sp.posthoc_conover(groups, p_adjust='bonferroni')
    # post_hoc['wilcoxon'] = sp.posthoc_wilcoxon(groups, p_adjust='bonferroni')  # only works for equal group size

    return h, p_value, post_hoc

def get_unique_pairs(lst):
    unique_pairs = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            pair = (lst[i], lst[j])
            unique_pairs.append(pair)
    return unique_pairs

def calculate_p_kruskal_dunn(df_all, tau_to_compare):
    lines = list(df_all.line.unique())
    groups = []
    for line in lines:
        group = df_all[(df_all.line == line)][tau_to_compare].values
        groups.append(group)

    _, p, post_hoc_p = kruskal_wallis_with_post_hoc(groups)
    p_dunn = post_hoc_p['dunn'].round(5)
    return p_dunn

def make_df_p_values(df_all, tau_to_compare):
    df_p_dunn = calculate_p_kruskal_dunn(df_all, tau_to_compare)
    lines = list(df_all.line.unique())
    line_nums = list(range(1,len(lines)+1))
    unique_pairs = get_unique_pairs(line_nums)

    p_dunn_unique = []
    for pair in unique_pairs:
         p_dunn_unique.append((pair,df_p_dunn[pair[0]][pair[1]]))

    df_p_dunn = pd.DataFrame(p_dunn_unique, columns = ['pair', 'p'])
    df_p_dunn['lines'] = get_unique_pairs(lines)

    significance = []
    for p in df_p_dunn.p:
        if p < 0.001:
            ps = '***'
        elif p < 0.01:
            ps = '**'
        elif p < 0.05:
            ps = '*'
        elif p > 0.05:
            ps = 'n.s.'
        significance.append(ps)

    df_p_dunn['significance'] = significance
    return df_p_dunn

def make_sig_annotations(df_p, y_annot_list):
    y_annotations = y_annot_list
    for idx in range(0, len(df_p.pair)):
        if df_p.significance[idx] != 'n.s.':
            x1, x2 = df_p.pair[idx][0]-1, df_p.pair[idx][1]-1
            y, h, col = y_annotations[idx], 0, 'gray'
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            plt.text((x1+x2)*.5, y+h, df_p.significance[idx] , ha='center', va='bottom', color=col)
        else: pass

def make_df_tau_times(df_all):
    df_tau_times = pd.DataFrame(df_all.tau_decay).copy()
    df_tau_times['line'] = df_all.line.copy()
    df_tau_times['tau_rise'] = df_all.tau_rise.copy()
    df_tau_times['fit_rise_tau'] = df_all.fit_rise_tau.copy()
    df_tau_times['fit_decay_tau'] = df_all.fit_decay_tau.copy()
    return df_tau_times

def get_n_of_unique_IDs_per_line(df_all, name_ID):
    IDs = df_all.groupby('line')[name_ID].unique()
    n_IDs = []
    for l in range(0,4):
        n = len(IDs[l])
        n_IDs.append(n)
    return n_IDs

def make_df_means_and_n_IDs(df_all):
    df_tau_times = make_df_tau_times(df_all)
    df_means = pd.DataFrame(df_tau_times.groupby(['line']).mean())
    n_fish = get_n_of_unique_IDs_per_line(df_all, 'fish_ID')
    n_cells = get_n_of_unique_IDs_per_line(df_all, 'cell_ID')
    n_events = get_n_of_unique_IDs_per_line(df_all, 'event_ID')
    df_means['n_fish'] = n_fish
    df_means['n_cells'] = n_cells
    df_means['n_events'] = n_events
    return df_means

def make_CV_barplots(df_coeff_variation):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(4,12))
    ax1.bar(x=df_coeff_variation.sensor, height=df_coeff_variation.tau_rise, width=0.4, color='orangered')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('CV of 63% rise time')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.bar(x=df_coeff_variation.sensor, height=df_coeff_variation.fit_rise_tau, width=0.4, color='darkorange')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('CV of rise time constant')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.bar(x=df_coeff_variation.sensor, height=df_coeff_variation.tau_decay, width=0.4, color='blue')
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('CV of 37% decay time')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax4.bar(x=df_coeff_variation.sensor, height=df_coeff_variation.fit_decay_tau, width=0.4, color='cornflowerblue')
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('sensor')
    ax4.set_ylabel('CV of decay time constant')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    plt.rcParams.update({'font.size': 12})

def make_df_fitquality_one_signal (filepath_fitquality):
    df_r_squared = pd.read_csv(filepath_fitquality)
    recording = re.search(r'\d{8}.+_\d{1}/', filepath_fitquality).group(0)[:-1]
    line = re.search(r'_\D{2,3}_\D{4,5}\d*\D*_', filepath_fitquality).group(0)[1:-1]
    event = re.search(r'mean_cell\d{1}_\d{1,2}', filepath_fitquality).group(0)
    df_r_squared['recording'] = recording
    df_r_squared['line'] = line
    df_r_squared['event'] = event
    return df_r_squared

def get_fitquality_filepaths(path_r2_folder):
    folderpaths = glob.glob(path_r2_folder + '/*')
    all_filepaths = []
    for path in folderpaths:
        subfolderpaths = glob.glob(path + '/*')
        for p in subfolderpaths:
            filepaths = glob.glob(p + '/*goodness_of_fit.csv')
            all_filepaths.extend(filepaths)
    return all_filepaths

def make_df_all_fitqualities(path_r2_folder):
    all_filepaths = get_fitquality_filepaths(path_r2_folder)
    all_dfs = []
    for f in all_filepaths:
        df = make_df_fitquality_one_signal(f)
        all_dfs.append(df)
    df_fitquality = pd.concat(all_dfs)
    return df_fitquality

def make_df_r_squared(df_fitquality):
    df_r_squared = pd.DataFrame(df_fitquality.rise_r_squared)
    df_r_squared['decay_r_squared'] = df_fitquality.decay_r_squared
    df_r_squared['line'] = df_fitquality.line
    return df_r_squared
