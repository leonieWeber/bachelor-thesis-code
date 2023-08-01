import fnmatch
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def get_all_sheet_names(notesAndMetadataFilepath):
    """sheet_names == fish_lines """
    df = pd.read_excel(notesAndMetadataFilepath, None)
    sheet_names = list(df.keys())
    return sheet_names


def df_all_fish_lines(notesAndMetadataFilepath):
    fish_lines = get_all_sheet_names(notesAndMetadataFilepath)
    dfs = []
    for fish_line in fish_lines:
        df_tmp = read_sheet(notesAndMetadataFilepath, fish_line)
        df_tmp['fish_line'] = fish_line
        dfs.append(df_tmp)
    df = pd.concat(dfs)
    return df


def read_sheet(notesAndMetadataFilepath, fish_line):
    df = pd.read_excel(notesAndMetadataFilepath, sheet_name=fish_line)
    df_with_folder = add_measurementPath_df(df)
    return df_with_folder


def add_measurementPath_df(df_meta):
    base_folder_measurements = 'data/grayscale_intensity_with_stimuli/'
    df_meta['folder'] = base_folder_measurements + '/' + df_meta.date_rec.astype(str)
    df = df_meta.copy()
    return df


def unique_dateNumCombinations(df_meta):
    df_meta.groupby(['date_rec', 'sweep_num']).min()
    dates_and_nums = list(df_meta.reset_index().set_index(['date_rec', 'sweep_num']).index.unique())
    return dates_and_nums


def extract_sweep_nums(date_and_nums):
    nums_string = date_and_nums[1]
    result = extract_sweep_nums_from_string(nums_string)
    return result


def extract_sweep_nums_from_string(nums_string):
    nums = nums_string.split('-')
    numList = list(range(int(nums[0]), int(nums[1]) + 1))
    return numList


def df_select_date_and_sweepNums(df, date, sweep_nums_string):
    df2 = df[(df.date_rec == date) & (df.sweep_num == sweep_nums_string)]
    return df2


def get_only_element(arrayLike):
    assert len(arrayLike) == 1
    result = arrayLike[0]
    return result


def extract_folder(df, date):
    result_tmp = df[df.date_rec == date].folder.unique()
    result = get_only_element(result_tmp)
    return result


def extract_Roi_perFish(df_total, date, sweep_nums_string):
    df_sweep = df_total[(df_total.date_rec == date) & (df_total.sweep_num == sweep_nums_string)].reset_index()
    ROI_perFish = df_sweep['ROI_perFish'][0]
    return ROI_perFish


def extract_fish_line(df_total, date, sweep_nums_string):
    df_sweep = df_total[(df_total.date_rec == date) & (df_total.sweep_num == sweep_nums_string)].reset_index()
    fish_line = df_sweep['fish_line'][0]
    return fish_line


def make_result_folder_path(df_total, date, sweep_nums_string):
    ProcessedData_baseFolder = 'data/preprocessing_out/'
    fish_line = extract_fish_line(df_total, date, sweep_nums_string)
    result_folder = ProcessedData_baseFolder + str(fish_line)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    return result_folder


def make_result_filename(df_total, date, sweep_nums_string, suffix):
    fish_line = extract_fish_line(df_total, date, sweep_nums_string)
    ROI_perFish = extract_Roi_perFish(df_total, date, sweep_nums_string)
    filename = str(date) + '_' + fish_line + '_' + sweep_nums_string + '_' + str(ROI_perFish) + '_' + suffix + '.csv'
    return filename


# ephys

def extract_first_ephys(df, date, sweep_nums_string='109-124'):
    df2 = df_select_date_and_sweepNums(df, date, sweep_nums_string)
    sweep_nums = extract_sweep_nums_from_string(sweep_nums_string)
    date = get_only_element(df2.date_rec.unique())
    folder = extract_folder(df2, date)
    result_tmp = glob.glob(folder + '/' + '*sw_' + str(sweep_nums[0]) + '_ephys_vrr.txt')
    result = get_only_element(result_tmp)
    return result


def date_time_to_secs(date_times):
    """ format: '2023-05-05 11:37:23.343000' """
    times_in_secs = []
    for t in date_times:
        time_in_secs = int(t[11:13]) * 3600 + int(t[14:16]) * 60 + float(t[17:])
        times_in_secs.append(time_in_secs)
    return times_in_secs


def make_ephys_df(filepath):
    df = pd.read_csv(filepath, header=None, sep="\t", names=['voltage1', 'voltage2', 'date_time'])
    df['fileName'] = filepath.split('/')[-1]
    df['times_in_secs'] = date_time_to_secs(df['date_time'])
    return df


def find_led_time_ephys(df_ephys):
    led_voltages_indices = np.where((df_ephys["voltage2"] > 0.15) & (df_ephys["voltage2"] < 0.16))
    led_time_ephys = df_ephys["times_in_secs"][led_voltages_indices[0][-1]]
    return led_time_ephys


def make_led_flash_ephys_control_plot(df_total, date, sweep_nums_string):
    first_ephys_path = extract_first_ephys(df_total, date, sweep_nums_string)
    df_ephys = make_ephys_df(first_ephys_path)
    led_time_ephys = find_led_time_ephys(df_ephys)
    index = np.where(df_ephys['times_in_secs'] == led_time_ephys)[0][0]
    window = 10000
    plt.plot(df_ephys['times_in_secs'][index - window:index + window],
             df_ephys['voltage2'][index - window:index + window])
    plt.axvline(led_time_ephys, c='red')
    plt.xlabel('time [s]')
    plt.ylabel('voltage led [mV]')
    plt.title(str(date) + '_' + sweep_nums_string)


def save_led_flash_ephys_control_plot(df_total, date, sweep_nums_string):
    first_ephys_path = extract_first_ephys(df_total, date, sweep_nums_string)
    df_ephys = make_ephys_df(first_ephys_path)
    led_time_ephys = find_led_time_ephys(df_ephys)
    index = np.where(df_ephys['times_in_secs'] == led_time_ephys)[0][0]
    window = 10000
    path_resultFolder = make_result_folder_path(df_total, date, sweep_nums_string) + '/'
    plt.plot(df_ephys['times_in_secs'][index - window:index + window],
             df_ephys['voltage2'][index - window:index + window])
    plt.axvline(led_time_ephys, c='red')
    plt.xlabel('time [s]')
    plt.ylabel('voltage led [mV]')
    plt.title(str(date) + '_' + sweep_nums_string)
    plt.savefig(path_resultFolder + str(date) + '_' + sweep_nums_string + '_ephysControl' + '.png')
    plt.clf()


# stimulation

def extract_moving_targets(df, date, sweep_nums_string='109-124'):
    df2 = df_select_date_and_sweepNums(df, date, sweep_nums_string)
    sweep_nums = extract_sweep_nums_from_string(sweep_nums_string)
    folderPath = df2.folder.unique()
    assert len(folderPath) == 1
    folderPath = folderPath[0]
    filePaths = glob.glob(folderPath + '/*')
    result = []
    for path in filePaths:
        if any(fnmatch.fnmatch(path, '*sw_' + str(sweep_num) + '_movingtarget*') for sweep_num in sweep_nums):
            result.append(path)
    return sorted(result)


def combine_multiple_movingtarget_files(paths):
    dfs = []
    for path in paths:
        df = read_movingtarget_file(path)
        dfs.append(df)
    df_total = pd.concat(dfs)
    return df_total


def read_movingtarget_file(filepath):
    df = pd.read_csv(filepath, header=None, sep="\t", names=['position_x', 'position_y', 'orientation', 'date_time'])
    df['fileName'] = filepath.split('/')[-1]
    return df


def make_df_all_stimTimes(df_total, date, sweep_nums_string):
    first_ephys_path = extract_first_ephys(df_total, date, sweep_nums_string)
    df_ephys = make_ephys_df(first_ephys_path)
    led_time_ephys = find_led_time_ephys(df_ephys)
    moving_target_paths = extract_moving_targets(df_total, date, sweep_nums_string)
    df = combine_multiple_movingtarget_files(moving_target_paths)
    df['stim_on'] = 0.175
    df['times_in_secs'] = date_time_to_secs(df['date_time'])
    df['times_afterFlash'] = df['times_in_secs'] - led_time_ephys
    return df


def make_df_stimTimes_relevantInfo(df_all_stimTimes):
    stim_startTimes = df_all_stimTimes.groupby(['fileName']).min().round(3).rename(
        columns={'times_afterFlash': 'start'})
    stim_endTimes = df_all_stimTimes.groupby(['fileName']).max().round(3)
    df_stimTimes = pd.DataFrame(data=stim_startTimes, columns=['orientation', 'start'])
    df_stimTimes['end'] = stim_endTimes['times_afterFlash']
    return df_stimTimes


def save_processed_stims_csv(df_total, date, sweep_nums_string):
    df_stimulation = make_df_all_stimTimes(df_total, date, sweep_nums_string)
    df_stimTimes = make_df_stimTimes_relevantInfo(df_stimulation)
    path_resultFolder = make_result_folder_path(df_total, date, sweep_nums_string) + '/'
    filename = make_result_filename(df_total, date, sweep_nums_string, 'stimulation')
    df_stimTimes.to_csv(path_resultFolder + filename, columns=['start', 'end', 'orientation'], index=False)


# gray Values/recording

def extract_resultsWhole_path(df_total, date, sweep_nums_string):
    df = df_select_date_and_sweepNums(df_total, date, sweep_nums_string)
    folderPath = extract_folder(df, date)
    sweep_nums_underscore = str(sweep_nums_string).replace('-', '_')
    result_tmp = glob.glob(folderPath + '/Results_whole_sw_' + sweep_nums_underscore + '.csv')  # unused function?
    result = get_only_element(result_tmp)
    return result


def extract_path_from_filenamepart(filenamepart, df_total, date, sweep_nums_string):
    df = df_select_date_and_sweepNums(df_total, date, sweep_nums_string)
    folderPath = extract_folder(df, date)
    sweep_nums_underscore = str(sweep_nums_string).replace('-', '_')
    result_tmp = glob.glob(folderPath + '/' + filenamepart + sweep_nums_underscore + '.csv')
    result = get_only_element(result_tmp)
    return result


def extract_path_results_whole(df_total, date, sweep_nums_string):
    path = extract_path_from_filenamepart('Results_whole_sw_', df_total, date, sweep_nums_string)
    return path


def extract_path_results_RoiSet(df_total, date, sweep_nums_string):
    path = extract_path_from_filenamepart('Results_RoiSet_sw_', df_total, date, sweep_nums_string)
    return path


def make_column_names(path):
    df_temp = pd.read_csv(path, header=None)
    if len(df_temp.columns) == 2:
        column_names = ['frame', 'mean_cell1']
    elif len(df_temp.columns) == 3:
        column_names = ['frame', 'mean_cell1', 'mean_cell2']
    elif len(df_temp.columns) == 4:
        column_names = ['frame', 'mean_cell1', 'mean_cell2', 'mean_cell3']
    elif len(df_temp.columns) == 5:
        column_names = ['frame', 'mean_cell1', 'mean_cell2', 'mean_cell3', 'mean_cell4']
    elif len(df_temp.columns) == 6:
        column_names = ['frame', 'mean_cell1', 'mean_cell2', 'mean_cell3', 'mean_cell4', 'mean_cell5']
    elif len(df_temp.columns) == 7:
        column_names = ['frame', 'mean_cell1', 'mean_cell2', 'mean_cell3', 'mean_cell4', 'mean_cell5', 'mean_cell6']
    elif len(df_temp.columns) == 8:
        column_names = ['frame', 'mean_cell1', 'mean_cell2', 'mean_cell3', 'mean_cell4', 'mean_cell5', 'mean_cell6',
                        'mean_cell7']
    elif len(df_temp.columns) == 9:
        column_names = ['frame', 'mean_cell1', 'mean_cell2', 'mean_cell3', 'mean_cell4', 'mean_cell5', 'mean_cell6',
                        'mean_cell7', 'mean_cell8']
    elif len(df_temp.columns) == 10:
        column_names = ['frame', 'mean_cell1', 'mean_cell2', 'mean_cell3', 'mean_cell4', 'mean_cell5', 'mean_cell6',
                        'mean_cell7', 'mean_cell8', 'mean_cell9']
    else:
        raise NotImplementedError("Check number of columns in gray values!")
    return column_names


def make_df_gray_values_Rois(path):
    column_names = make_column_names(path)
    df = pd.read_csv(path, header=None, names=column_names, skiprows=1)
    return df


def make_df_gray_values_whole(path, column_names=['frame', 'mean']):
    df = pd.read_csv(path, header=None, names=column_names, skiprows=1)
    return df


def find_led_flash_in_recording(df_whole):
    """Diff takes e.g. id1-id0 = id1_diff, frame where flash is over"""
    led_flash_window = df_whole[:400]
    led_rec_index = led_flash_window['mean'].diff().idxmin()
    return led_rec_index


def make_led_flash_rec_control_plot(df_total, date, sweep_nums_string):
    path_results_whole = extract_path_results_whole(df_total, date, sweep_nums_string)
    df_grayValues_whole = make_df_gray_values_whole(path_results_whole)
    led_rec_index = find_led_flash_in_recording(df_grayValues_whole)
    plt.plot(df_grayValues_whole['frame'][0:400], df_grayValues_whole['mean'][0:400])
    plt.axvline(led_rec_index, c='red')
    plt.xlabel('frames')
    plt.ylabel('gray values [a.u.]')
    plt.title(str(date) + '_' + sweep_nums_string)


def save_led_flash_rec_control_plot(df_total, date, sweep_nums_string):
    path_results_whole = extract_path_results_whole(df_total, date, sweep_nums_string)
    df_grayValues_whole = make_df_gray_values_whole(path_results_whole)
    led_rec_index = find_led_flash_in_recording(df_grayValues_whole)
    path_resultFolder = make_result_folder_path(df_total, date, sweep_nums_string) + '/'
    plt.plot(df_grayValues_whole['frame'][0:400], df_grayValues_whole['mean'][0:400])
    plt.axvline(led_rec_index, c='red')
    plt.xlabel('frames')
    plt.ylabel('gray values [a.u.]')
    plt.title(str(date) + '_' + sweep_nums_string)
    plt.savefig(path_resultFolder + str(date) + '_' + sweep_nums_string + '_recControl' + '.png')
    plt.clf()


def make_df_grayValues_afterFlash(df_total, date, sweep_nums_string):
    path_results_whole = extract_path_results_whole(df_total, date, sweep_nums_string)
    df_grayValues_whole = make_df_gray_values_whole(path_results_whole)
    led_rec_index = find_led_flash_in_recording(df_grayValues_whole)
    path_results_roiset = extract_path_results_RoiSet(df_total, date, sweep_nums_string)
    df_grayValues = make_df_gray_values_Rois(path_results_roiset)
    df_grayValues_afterFlash = df_grayValues[led_rec_index:]
    return df_grayValues_afterFlash


def save_processed_grayValues_csv(df_total, date, sweep_nums_string):
    df_grayValues_afterFlash = make_df_grayValues_afterFlash(df_total, date, sweep_nums_string)
    path_results_RoiSet = extract_path_results_RoiSet(df_total, date, sweep_nums_string)
    column_names = make_column_names(path_results_RoiSet)[1:]
    path_resultFolder = make_result_folder_path(df_total, date, sweep_nums_string) + '/'
    filename = make_result_filename(df_total, date, sweep_nums_string, 'grayValues')
    df_grayValues_afterFlash.to_csv(path_resultFolder + filename, columns=column_names, index=False)
