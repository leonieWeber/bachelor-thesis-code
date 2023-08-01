import fnmatch
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calcium_signal_analysis as c 
import preprocessing as p 


def read_data():
    notesAndMetadataFilepath = 'data/RoiSetsNotesAndMetadata.ods'
    df_total = p.df_all_fish_lines(notesAndMetadataFilepath)
    return df_total


def run_preprocessing(df_total):
    all_dateNumCombinations = p.unique_dateNumCombinations(df_total)
    for comb in all_dateNumCombinations:
        date = comb[0]
        sweep_nums_string = comb[1]
        try:
            p.save_processed_stims_csv(df_total, date, sweep_nums_string)
            p.save_processed_grayValues_csv(df_total, date, sweep_nums_string)
            p.save_led_flash_ephys_control_plot(df_total, date, sweep_nums_string)
            p.save_led_flash_rec_control_plot(df_total, date, sweep_nums_string)
        except:
            print('check file', date, sweep_nums_string)


df_total = read_data()
run_preprocessing(df_total)
