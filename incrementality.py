# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: demo

Date: 01-02-2020
"""

import glob
import os
from math import sqrt, exp

import numpy as np
import pandas as pd


def get_q_value(data):
    return data['w_log_rr_sq'].sum() - ((data['w_log_rr'].sum() ** 2) / data['w'].sum())


def get_tow_value(data, q):
    toue = 0
    k = len(data)
    if q > k - 1:
        toue = (q - k - 1) / (data['w'].sum() - (data['w_sq'].sum() / data['w'].sum()))
    return toue


def apply_zero_correction_factor(row):
    ''' with sample size'''
    if (row["ctrl_conv"] == 0):
        row["zc_ctrl_conv"] = 1 / row["control_reach"]
        row["zc_control_reach"] = row["control_reach"] + (2 / row["control_reach"])
        row["zc_test_conv"] = row["test_conv"] + (1 / row["test_reach"])
        row["zc_test_reach"] = row["test_reach"] + (2 / row["test_reach"])
    if (row["test_conv"] == 0 and row["zc_test_conv"] == 0):
        row["zc_test_conv"] = 1 / row["test_reach"]
        row["zc_test_reach"] = row["test_reach"] + (2 / row["test_reach"])
        row["zc_ctrl_conv"] = row["ctrl_conv"] + (1 / row["control_reach"])
        row["zc_control_reach"] = row["control_reach"] + (2 / row["control_reach"])

    return row


def apply_zero_correction_factor_static(row):
    if (row["ctrl_conv"] == 0):
        row["zc_ctrl_conv"] = 0.5
        row["zc_control_reach"] = row["control_reach"] + 1
        row["zc_test_conv"] = row["test_conv"] + 0.5
        row["zc_test_reach"] = row["test_reach"] + 1
    if (row["test_conv"] == 0 and row["zc_test_conv"] == 0):
        row["zc_test_conv"] = 0.5
        row["zc_test_reach"] = row["test_reach"] + 1
        row["zc_ctrl_conv"] = row["ctrl_conv"] + 0.5
        row["zc_control_reach"] = row["control_reach"] + 1

    return row


def do_meta_analysis(files, output_dir):
    for file in files:
        print('Analyzing file: ', file)
        dataset_orginal = pd.read_excel(file, sheet_name=0)
        reqvars = ['dimension_type', 'dimension_value', 'cuts', 'interval_start_date', 'interval_end_date',
                   'control_reach', 'ctrl_conv', 'test_reach', 'test_conv']
        dataset = dataset_orginal.loc[:, reqvars]

        # Zero correction factor
        zc = 0.5

        # Adding zero correction factor

        dataset["zc_ctrl_conv"] = dataset["ctrl_conv"]
        dataset["zc_control_reach"] = dataset["control_reach"]
        dataset["zc_test_reach"] = dataset["test_reach"]
        dataset["zc_test_conv"] = dataset["test_conv"]
        dataset = dataset.apply(apply_zero_correction_factor_static, axis=1)

        # sample proportions
        dataset["pc"] = dataset["zc_ctrl_conv"] / dataset["zc_control_reach"]
        dataset["pt"] = dataset["zc_test_conv"] / dataset["zc_test_reach"]

        # Risk ratio
        dataset["rr"] = dataset["pc"] / dataset["pt"]

        # incrementality
        dataset["inc"] = 1 - dataset["rr"]

        # logarithm Risk ratio
        dataset["log_rr"] = np.log(dataset["rr"])

        # variance
        dataset['vari'] = (1 / dataset["zc_ctrl_conv"]) - (1 / dataset["zc_control_reach"]) + (
                1 / dataset["zc_test_conv"]) - (1 / dataset["zc_test_reach"])

        # Confidence interval
        dataset["ci_u"] = 1 - np.exp(dataset["log_rr"] - (1.96 * np.sqrt(dataset["vari"])))
        dataset["ci_l"] = 1 - np.exp(dataset["log_rr"] + (1.96 * np.sqrt(dataset["vari"])))

        # Weight of a study of each study
        dataset['w'] = 1 / dataset['vari']
        dataset['w_sq'] = dataset['w'] ** 2

        dataset['w_log_rr'] = dataset['w'] * dataset['log_rr']
        dataset['log_rr_sq'] = dataset['log_rr'] ** 2
        dataset['w_log_rr_sq'] = dataset['w'] * dataset['log_rr_sq']

        # Writing to files
        output_file = output_dir + os.path.basename(file)
        writer_map = {'processed': dataset}

        results = []
        # Meta-analysis of Adset
        # Group them by dimension and calculate Aggregate measure
        for dimension, df_dim_dateset in dataset.groupby('dimension_type'):
            q = get_q_value(df_dim_dateset)
            tow = get_tow_value(df_dim_dateset, q)
            dataset_dim = dataset[dataset['dimension_type'] == dimension]
            # adjusted weight
            dataset_dim['w_adj'] = 1 / (dataset_dim['vari'] + (tow ** 2))
            dataset_dim['w_adj_log_rr'] = dataset_dim['w_adj'] * dataset_dim['log_rr']
            agg_theta = dataset_dim['w_adj_log_rr'].sum() / dataset_dim['w_adj'].sum()
            agg_variance = 1 / dataset_dim['w_adj'].sum()
            # Aggregate incrementality
            agg_inc = 1 - exp(agg_theta)
            # print('Aggregate incrementality for dimension: ', dimension)
            # Aggregate Confidence interval
            agg_ci_lb = 1 - exp(agg_theta + (1.96 * sqrt(agg_variance)))
            agg_ci_ub = 1 - exp(agg_theta - (1.96 * sqrt(agg_variance)))
            print('For dimension:', dimension)
            print('Aggregate incrementality: ', agg_inc)
            print('confidence interval lower bound: ', agg_ci_lb)
            print('confidence interval upper bound: ', agg_ci_ub)
            results.append([dimension, agg_inc, agg_ci_lb, agg_ci_ub])
            writer_map[dimension] = dataset_dim

        meta_df = pd.DataFrame(results, columns=['dimenstion', 'incrementality', 'lower_bound', 'upper_bound'])
        writer_map['resutls'] = meta_df
        write_data_frame_as_sheet(output_file, writer_map)
        print('Done')


def write_data_frame_as_sheet(output_file, writer_map):
    print('Writing to file: ', output_file)
    writer = pd.ExcelWriter(output_file)
    for key, value in writer_map.items():
        value.to_excel(writer, key)
    writer.save()


def get_files_in_directory(dirpath):
    return glob.glob(dirpath)


if __name__ == '__main__':
    raw_dir_path = './data/raw/FB-Aggregation-Science/*'
    processed_dir_path = './data/processed/FB-Aggregation-Science/'
    do_meta_analysis(get_files_in_directory(raw_dir_path), processed_dir_path)
