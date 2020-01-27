# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: demo

Date: 01-09-2020
"""

import os
from math import sqrt, exp

import numpy as np
import pandas as pd


def get_q_value(dataset):
    return dataset['w_log_rr_sq'].sum() - ((dataset['w_log_rr'].sum() ** 2) / dataset['w'].sum())


def get_tow_value(dataset, q):
    toue = 0
    k = len(dataset)
    if q > k - 1:
        toue = (q - k - 1) / (dataset['w'].sum() - (dataset['w_sq'].sum() / dataset['w'].sum()))
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


if __name__ == '__main__':
    print('Begin Analysis')
    os.chdir("/home/mario/work/measured")
    beer_hawk = pd.read_excel("/home/mario/work/measured/Beer Hawk.xlsx", sheet_name=0)
    reqvars = ['dimension_type', 'dimension_value', 'cuts', 'interval_start_date', 'interval_end_date',
               'control_reach', 'ctrl_conv', 'test_reach', 'test_conv']
    beer_hawk_inc = beer_hawk.loc[:, reqvars]

    # Zero correction factor
    zc = 0.5

    # Adding zero correction factor

    beer_hawk_inc["zc_ctrl_conv"] = beer_hawk_inc["ctrl_conv"]
    beer_hawk_inc["zc_control_reach"] = beer_hawk_inc["control_reach"]
    beer_hawk_inc["zc_test_reach"] = beer_hawk_inc["test_reach"]
    beer_hawk_inc["zc_test_conv"] = beer_hawk_inc["test_conv"]
    beer_hawk_inc = beer_hawk_inc.apply(apply_zero_correction_factor, axis=1)

    # sample proportions
    beer_hawk_inc["pc"] = beer_hawk_inc["zc_ctrl_conv"] / beer_hawk_inc["zc_control_reach"]
    beer_hawk_inc["pt"] = beer_hawk_inc["zc_test_conv"] / beer_hawk_inc["zc_test_reach"]

    # Risk ratio
    beer_hawk_inc["rr"] = beer_hawk_inc["pc"] / beer_hawk_inc["pt"]

    # incrementality
    beer_hawk_inc["inc"] = 1 - beer_hawk_inc["rr"]

    # logarithm Risk ratio
    beer_hawk_inc["log_rr"] = np.log(beer_hawk_inc["rr"])

    # variance
    beer_hawk_inc['vari'] = (1 / beer_hawk_inc["zc_ctrl_conv"]) - (1 / beer_hawk_inc["zc_control_reach"]) + (
            1 / beer_hawk_inc["zc_test_conv"]) - (1 / beer_hawk_inc["zc_test_reach"])

    # Confidence interval
    beer_hawk_inc["ci_u"] = 1 - np.exp(beer_hawk_inc["log_rr"] - (1.96 * np.sqrt(beer_hawk_inc["vari"])))
    beer_hawk_inc["ci_l"] = 1 - np.exp(beer_hawk_inc["log_rr"] + (1.96 * np.sqrt(beer_hawk_inc["vari"])))

    # Weight of a study of each study
    beer_hawk_inc['w'] = 1 / beer_hawk_inc['vari']
    beer_hawk_inc['w_sq'] = beer_hawk_inc['w'] ** 2

    beer_hawk_inc['w_log_rr'] = beer_hawk_inc['w'] * beer_hawk_inc['log_rr']
    beer_hawk_inc['log_rr_sq'] = beer_hawk_inc['log_rr'] ** 2
    beer_hawk_inc['w_log_rr_sq'] = beer_hawk_inc['w'] * beer_hawk_inc['log_rr_sq']

    # Meta-analysis of Adset
    # Group them by dimension and calculate Aggregate measure
    for dimension, df_dim_dateset in beer_hawk_inc.groupby('dimension_type'):
        q = get_q_value(df_dim_dateset)
        tow = get_tow_value(df_dim_dateset, q)
        beer_hawk_inc_dim = beer_hawk_inc[beer_hawk_inc['dimension_type'] == dimension]
        # adjusted weight
        beer_hawk_inc_dim['w_adj'] = 1 / (beer_hawk_inc_dim['vari'] + (tow ** 2))
        beer_hawk_inc_dim['w_adj_log_rr'] = beer_hawk_inc_dim['w_adj'] * beer_hawk_inc_dim['log_rr']
        agg_theta = beer_hawk_inc_dim['w_adj_log_rr'].sum() / beer_hawk_inc_dim['w_adj'].sum()
        agg_variance = 1 / beer_hawk_inc_dim['w_adj'].sum()
        # Aggregate incrementality
        agg_inc = 1 - exp(agg_theta)
        # print('Aggregate incrementality for dimension: ', dimension)
        # Aggregate Confidence interval
        agg_ci_lb = 1 - exp(agg_theta + (1.96 * sqrt(agg_variance)))
        agg_ci_ub = 1 - exp(agg_theta - (1.96 * sqrt(agg_variance)))
        print('For dimension:', dimension)
        print('Aggregate incrementality: ', agg_inc)
        # print('confidence interval lower bound: ', agg_ci_lb)
        # print('confidence interval upper bound: ', agg_ci_ub)
    print('Done')
