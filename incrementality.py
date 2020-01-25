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
    # print(dataset)
    theta_mean = dataset['log_rr'].mean()
    dataset['log_rr_mean'] = theta_mean
    dataset['q_i'] = dataset['w'] * ((dataset['log_rr'] - theta_mean) ** 2)
    dataset['q'] = dataset['q_i'].sum()
    return dataset['q_i'].sum()


def get_tow_value(dataset, q):
    tow = 0
    k = len(dataset)
    if q > k - 1:
        tow = dataset['w'].sum() - (dataset['w_sq'].sum() / dataset['w'].sum())
    return tow


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
    beer_hawk_inc["zc_ctrl_conv"] = beer_hawk_inc["ctrl_conv"] + 0.5
    beer_hawk_inc["zc_control_reach"] = beer_hawk_inc["control_reach"] + 0.5
    beer_hawk_inc["zc_test_reach"] = beer_hawk_inc["test_reach"] + 0.5
    beer_hawk_inc["zc_test_conv"] = beer_hawk_inc["test_conv"] + 0.5

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
    beer_hawk_inc["ci_l"] = 1 - np.exp(beer_hawk_inc["log_rr"] + (1.96 * np.sqrt(beer_hawk_inc["vari"])))
    beer_hawk_inc["ci_u"] = 1 + np.exp(beer_hawk_inc["log_rr"] + (1.96 * np.sqrt(beer_hawk_inc["vari"])))

    # Weight of a study of each study
    beer_hawk_inc['w'] = 1 / beer_hawk_inc['vari']
    beer_hawk_inc['w_sq'] = beer_hawk_inc['w'] ** 2

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
        print('Aggregate incrementality for dimenstion: ', dimension, ' is: ', agg_inc)
        # Aggregate Confidence interval
        agg_ci_lb = 1 - exp(agg_theta + (1.96 * sqrt(agg_variance)))
        agg_ci_ub = 1 + exp(agg_theta + (1.96 * sqrt(agg_variance)))
        print('For dimension:', dimension)
        print('Aggregate incrementality: ', agg_inc)
        print('confidence interval: ', agg_ci_lb, agg_ci_ub)

    print('Done')

# theta = np.log(pc / pt)
#
# var = (1 / beer_hawk_inc["zc_ctrl_conv"]) - (1 / beer_hawk_inc["zc_control_reach"]) + (
#        1 / beer_hawk_inc["zc_test_conv"]) - (1 / beer_hawk_inc["zc_test_reach"])
#
# weight = 1 / var
#
# wtheta = weight * theta
#
# wtheta2 = weight * (theta ** 2)
#
# weight2 = weight ** 2
#
# beer_hawk_inc["theta"] = theta
# beer_hawk_inc["var"] = var
# beer_hawk_inc["weight"] = weight
# beer_hawk_inc["wtheta"] = wtheta
# beer_hawk_inc["wtheta2"] = wtheta2
# beer_hawk_inc["weight2"] = weight2
#
# beer_hawk_inc["nstudies"] = beer_hawk_inc.apply(lambda x: 0 if np.isnan(x["control_reach"]) or np.isnan(x["test_reach"])
# else 1, axis=1)
# beer_hawk_inc["nstudies"].head()
#
# cleandata = beer_hawk_inc.loc[beer_hawk_inc["nstudies"] == 1, :]
#
# cleandata.dimension_type.value_counts()
#
## -----------------------------------------------Filtering the Data at three Levels-----------------------------------------#
#
# rollups = ["Adset", "Campaign", "Summary"]
#
# '''
# for level in rollups:
#    "".join(["cleandata","_",level]) = cleandata.loc[cleandata["dimension_type"] == level,:]
#    
# '''
#
# cleandata_Adset = cleandata.loc[cleandata["dimension_type"] == "Adset", :]
# cleandata_Campaign = cleandata.loc[cleandata["dimension_type"] == "Campaign", :]
# cleandata_Summary = cleandata.loc[cleandata["dimension_type"] == "Summary", :]
#
## ---------------------------------------------------Meta_Ma--------------------------------------------------------------#
#
# '''
#
# varNames = ['dimension_type','dimension_value','theta','variance',
#            'weight','cochranQ','tau2','adjTheta','adjWeight','dof_ma']
#
#
# cleandata_Adset.reset_index(drop = True,inplace = True)
## cleandata_Adset.drop_duplicates(inplace = True)
#
# unique_dim_val = cleandata_Adset["dimension_value"].value_counts().index
#
#
# nrows_ind = len(unique_dim_val)
#
#
# metadata = (pd.DataFrame(index=np.arange(nrows_ind), columns=['dimension_type','dimension_value','theta','variance',
#            'weight','cochranQ','tau2','adjTheta','adjWeight','dof_ma'])).fillna(0)
#
# metadata.loc[:,"dimension_value"] = unique_dim_val
# metadata.loc[:,"dimension_type"] = cleandata_Adset.loc[0,"dimension_type"]
#
# cleandata_Adset.to_csv("./DataScience/Analysis/Adset_data.csv",index = False)
#
#
# '''
# count_df = pd.DataFrame(cleandata_Adset["dimension_value"].value_counts())
#
# count_df.reset_index(inplace=True)
# count_df.head()
#
# cleandata_Adset_dof = pd.merge(cleandata_Adset, count_df, how="left", left_on="dimension_value",
#                               right_on="index")
#
# cleandata_Adset_dof.drop(["index"], axis=1, inplace=True)
#
# cleandata_Adset_dof.rename(columns={"dimension_value_x": "dimension_value", "dimension_value_y": "counts"},
#                           inplace=True)
#
# cleandata_Adset_dof["dof_ma"] = cleandata_Adset_dof["counts"] - 1
#
# cleandata_Adset_dof.loc[cleandata_Adset_dof["dof_ma"] == 0, "cochranQ"] = 0
# cleandata_Adset_dof.loc[cleandata_Adset_dof["dof_ma"] == 0, "tau2"] = 0
# cleandata_Adset_dof.loc[cleandata_Adset_dof["dof_ma"] == 0, "adjTheta"] = cleandata_Adset_dof.loc[
#    cleandata_Adset_dof["dof_ma"] == 0, "theta"]
# cleandata_Adset_dof.loc[cleandata_Adset_dof["dof_ma"] == 0, "adjWeight"] = cleandata_Adset_dof.loc[
#    cleandata_Adset_dof["dof_ma"] == 0, "weight"]
#
# unique_dim_val = list(
#    cleandata_Adset_dof["dimension_value"].value_counts().reset_index(name="count").query("count > 1")["index"])
#
# for dim_val in unique_dim_val:
#    cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "theta"] = np.dot(
#        cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "theta"],
#        cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "weight"]) / (np.sum(
#        cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "weight"]))
#
#    cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "variance"] = 1 / np.sum(
#        cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "weight"])
#
#    cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "weight"] = np.sum(
#        cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "weight"])
#
#    cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "cochranQ"] = np.sum(
#        cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "wtheta2"]) - np.sum(
#        np.square(cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "wtheta"]))
#
#    tau2 = cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "cochranQ"] - \
#           cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "dof_ma"] / np.sum(
#        cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "weight"]) - (np.sum(
#        cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "weight2"] / np.sum(
#            cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "weight"])))
#
#    cleandata_Adset_dof.loc[cleandata_Adset_dof["dimension_value"] == dim_val, "tau2"] = np.max(tau2)
#
#    cleandata_Adset_dof.loc["adjWeight"] = np.divide(1, cleandata_Adset_dof["variance"] + cleandata_Adset_dof.loc[
#        cleandata_Adset_dof["dimension_value"] == dim_val, "tau2"])
#
## for row in cleandata_Adset_dof.itertuples(index = False)
