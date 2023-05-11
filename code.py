#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 00:43:13 2023

@author: Sania
"""

from numpy.polynomial.polynomial import polyfit
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as opt
import os
import itertools as iter


def err_ranges(x, func, param, sigma):
    """
    returns the upper and lower ranges for error in data

    """
    # initiate arrays for lower and upper limits
    low = func(x, *param)
    upper = low
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        low = np.minimum(low, y)
        upper = np.maximum(upper, y)
        
    return low, upper   

 

def cluster_cleanfuel_youth(df, xcolumn, ycolumn, xlabel, ylabel, title):
    """
    Parameters
    ----------
    df : DataFrame
        Dataframe object holding the dta.
    xcolumn : String
        Column name of the column for x-axis.
    ycolumn : String
        Column name of the column for Y-axis.
    xlabel : String
        label for x-axis.
    ylabel : String
        label for y-axis.
    title : String
        Title of the plot.

    Returns
    -------
    None.

    """
    # Separate x and y axis data in x, y variables
    x = df[xcolumn]
    y = df[ycolumn]
    
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[[xcolumn, ycolumn]])

    # Get centers for data clusters
    centroids = kmeans.cluster_centers_
    cen_x = [i[0] for i in centroids] 
    cen_y = [i[1] for i in centroids]
    ## add to df
    df['cen_x'] = df.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
    df['cen_y'] = df.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})
    
    # define and map colors
    colours = ['#DF2020', '#81DF20', '#2095DF']
    df['colour'] = df.cluster.map({0:colours[0], 1:colours[1], 2:colours[2]})

    fig, ax = plt.subplots()

    ax.scatter(x, y, 
                c=df.colour, alpha = 1, s=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.scatter(df['cen_x'], df['cen_y'], 10, "blue", marker="d",)
    b, m = polyfit(x, y, 1)
    plt.plot(x, b + m * x, '--')
    
    plt.plot([x.mean()]*2, [0,40], color='#ddd', lw=0.5, linestyle='--')
    plt.plot([0,100], [y.mean()]*2, color='#ddd', lw=0.5, linestyle='--')
    
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    ax.plot(x, p(x), label='Fit', linewidth=0.1)
    
    
def logistics(t, n0, g, t0):
    """
    Parameters
    ----------
    t : number
    n0 : number
    g : number
    t0 : number

    Returns
    -------
    f : Logistic function output

    """
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

def linfunc(x, a, b):
    y = a*x + b
    return y
    
    
def youth_dying_plot(df):
    """
    Parameters
    ----------
    df : DataFrame
        DataFrame object holding the required data.

    Returns
    -------
    None.

    """
    yearly_indexed_df = df.copy()
    yearly_indexed_df = yearly_indexed_df.set_index("Country Name")

    yearly_indexed_df = yearly_indexed_df.drop(
        ['Indicator Name', 'Indicator Code', 'Country Code'], axis=1)

    yearly_indexed_df = yearly_indexed_df.T
    yearly_indexed_df = yearly_indexed_df.dropna()
    af_data = yearly_indexed_df[["Africa Eastern and Southern"]]
    af_data['year'] = pd.to_numeric(af_data.index)
    param, covar = opt.curve_fit(logistics, af_data["year"], 
                                 af_data["Africa Eastern and Southern"], 
                                 p0=(60, 0.01, 2005))
    af_data["fit"] = logistics(af_data["year"], *param)
    af_data.plot("year", ["Africa Eastern and Southern", "fit"])
    # af_data.plot(af_data["year"], af_data['Africa Eastern and Southern'])
    plt.plot(af_data["year"], af_data['Africa Eastern and Southern'])
    plt.ylabel("Probability of dying youth")
    plt.xlabel("Year")
    plt.title("Probability of dying youth (ages 20-24 years)")
    
    # Error Ranges
    sigma = np.sqrt(np.diag(covar))
    year = np.arange(1990, 2021)
    low, up = err_ranges(year, logistics, param, sigma)
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    
    plt.show()



dirc = os.path.dirname(__file__) # Get absolute path of current directory

# File names
clean_fuel_path = 'clean_fuel_access.csv'
agri_nitrous_path = 'agri_nitrouse_oxide_emission.csv'
youth_dying_path = 'youth_dying_probability.csv'

# Read data from files as Panda's DataFrame objects
clean_fuel_data = pd.read_csv(os.path.join(dirc, clean_fuel_path))
nitrous_data = pd.read_csv(os.path.join(dirc, agri_nitrous_path))
youth_dying_data = pd.read_csv(os.path.join(dirc, youth_dying_path))

years_2000_2008 = [str(i) for i in range(2000, 2008)]

# Scatter matrix
pd.plotting.scatter_matrix(nitrous_data[years_2000_2008], 
                                      figsize=(8, 5), s=5, alpha=0.8)

common_year = '2008' # A common and last year in all datasets

# Clean the data for year 2008
clean_fuel_data = clean_fuel_data[clean_fuel_data[common_year].notna()]
nitrous_data = nitrous_data[nitrous_data[common_year].notna()]
youth_dying_data = youth_dying_data[youth_dying_data[common_year].notna()]

# copy data for only year 2008
clean_fuel_2008 =  clean_fuel_data[["Country Name", "Country Code", 
                                 common_year]].copy()
nitrous_2008 =  nitrous_data[["Country Name", "Country Code", 
                                 common_year]].copy()
youth_dying_2008 =  youth_dying_data[["Country Name", "Country Code", 
                                 common_year]].copy()


data_clean_fuel_youth = pd.merge(clean_fuel_2008, youth_dying_2008, 
                                 on="Country Name", how="outer")
data_clean_fuel_youth = data_clean_fuel_youth.dropna()

# Rename columns
data_clean_fuel_youth = data_clean_fuel_youth.rename(columns = {
    "2008_x":"Clean Fuel Access", "2008_y":"Youth Dying"})
cluster_cleanfuel_youth(data_clean_fuel_youth, "Clean Fuel Access", "Youth Dying",
                        "Clean Fuel Access (% of population)",
                        "Probability of dying youth ages 20-24 years",
                        "Relation between Probability of youth dying and clean fuel \
                    access")

data_nitrous_youth = pd.merge(nitrous_2008, youth_dying_2008, 
                                 on="Country Name", how="outer")
data_nitrous_youth = data_nitrous_youth.dropna()

# Rename columns
data_nitrous_youth = data_nitrous_youth.rename(columns = {
    "2008_x":"Agritural Nitrous Emission", "2008_y":"Youth Dying"})



cluster_cleanfuel_youth(data_nitrous_youth, "Agritural Nitrous Emission", 
                        "Youth Dying",
                        "Agritural Nitrous Emission",
                        "Probability of dying youth ages 20-24 years",
                        "Relation between Probability of youth dying and \
agritural nitrous emission")

youth_dying_plot(youth_dying_data)












