import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time
import multiprocess
from multiprocess import Pool
from itertools import product
from SeriesAnalysis import funs as f

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

num_procs = multiprocess.cpu_count()

df = pd.read_csv('data/series_2019.csv')
airports = pd.read_csv('data/airports.csv')

departures_2019, arrivals_2019 = f.get_departure_arrivals(df, airports)

departure = True

# day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# tol = [30, 35, 40, 45, 50, 55, 60]
# min_oc = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
# f.find_series_plot(departures_2019, day, call_icao, num_procs, departure, 2019, save=True)

day = 0
tol = 30
max_occurence = 0.7
series_2019 = f.find_series(departures_2019, day, tol, max_occurence, departure, num_procs)

series_2019["arrival"].unique()

departures_2019[(departures_2019.departure == "BIKF") & (departures_2019.arrival == "EGKR")].sort_values(
    by=["week day", "day"])

not_regular = departures_2019[(~departures_2019.series.isin(series_2019.series)) & (departures_2019["week day"] == 0)]

not_regular.callsign.unique().shape

plt.rcParams["figure.figsize"] = (20, 3)

to_plot = []
ind = 0
for i in not_regular.callsign.unique()[:40]:
    ice = not_regular[not_regular.callsign == i]
    to_plot = ice["dep time minute"].to_list()
    plt.plot(range(ind, ind + len(to_plot)), to_plot)
    ind += len(to_plot)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()

not_regular.iloc[:40]
# not_regular[not_regular.series==75725] da testare
columns = ['icao24', 'departure', 'arrival', 'callsign', 'day', 'week day', 'series code', 'series',
           'airline', 'is_departure', 'mean_time']
df_double = pd.DataFrame(columns=columns)
for i in not_regular.callsign.unique():
    ddf = not_regular[not_regular.callsign== i]
    mean = ddf["dep time minute"].mean()
    above = ddf[ddf["dep time minute"] > mean + 60]

    above_mean = above["dep time minute"].mean()
    # print(mean)
    # print(above)
    under = ddf[ddf["dep time minute"] < mean - 60]
    under_mean = under["dep time minute"].mean()
    # print(under)

    if under.shape[0] > 5 and above.shape[0] > 5:
        print(i)
        if above[(above["dep time minute"] < above_mean + 30) & (above["dep time minute"] > above_mean - 30)].shape[0] / above.shape[
            0] > 0.7:
            to_append = above[columns[:-2]].iloc[0].to_list() + [True] + [above_mean]
            df_double = df_double.append(dict(zip(columns, to_append)), ignore_index=True)

        if under[(under["dep time minute"] < under_mean + 30) & (under["dep time minute"] > under_mean - 30)].shape[0] / under.shape[
            0] > 0.7:
            to_append = under[columns[:-2]].iloc[0].to_list() + [True] + [under_mean]
            df_double = df_double.append(dict(zip(columns, to_append)), ignore_index=True)

df_double


columns = ['icao24', 'departure', 'arrival', 'callsign', 'day', 'week day', 'series code', 'series',
           'airline', 'is_departure', 'mean_time']
midnight = pd.DataFrame(columns=columns)
for i in not_regular.callsign.unique()[20:25]:
    ddf = not_regular[not_regular.callsign== i]
    mean = ddf["dep time minute"].mean()
    above = ddf[ddf["dep time minute"] > mean + 60]
    above_mean = above["dep time minute"].mean()
    print(above)
    # print(above)
    under = ddf[ddf["dep time minute"] < mean - 60]
    under_mean = under["dep time minute"].mean()
    # print(under_mean)
    if above_mean < 120: #and under.mean >1300:
        print(i)
        new_under = 1400 - under["dep time minute"]
        new_mean = np.append(new_under, above["dep time minute"]).mean()
        new_mean = new_mean if new_mean > 0 else 1400 + new_mean

        to_append = above[columns[:-2]].iloc[0].to_list() + [True] + [above_mean]
        midnight = midnight.append(dict(zip(columns, to_append)), ignore_index=True)



to_plot = []
ind = 0
for i in df_double.callsign.unique()[25:30]:
    ddf = not_regular[not_regular.callsign == i]
    to_plot = ddf["dep time minute"].to_list()
    print(i)
    plt.plot(range(ind, ind + len(to_plot)), to_plot)
    ind += len(to_plot)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()

mid = not_regular[not_regular.callsign== "THY4LE  " ]
mid




"""
2018
"""

df = pd.read_csv('data/summer_2018.csv')
airports = pd.read_csv('data/airports.csv')
departures_2018, arrivals_2018 = f.get_departure_arrivals(df, airports)

series_2018 = f.find_series(departures_2018, day, tol, max_occurence, departure, num_procs)
