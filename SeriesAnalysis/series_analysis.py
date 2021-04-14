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
call_icao = "callsign"

day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
tol = [30, 35, 40, 45, 50, 55, 60]
min_oc = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
f.find_series_plot(departures_2019, day, call_icao, num_procs, departure, 2019, save=True)

day = 0
tol = 30
max_occurence = 0.7


europe = pd.read_csv("data/summer_2019.csv")

eu = europe[europe.day==europe.day.unique()[100]]
len(eu["icao24"].unique())

multi = 0
for flight in eu["icao24"].unique():
    d = eu[eu.icao24 == flight]
    pippo = d["arrival"].unique()
    if pippo.shape[0]>1:
        # print(d.icao24.values[0])
        multi += 1
        # if multi >40:
        #     break
multi


#
# ppp = departures_2019.series.unique()
# print(ppp)
#
series_2019 = f.find_series(departures_2019, day, tol, max_occurence, departure, num_procs)





for d in series_2019.day.unique():
    print(series_2019[series_2019.day==d].shape[0])

air = series_2019["airline"].unique()



df = pd.read_csv('data/series_2018.csv')
airports = pd.read_csv('data/airports.csv')
departures_2018, arrivals_2018 = f.get_departure_arrivals(df, airports)

series_2018 = f.find_series(departures_2018, day, tol, max_occurence, departure, num_procs)

# for tw in range(0,60*24+1,60):
#


# def gran_fathers(flight):
#     dep = series_2019[series_2019.callsign == flight]["departure"].unique()[0]
#     arr = series_2019[series_2019.callsign == flight]["arrival"].unique()[0]
#     air = series_2019[series_2019.callsign == flight]["airline"].unique()[0]
#     t = series_2019[series_2019.callsign == flight]["mean_time"].unique()[0]
#     df_fl = series_2018[(series_2018.departure == dep) & (series_2018.arrival == arr) & (series_2018.airline == air)
#                         & (series_2018.mean_time >= t-gf_tolerance) & (series_2018.mean_time <= t+gf_tolerance)]
#     if df_fl.shape[0]>0:
#         return 1
#     else:
#         return 0
#
#
# gf_tolerance= 30
# gf_rules = 0
#
# tt = time.time()
#
# pool = Pool(num_procs)
# result = sum(pool.map(gran_fathers, series_2019.callsign.unique()))
# pool.close()
# pool.join()
#
#
# print(time.time() - tt)
#
lunedi = departures_2019[departures_2019["week day"] == 0]

for flight in lunedi.callsign.unique()[:1000]:
    d = lunedi[lunedi.callsign == flight]
    pippo = d["departure"].unique()
    if pippo.shape[0]>1:
        print(d)