import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time
import multiprocess
from multiprocess import Pool
from itertools import product
from SeriesAnalysis import funs as f

num_procs = multiprocess.cpu_count()

df = pd.read_csv('../data/series_2019.csv')
airports = pd.read_csv('../data/airports.csv')
departures = df[df["departure"].isin(airports["airport"])].copy().drop_duplicates()
arrivals = df[df["arrival"].isin(airports["airport"])].copy().drop_duplicates()
time_ = departures["dep time"].apply(lambda d: datetime.datetime.fromtimestamp(d).time() if not np.isnan(d) else "NaN")
departures["dep time"] = time_
departures["dep time minute"] = time_.apply(lambda t: np.round(t.hour * 60 + t.minute + t.second * 0.1)).astype(int)

time_ = arrivals["arr time"].apply(lambda d: datetime.datetime.fromtimestamp(d).time() if not np.isnan(d) else "NaN")
arrivals["arr time"] = time_
arrivals["arr time minute"] = time_.apply(lambda t: np.round(t.hour * 60 + t.minute + t.second * 0.1)).astype(int)

flights_departure = departures["callsign"].unique()
flights_arrival = arrivals["callsign"].unique()


dep_arr = "dep time minute"
call_icao = "callsign"


f.find_series_plot(departures, call_icao, num_procs, dep_arr, save=False)






