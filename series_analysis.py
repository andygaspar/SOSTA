# %%

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time
import multiprocess
from multiprocess import Pool

num_procs = multiprocess.cpu_count()
print(num_procs)

# %%

df = pd.read_csv('data/series.csv')
airports = pd.read_csv('data/airports.csv')
departures = df[df["departure"].isin(airports["airport"])].copy().drop_duplicates()
arrivals = df[df["arrival"].isin(airports["airport"])].copy().drop_duplicates()
time_ = departures["dep time"].apply(lambda d: datetime.datetime.fromtimestamp(d).time() if not np.isnan(d) else "NaN")
departures["dep time"] = time_
departures["dep time minute"] = time_.apply(lambda t: np.round(t.hour * 60 + t.minute + t.second * 0.1)).astype(int)

time_ = arrivals["arr time"].apply(lambda d: datetime.datetime.fromtimestamp(d).time() if not np.isnan(d) else "NaN")
arrivals["arr time"] = time_
arrivals["arr time minute"] = time_.apply(lambda t: np.round(t.hour * 60 + t.minute + t.second * 0.1)).astype(int)

# %%

departure_day = departures[departures["week day"] == 0]
departure_day

# %%

flights_departure = departure_day["callsign"].unique()
flights_arrival = arrivals["callsign"].unique()
flights_departure.shape


# %%

def compute(tup):
    fl_dep, dep, time_tolerance, min_occourrency = tup
    regular = 0
    for flight in fl_dep:
        f = dep[dep["callsign"] == flight]["dep time minute"]
        mean, std = f.mean(), f.std()
        if f[(f < mean + time_tolerance) & (f > mean - time_tolerance)].shape[0] / f.shape[0] > min_occourrency:
            regular += 1
    return regular


# %%

time_tolerance = 45
min_occourrency = 0.75

len_slice = flights_departure.shape[0] // num_procs
split_fl = [i * len_slice for i in range(num_procs)] + [flights_departure.shape[0]]

split_flights = tuple([(flights_departure[split_fl[i]:split_fl[i + 1]],
                        departure_day[departure_day["callsign"].isin(flights_departure[split_fl[i]:split_fl[i + 1]])],
                        time_tolerance, min_occourrency) for i in range(num_procs)])

# %%

t = time.time()

pool = Pool(num_procs)
reg = sum(pool.map(compute, split_flights))
pool.close()
pool.join()
print(time.time() - t)
print(reg)

# %%

from itertools import product

tol = [30, 35, 40, 45, 50, 55, 60]
min_oc = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
grid = list(product(tol, min_oc))

# %%

fls = []
for point in grid:
    print(point)
    time_tolerance = point[0]
    min_occourrency = point[1]
    split_flights = tuple([(flights_departure[split_fl[i]:split_fl[i + 1]],
                            departures[departures["callsign"].isin(flights_departure[split_fl[i]:split_fl[i + 1]])],
                            time_tolerance, min_occourrency) for i in range(num_procs)])
    pool = Pool(num_procs)
    fls.append(sum(pool.map(compute, split_flights)))
    pool.close()
    pool.join()

# %%

fls = np.array(fls)
plt.rcParams["figure.figsize"] = (30, 25)
plt.rcParams["font.size"] = 25
points = np.array(grid).T
plt.xticks(tol)
plt.yticks(min_oc)
plt.xlabel("TIME TOLERANCE")
plt.xlim(25, 65)
plt.ylim(0.65, 1)
plt.ylabel("OCCURRENCE TOLERANCE")
plt.title
for i in range(len(grid)):
    plt.annotate(fls[i], (grid[i][0] - 1, grid[i][1] - 0.001), color='white')
plt.scatter(points[0], points[1], s=fls - 5000)
plt.savefig("tol.png")

# %% md

## Per week day

# %%

day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
for j in range(7):
    departures_day = arrivals[arrivals["week day"] == j]
    flights_departure = departures_day["callsign"].unique()
    print(flights_departure.shape)
    len_slice = flights_departure.shape[0] // num_procs
    split_fl = [i * len_slice for i in range(num_procs)] + [flights_departure.shape[0]]
    fls = []
    t = time.time()
    for point in grid:
        partial_time = time.time()
        print(point)
        time_tolerance = point[0]
        min_occourrency = point[1]
        split_flights = tuple([(flights_departure[split_fl[i]:split_fl[i + 1]], departures_day[
            departures_day["callsign"].isin(flights_departure[split_fl[i]:split_fl[i + 1]])], time_tolerance,
                                min_occourrency) for i in range(num_procs)])
        pool = Pool(num_procs)
        fls.append(sum(pool.map(compute, split_flights)))
        pool.close()
        pool.join()
        print("paritial", time.time() - partial_time)

    print("time total: ", time.time() - t)

    fls = np.array(fls)
    plt.rcParams["figure.figsize"] = (30, 25)
    plt.rcParams["font.size"] = 25
    points = np.array(grid).T
    plt.xticks(tol)
    plt.yticks(min_oc)
    plt.xlabel("TIME TOLERANCE")
    plt.xlim(25, 65)
    plt.ylim(0.65, 1)
    plt.ylabel("OCCURRENCE TOLERANCE")
    plt.title(day[j])
    for i in range(len(grid)):
        plt.annotate(fls[i], (grid[i][0], grid[i][1]), color='white', horizontalalignment='center',
                     verticalalignment='center')
    plt.scatter(points[0], points[1], s=fls * 1.5)
    plt.savefig("plots/arrivals_" + day[j] + ".png")
    plt.cla()
    plt.clf()
    plt.close()
    print(j)

# %%

arrivals


# %% md

## Global std

# %%

def compute_std(tup):
    fl_dep, dep, tolerance = tup
    std_list = []

    for flight in fl_dep:
        f = dep[dep["callsign"] == flight]["dep time minute"]
        mean, std = f.mean(), f.std()
        std_list.append(std)

    return std_list


# %%


t = time.time()

pool = Pool(num_procs)
reg = pool.map(compute_std, split_fl)
pool.close()
pool.join()
print(time.time() - t)

