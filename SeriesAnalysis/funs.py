import copy
import datetime
import time
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocess import Pool


def compute(tup):
    series_list, df, time_tolerance, min_occurrences, call_icao, is_departure = tup
    regular = 0
    dep_arr = "dep time minute" if is_departure else "arr time minute"
    for ser in series_list:
        f = df[df.series == ser][dep_arr]
        mean, std = f.mean(), f.std()
        if f[(f < mean + time_tolerance) & (f > mean - time_tolerance)].shape[0] / f.shape[0] > min_occurrences:
            regular += 1
    return regular


def find_series_plot(df, day, call_icao, num_procs, is_departure: bool, year, save=False):
    tol = [30, 35, 40, 45, 50, 55, 60]
    min_oc = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    grid = list(product(tol, min_oc))

    for j in range(7):
        df_day = df[df["week day"] == j]
        series = df_day.series.unique()
        len_tot = series.shape[0]
        len_slice = len_tot // num_procs
        split_series = [i * len_slice for i in range(num_procs)] + [len_tot]
        fls = []
        t = time.time()
        for point in grid:
            partial_time = time.time()
            print(point)
            time_tolerance = point[0]
            min_occurrence = point[1]
            split_flights = tuple([(series[split_series[i]: split_series[i + 1]], df_day[
                df_day.series.isin(series[split_series[i]: split_series[i + 1]])], time_tolerance,
                                    min_occurrence, call_icao, is_departure) for i in range(num_procs)])
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
        if save:
            dep_arr = "departures" if is_departure else "arrivals"
            plt.savefig("plots/"+dep_arr+"_"+str(year)+"_"+day[j]+".png")
            plt.cla()
            plt.clf()
            plt.close()
        else:
            plt.show()
        print(j)


def compute_series(tup):
    series_list, dep, time_tolerance, min_occurrences, is_departure = tup
    columns = ['icao24', 'departure', 'arrival', 'callsign', 'day', 'week day', 'series code', 'series',
               'airline', 'is_departure', 'mean_time']
    series = pd.DataFrame(columns=columns)
    dep_arr = "dep time minute" if is_departure else "arr time minute"

    for s in series_list:
        ser = dep[dep.series == s]
        ser = ser.sort_values(by="day")
        f = ser[dep_arr]
        mean, std = f.mean(), f.std()
        if f[(f < mean + time_tolerance) & (f > mean - time_tolerance)].shape[0] / f.shape[0] > min_occurrences:
            to_append = ser[columns[:-2]].iloc[0].to_list() + [is_departure] + [mean]
            series = series.append(dict(zip(columns, to_append)), ignore_index=True)

    return series


def find_series(df, day, tol, min_occurrence, is_departure: bool, num_procs=1):

    df_day = df[df["week day"] == day]
    series = df_day.series.unique()
    len_tot = series.shape[0]
    len_slice = len_tot // num_procs
    split_series = [i * len_slice for i in range(num_procs)] + [len_tot]
    partial_time = time.time()

    split_flights = tuple([(series[split_series[i]: split_series[i + 1]], df_day[
        df_day.series.isin(series[split_series[i]: split_series[i + 1]])], tol,
                            min_occurrence, is_departure) for i in range(num_procs)])

    pool = Pool(num_procs)
    result = pool.map(compute_series, split_flights)
    final_df = pd.concat(result, ignore_index=True)
    pool.close()
    pool.join()
    print("paritial", time.time() - partial_time)

    return final_df


def get_departure_arrivals(df, airports):
    departures = df[df["departure"].isin(airports["airport"])].copy().drop_duplicates()
    arrivals = df[df["arrival"].isin(airports["airport"])].copy().drop_duplicates()

    time_ = departures["dep time"].apply(
        lambda d: datetime.datetime.fromtimestamp(d).time() if not np.isnan(d) else "NaN")
    departures["dep time"] = time_
    departures["dep time minute"] = time_.apply(lambda t: np.round(t.hour * 60 + t.minute + t.second * 0.1)).astype(int)

    time_ = arrivals["arr time"].apply(
        lambda d: datetime.datetime.fromtimestamp(d).time() if not np.isnan(d) else "NaN")
    arrivals["arr time"] = time_
    arrivals["arr time minute"] = time_.apply(lambda t: np.round(t.hour * 60 + t.minute + t.second * 0.1)).astype(int)

    return departures, arrivals



