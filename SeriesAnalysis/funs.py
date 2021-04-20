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


def compute_mean(time_vect, mean, time_tolerance):
    return time_vect[(time_vect < mean + time_tolerance) & (time_vect > mean - time_tolerance)].mean()


def compute_series(tup):
    series_list, dep, time_tolerance, min_occurrences, is_departure = tup
    columns = ['icao24', 'departure', 'arrival', 'callsign', 'day', 'week day', 'series code', 'series',
               'airline', 'is_departure', 'mean_time', 'start_day', 'len_series']
    series = pd.DataFrame(columns=columns)
    dep_arr = "dep time minute" if is_departure else "arr time minute"

    for s in series_list:
        ser = dep[dep.series == s]
        ser = ser.sort_values(by="day")
        f = ser[dep_arr]
        mean, std = f.mean(), f.std()
        if f[(f < mean + time_tolerance) & (f > mean - time_tolerance)].shape[0] / f.shape[0] > min_occurrences:
            start_day = min(ser.day_num)
            len_series = max(ser.day_num) - start_day + 1
            mean = compute_mean(f, mean, time_tolerance)
            to_append = ser[columns[:-4]].iloc[0].to_list() + [is_departure] + [mean] + [start_day] + [len_series]
            series = series.append(dict(zip(columns, to_append)), ignore_index=True)

    return series

def compute_series_1(tup):
    series_list, dep, time_tolerance, min_occurrences, is_departure = tup
    columns = ['icao24', 'departure', 'arrival', 'callsign', 'day', 'week day', 'series code', 'series',
               'airline', 'is_departure', 'mean_time', 'len_series']
    series = pd.DataFrame(columns=columns)
    dep_arr = "dep time minute" if is_departure else "arr time minute"

    for s in series_list:
        ser = dep[dep.series == s]
        ser = ser.sort_values(by="day")
        mean = ser[dep_arr].mean()
        outs = ser.iloc[:5].copy()
        outs["deviation"] = np.absolute(outs.dep_arr - mean)
        mean = outs[outs.deviation != max(outs.deviation)][dep_arr].mean()
        if ser.shape[0] < 10:
            to_append = ser[columns[:-2]].iloc[0].to_list() + [is_departure] + [mean] + [ser.shape[0]]
            series = series.append(dict(zip(columns, to_append)), ignore_index=True)
        else:
            initial = 0
            final = 6
            for i in range(6, ser.shape[0]-4):
                t = ser.iloc[i][dep_arr].values[0]
                if t < mean + time_tolerance or  t > mean - time_tolerance:
                    final = i
                else:
                    right_mean = ser.iloc[i+1:i+5][dep_arr].mean()
                    if right_mean < mean + time_tolerance or  right_mean > mean - time_tolerance:
                        final = i
                    else:
                        pass
    return series


def find_day_series(df, day, tol, min_occurrence, is_departure: bool, num_procs=1):

    df_day = df[df["week day"] == day].copy()
    date_num = dict(zip(np.sort(df_day.day.unique()), range(len(df_day.day.unique()))))
    df_day["day_num"] = df_day.day.apply(lambda d: date_num[d])
    series = df_day.series.unique()
    len_tot = series.shape[0]
    len_slice = len_tot // num_procs
    split_series = [i * len_slice for i in range(num_procs)] + [len_tot]
    split_flights = tuple([(series[split_series[i]: split_series[i + 1]], df_day[
        df_day.series.isin(series[split_series[i]: split_series[i + 1]])], tol,
                            min_occurrence, is_departure) for i in range(num_procs)])

    pool = Pool(num_procs)
    result = pool.map(compute_series, split_flights)
    final_df = pd.concat(result, ignore_index=True)
    pool.close()
    pool.join()

    return final_df


def find_series(df, days, tol, min_occurrence, is_departure: bool, num_procs=1):

    if type(days) == int:
        final_df = find_day_series(df, days, tol, min_occurrence, is_departure, num_procs)
        final_df["time_request"] = final_df.mean_time.apply(
            lambda t: int(t / 10) * 10 if (t // 5) % 2 == 0 else int(t / 10) * 10 + 5).astype(int)
        return final_df

    columns = ['icao24', 'departure', 'arrival', 'callsign', 'day', 'week day', 'series code', 'series',
               'airline', 'is_departure', 'mean_time']
    final_df = pd.DataFrame(columns=columns)

    for day in days:
        result = find_day_series(df, day, tol, min_occurrence, is_departure, num_procs)
        final_df = pd.concat([final_df, result], ignore_index=True)

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

    departures.callsign = departures.callsign.apply(lambda call: call.replace(" ",""))
    arrivals.callsign = arrivals.callsign.apply(lambda call: call.replace(" ", ""))

    return departures, arrivals


def add_midnights(series, departures, week_day, is_departure):
    columns = ['icao24', 'departure', 'arrival', 'callsign', 'day', 'week day', 'series code', 'series',
               'airline', 'is_departure', 'mean_time', 'start_day', 'len_series']
    not_regular = departures[(~departures.series.isin(series.series)) & (departures["week day"] == week_day)].copy()
    date_num = dict(zip(np.sort(not_regular.day.unique()), range(len(not_regular.day.unique()))))
    not_regular["day_num"] = not_regular.day.apply(lambda d: date_num[d])
    midnight = pd.DataFrame(columns=columns)
    dep_arr = "dep time minute" if is_departure else "arr time minute"
    for i in not_regular.callsign.unique():
        ddf = not_regular[not_regular.callsign == i]
        mean = ddf[dep_arr].mean()
        above = ddf[ddf[dep_arr] > mean + 60]
        above_mean = above[dep_arr].mean()
        under = ddf[ddf[dep_arr] < mean - 60]
        under_mean = under[dep_arr].mean()
        if above_mean > 1320 and under_mean < 120:
            new_above = 1440 - above[dep_arr]
            new_mean = np.append(new_above, above[dep_arr]).mean()
            new_mean = new_mean if new_mean > 0 else 1440 + new_mean
            start_day = min(ddf.day_num)
            len_series = max(ddf.day_num) - start_day + 1
            to_append = above[columns[:-4]].iloc[0].to_list() + [is_departure] + [new_mean] + [start_day] + [len_series]
            midnight = midnight.append(dict(zip(columns, to_append)), ignore_index=True)

    midnight["time_request"] = midnight.mean_time.apply(
        lambda t: int(t / 10) * 10 if (t // 5) % 2 == 0 else int(t / 10) * 10 + 5).astype(int)
    return pd.concat([series, midnight], ignore_index=True)


def plot_series(df, callsigns):
    plt.rcParams["figure.figsize"] = (20, 3)
    ind = 0
    for i in callsigns:
        ddf = df[df.callsign == i]
        to_plot = ddf["dep time minute"].to_list()
        plt.plot(range(ind, ind + len(to_plot)), to_plot)
        ind += len(to_plot)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()


def get_gf_rights(series, departure):
    gf = pd.read_csv("data/test_gf.csv", sep="\t")
    gf_columns = gf.columns.to_list() + ["day"]
    dep_arr = "departure" if departure else "arrival"

    start_minute = np.sort(gf.start.unique())
    end_minute = np.sort(gf.end.unique())

    gf_rights = pd.DataFrame(columns=gf.columns)
    for d in np.sort(series.day.unique()):
        df_day = series[series.day == d]
        for airport in df_day[dep_arr].unique():
            df_airport = df_day[df_day[dep_arr] == airport]
            for airline in df_airport.airline.unique():
                df_airline = df_airport[df_airport.airline == airline]
                for interval in range(len(start_minute)):
                    df_interval = df_airport[(df_airport.mean_time >= start_minute[interval]) & (
                                df_airport.mean_time < end_minute[interval])]
                    if df_interval.shape[0] > 0:
                        to_append = [airport, airline, start_minute[interval], end_minute[interval],
                                     df_interval.shape[0], d]
                        gf_rights = gf_rights.append(dict(zip(gf_columns, to_append)), ignore_index=True)

    return gf_rights




