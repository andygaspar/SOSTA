import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocess
from multiprocess import Pool
from SeriesAnalysis import funs as f

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

num_procs = multiprocess.cpu_count()

df = pd.read_csv('data/series_2019.csv')
airports = pd.read_csv('data/airports.csv')

departures_2019, arrivals_2019 = f.get_departure_arrivals(df, airports)

departure = True
arrival = False

# day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# tol = [30, 35, 40, 45, 50, 55, 60]
# min_oc = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
# f.find_series_plot(departures_2019, day, call_icao, num_procs, departure, 2019, save=True)

day = 0
tol = 30
max_occurence = 0.7
series_2019_departures = f.find_series(departures_2019, day, tol, max_occurence, departure, num_procs)
series_2019_departures = f.add_midnights(series_2019_departures, departures_2019, day, departure)
series_2019_arrival = f.find_series(arrivals_2019, day, tol, max_occurence, arrival, num_procs)
series_2019_arrival = f.add_midnights(series_2019_arrival, arrivals_2019, day, arrival)



date_to_num = dict(zip(np.sort(series_2019_departures.day.unique()), range(len(series_2019_departures.day.unique()))))
num_to_date = dict(zip(range(len(series_2019_departures.day.unique())), np.sort(series_2019_departures.day.unique())))

"""
fixing mismatching dep arr
"""

dep = series_2019_departures[series_2019_departures.arrival.isin(airports.airport)]
arr = series_2019_arrival[series_2019_arrival.departure.isin(airports.airport)]

not_ok_arr = dep[~dep.callsign.isin(arr.callsign.to_list())]
not_ok_dep = arr[~arr.callsign.isin(dep.callsign.to_list())]

d = departures_2019[departures_2019.callsign.isin(not_ok_arr.callsign.to_list())]
a = arrivals_2019[arrivals_2019.callsign.isin(not_ok_arr.callsign.to_list())]

columns = ['icao24', 'departure', 'arrival', 'callsign', 'day', 'week day', 'series code', 'series',
           'airline', 'is_departure', 'mean_time', 'start_day', 'len_series']

add_to_departure = pd.DataFrame(columns=columns)
for call in not_ok_dep.callsign.unique():
    dc = departures_2019[departures_2019.callsign == call]
    for d in dc.departure.unique():
        dd = dc[dc.departure == d]
        for a in dd.arrival.unique():
            da = dd[dd.arrival == a]
            line = series_2019_arrival[series_2019_arrival.callsign == call].copy()
            print(line)
            line.is_departure = True
            line.mean_time = f.compute_mean(da["dep time minute"], da["dep time minute"].mean(), 30)
            add_to_departure = pd.concat([add_to_departure, line])

add_to_departure = add_to_departure.drop_duplicates(["series"])
add_to_departure = add_to_departure[add_to_departure.departure.isin(airports.airport.to_list())]
series_2019_departures = pd.concat([series_2019_departures, add_to_departure])

add_to_arrival = pd.DataFrame(columns=columns)
for call in not_ok_arr.callsign.unique():
    dc = arrivals_2019[arrivals_2019.callsign == call]
    for d in dc.departure.unique():
        dd = dc[dc.departure == d]
        for a in dd.arrival.unique():
            da = dd[dd.arrival == a]
            line = series_2019_departures[series_2019_departures.callsign == call].copy()
            print(line)
            line.is_departure = False
            line.mean_time = f.compute_mean(da["arr time minute"], da["arr time minute"].mean(), 30)
            add_to_arrival = pd.concat([add_to_arrival, line])

add_to_arrival = add_to_arrival.drop_duplicates(["series"])
add_to_arrival = add_to_arrival[add_to_arrival.arrival.isin(airports.airport.to_list())]
series_2019_arrival = pd.concat([series_2019_arrival, add_to_arrival])

series_2019_departures.series = range(series_2019_departures.shape[0])
series_2019_arrival.series = range(series_2019_departures.shape[0],series_2019_departures.shape[0]+series_2019_arrival.shape[0])


# final day all series
all_series = pd.concat([series_2019_departures, series_2019_arrival], ignore_index=True)
all_series.time_request = all_series.time_request.astype(int)
all_series.series.unique().shape

"""
day computation
"""

day = 0
first_day = all_series[all_series.start_day == 0]
dep_day = departures_2019[departures_2019.day == num_to_date[0]]
arr_day = arrivals_2019[arrivals_2019.day == num_to_date[0]]
aircraft = pd.read_csv("data/aircraft/aircraftDatabase.csv", dtype="str")

columns=["series", "airline", "Flow", "airport", "callsign", "icao24", "SlotDate", "TimeRequested",
                                 "TurnAround", "Match", "aircraftType"]

day_0_df = pd.DataFrame(columns=columns)
for icao24 in first_day.icao24.unique():
    df_icao = first_day[first_day.icao24 == icao24].sort_values(by="time_request")
    turn_around = False
    for series in df_icao.series:
        df_is = df_icao[df_icao.series == series]
        is_departure = df_is.is_departure.unique()[0]
        try:
            aircraft_type = aircraft[aircraft.icao24 == icao24]["typecode"].unique()[0]
        except:
            aircraft_type = "unknown"
        match = first_day[(first_day.callsign == df_is.callsign.unique()[0]) &
                          (first_day.departure == df_is.departure.unique()[0]) &
                          (first_day.series != df_is.series.unique()[0])]["series"].values
        match = [-1] if len(match) == 0 else list(match)
        line = [series] + [df_is["airline"].unique()[0]] + ["D" if is_departure else "A"] + \
               [df_is.departure.unique()[0] if is_departure else df_is.arrival.unique()[0]] + \
                [df_is.callsign.unique()[0]] + [df_is.icao24.unique()[0]] + [day] + [df_is.time_request.unique()[0]] \
               + [turn_around] + match + [aircraft_type]
        day_0_df = day_0_df.append(dict(zip(columns,line)), ignore_index=True)
        turn_around = True



"""
not regular
"""

not_regular = departures_2019[
    (~departures_2019.series.isin(series_2019_departures.series)) & (departures_2019["week day"] == 0)]

date_num = dict(zip(np.sort(series_2019_departures.day.unique()), range(len(series_2019_departures.day.unique()))))

columns = ['icao24', 'departure', 'arrival', 'callsign', 'day', 'week day', 'series code', 'series',
           'airline', 'is_departure', 'mean_time']
df_double = pd.DataFrame(columns=columns)
for i in not_regular.callsign.unique():
    ddf = not_regular[not_regular.callsign == i]
    mean = ddf["dep time minute"].mean()
    above = ddf[ddf["dep time minute"] > mean + 60]

    above_mean = above["dep time minute"].mean()
    # print(mean)
    # print(above)
    under = ddf[ddf["dep time minute"] < mean - 60]
    under_mean = under["dep time minute"].mean()
    # print(under)

    if under.shape[0] > 4 and above.shape[0] > 4:
        if above[(above["dep time minute"] < above_mean + 60) & (above["dep time minute"] > above_mean - 60)].shape[0] / \
                above.shape[
                    0] > 0.7:
            to_append = above[columns[:-2]].iloc[0].to_list() + [True] + [above_mean]
            df_double = df_double.append(dict(zip(columns, to_append)), ignore_index=True)

        if under[(under["dep time minute"] < under_mean + 60) & (under["dep time minute"] > under_mean - 60)].shape[0] / \
                under.shape[
                    0] > 0.7:
            to_append = under[columns[:-2]].iloc[0].to_list() + [True] + [under_mean]
            df_double = df_double.append(dict(zip(columns, to_append)), ignore_index=True)

df_not_double = not_regular[~not_regular.callsign.isin(df_double.callsign)]
df_not_double.callsign.unique().shape

f.plot_series(not_regular, df_not_double.callsign.unique()[:70])

"""
2018
"""

df = pd.read_csv('data/series_2018.csv')
airports = pd.read_csv('data/airports.csv')
departures_2018, arrivals_2018 = f.get_departure_arrivals(df, airports)

series_2018_departure = f.find_series(departures_2018, day, tol, max_occurence, departure, num_procs)
series_2018_arrival = f.find_series(arrivals_2018, day, tol, max_occurence, arrival, num_procs)
series_2018_departure = f.add_midnights(series_2018_departure, departures_2018, day, departure)
series_2019_arrival = f.add_midnights(series_2018_arrival, arrivals_2018, day, arrival)

date_to_num = dict(zip(np.sort(series_2018_departure.day.unique()), range(len(series_2018_departure.day.unique()))))
num_to_date = dict(zip(range(len(series_2018_departure.day.unique())), np.sort(series_2018_departure.day.unique())))

first_day_gf = series_2018_departure[series_2018_departure.day == num_to_date[0]]
gf = f.get_gf_rights(first_day_gf, departure)


def get_new_entries(gf, next_year_series, departure):
    new_entries = pd.DataFrame(columns=["airport", "airline"])
    for airport in gf.airport.unique():
        dep_arr = "departure" if departure else "arrival"
        df_a = next_year_series[next_year_series[dep_arr] == airport]
        in_gf = gf[gf.airport == airport].airline.to_list()
        new = df_a[~df_a.airline.isin(in_gf)].airline.unique()
        new_entries = pd.concat([new_entries, pd.DataFrame(
            {"airport": [airport]*len(new), "airline": new})], ignore_index=True)
    return new_entries


new_entries = get_new_entries(gf, first_day, True)