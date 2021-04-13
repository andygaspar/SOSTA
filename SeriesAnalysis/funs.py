import time
from itertools import product

import numpy as np
import pandas as pd
import matplotlib as plt
from multiprocess import Pool

def compute(tup):
    fl_dep, dep, time_tolerance, min_occurrences, call_icao, dep_arrive = tup
    regular = 0
    for flight in fl_dep:
        f = dep[dep[call_icao] == flight][dep_arrive]
        mean, std = f.mean(), f.std()
        if f[(f < mean + time_tolerance) & (f > mean - time_tolerance)].shape[0] / f.shape[0] > min_occurrences:
            regular += 1
    return regular


def find_series_plot(df, call_icao, num_procs, dep_arrive, save=False):
    tol = [30, 35, 40, 45, 50, 55, 60]
    min_oc = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    grid = list(product(tol, min_oc))
    day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for j in range(7):
        df_day = df[df["week day"] == j]
        flights_departure = df_day[call_icao].unique()
        print(flights_departure.shape)
        len_slice = flights_departure.shape[0] // num_procs
        split_fl = [i * len_slice for i in range(num_procs)] + [flights_departure.shape[0]]
        fls = []
        t = time.time()
        for point in grid:
            partial_time = time.time()
            print(point)
            time_tolerance = point[0]
            min_occurrence = point[1]
            split_flights = tuple([(flights_departure[split_fl[i]:split_fl[i + 1]], df_day[
                df_day[call_icao].isin(flights_departure[split_fl[i]:split_fl[i + 1]])], time_tolerance,
                                    min_occurrence, call_icao, dep_arrive) for i in range(num_procs)])
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
            plt.savefig("../plots/departures_icao"+day[j]+".png")
            plt.cla()
            plt.clf()
            plt.close()
        else:
            plt.show()

        print(j)