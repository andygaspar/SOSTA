import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing
import time


def compute_series(tup):
    df_in, airp, num, callsign_or_icao = tup
    df_to_append = pd.DataFrame(columns=df_in.columns)

    for i in range(7):
        for key in ["departure", "arrival"]:
            hhh = df_in[df_in[key] == airp]
            hhh = hhh[hhh["week day"] == i]
            for cs in hhh[callsign_or_icao].unique():
                same_call = hhh[hhh[callsign_or_icao] == cs]
                other_key = "arrival" if key == "departure" else "departure"
                for other_orig_dest in same_call[other_key].unique():
                    if same_call[same_call[other_key] == other_orig_dest].shape[0] > 4:
                        series = same_call[same_call[other_key] == other_orig_dest].copy().drop_duplicates()
                        series["series code"] = series[key].values[0] + "_" + series[other_key].values[0] \
                                                + "_" + cs + "_" + str(i)
                        df_to_append = pd.concat([df_to_append, series], ignore_index=True)
    print(num)
    return df_to_append


def find_series(df_summer, year, callsign_or_icao="callsign", save=False):

    airports = df_summer

    split_df = []
    ind = 0
    for airport in airports["airport"]:
        split_df.append((df_summer[(df_summer["departure"] == airport) ^ (df_summer["arrival"] == airport)].copy(),
                         airport, ind, callsign_or_icao))
        ind += 1

    split_df = tuple(split_df)
    num_procs = multiprocessing.cpu_count()
    t = time.time()
    pool = Pool(num_procs)
    result = pool.map(compute_series, split_df)
    final_df = pd.concat(result, ignore_index=True)
    pool.close()
    pool.join()
    final_df = final_df[final_df.duplicated(subset=["departure", "arrival", "week day", "day", "callsign", "dep time"])]
    final_df["series"] = np.zeros(final_df.shape[0])
    ind = 0
    for series_code in final_df["series code"].unique():
        final_df.loc[final_df["series code"] == series_code, "series"] = ind
        ind += 1

    final_df["series"] = final_df["series"].astype(int)
    final_df["airport"] = final_df["callsign"].apply(lambda call: call[:3])
    print(time.time() - t)

    if save:
        final_df.to_csv("data/series_"+str(year)+".csv", index=False)
    else:
        print(final_df)

# df = pd.read_csv("data/summer_2018.csv")
