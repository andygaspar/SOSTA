import pandas as pd
from multiprocessing import Pool
import time


def compute_series(tup):
    df_in, airp, num = tup
    df_to_append = pd.DataFrame(columns=df_in.columns)
    num_series = 0
    for i in range(7):
        hhh = df_in[df_in["week day"] == i]
        for cs in hhh["callsign"].unique():
            same_call = hhh[hhh["callsign"] == cs].copy()
            if same_call.shape[0] > 4:
                same_call["series"] = num_series
                df_to_append = pd.concat([df_to_append, same_call], ignore_index=True)
                num_series += 1
    print(num)
    return df_to_append


if __name__ == '__main__':
    df = pd.read_csv("data/summer_2019.csv")
    airports = pd.read_csv("data/airports.csv")

    split_df = []
    ind = 0
    for airport in airports["airport"]:
        split_df.append((df[(df["departure"] == airport) ^ (df["arrival"] == airport)].copy(), airport, ind))
        ind += 1
    split_df = tuple(split_df)
    num_procs = 16
    t = time.time()
    pool = Pool(num_procs)
    result = pool.map(compute_series, split_df)
    final_df = pd.concat(result, ignore_index=True)
    print(time.time()-t)
    pool.close()
    pool.join()
    final_df["series"] = final_df["series"].astype(int)
    print(final_df)

    final_df.to_csv("data/series_1.csv", index=False)