import pandas as pd
from multiprocessing import Pool


def compute_series(tup):
    airp, df_in, proc_num = tup
    df_to_append = pd.DataFrame(columns=df_in.columns)
    num_series = 0
    num_airport = 0
    len_airp = airp.shape[0]
    for airport in airp:
        air = df_in[(df_in["departure"] == airport) ^ (df_in["arrival"] == airport)]
        for i in range(7):
            print(proc_num, len_airp,":", num_airport, i)
            hhh = air[air["week day"] == i]
            for cs in hhh["callsign"]:
                same_call = hhh[hhh["callsign"] == cs].copy()
                if same_call.shape[0] > 4:
                    same_call["series"] = num_series
                    df_to_append = pd.concat([df_to_append, same_call], ignore_index=True)
                    num_series += 1
        num_airport += 1
    print("done")
    return df_to_append


if __name__ == '__main__':
    df = pd.read_csv("data/summer_2019.csv")
    airports = pd.read_csv("data/airports.csv")

    num_procs = 16
    pool = Pool(num_procs)
    len_slice = airports.shape[0]//num_procs
    split_df_idx = [i*len_slice for i in range(num_procs)] + [airports.shape[0]]


    split_df = []
    for i in range(num_procs):
        partial_idx = airports["airport"][split_df_idx[i]:split_df_idx[i+1]]
        print(partial_idx)
        split_df.append((partial_idx.copy(),
                         df[(df["departure"].isin(partial_idx)) ^ (df["arrival"].isin(partial_idx))].copy(), i))

    final_df = pd.concat(pool.map(compute_series, split_df))
    pool.close()
    pool.join()
    print(final_df)
    final_df["series"] = final_df["series"].astype(int)
    final_df.to_csv("data/series_1.csv", index=False)