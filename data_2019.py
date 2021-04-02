import pandas as pd
from airport import Airport
from flight import Flight
import numpy as np
import datetime

pd.set_option('display.max_columns', None)




def filter_airports(df_fulvio):
    df = pd.read_csv('data/data_row_fulvio.csv', sep="\t")

    airports = pd.read_csv("data/airports.csv", index_col=None).drop(columns="Unnamed: 0")
    final_df = pd.DataFrame(columns=df.columns)
    i = 0
    for airport in airports["airport"]:
        print(airport, i)
        i += 1
        temp = df[(df["estdepartureairport"] == airport) ^ (df["estarrivalairport"] == airport)]
        final_df = pd.concat([final_df, temp])

    # final_df.to_csv("europe_2019.csv")
    return final_df


def rename(df_in):
    renamed = df_in[["icao24", "firstseen", "estdepartureairport", "lastseen", "estarrivalairport", "callsign",
                     "estdepartureairporthorizdistance", "estdepartureairportvertdistance",
                     "estarrivalairporthorizdistance",
                     "estarrivalairportvertdistance", "departureairportcandidatescount",
                     "arrivalairportcandidatescount",
                     "day"]].copy()
    renamed.columns = ['icao24', "dep time", 'departure', "arr time", 'arrival', 'callsign', 'dep dist', 'dep alt',
                       'arr dist', 'arr alt', 'candidate dep airports', 'candidate arr airports', 'day']
    return renamed


def day_converter(df_in):
    df_in.sort_values(by="day", inplace=True, ignore_index=True)
    df_in["week day"] = df_in["day"].apply(lambda d: datetime.datetime.fromtimestamp(d).weekday())
    df_in["day"] = df_in["day"].apply(lambda d: str(datetime.datetime.fromtimestamp(d))[:10])
    return df_in

df = pd.read_csv("data/europe_2019.csv")
print(df.shape)
df = df[(pd.to_datetime(df["day"]) >= datetime.datetime(2019, 3, 21)) &
        (pd.to_datetime(df["day"]) < datetime.datetime(2019, 10, 27))]
print(df.shape)
df.to_csv("data/summer_2019.csv", index_label=False, index=False)

#31 3 2019
# p = datetime.strptime("2019-01-01", '%Y-%m-%d').date()
# print(p)