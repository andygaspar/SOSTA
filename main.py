import pandas as pd
from airport import Airport
from flight import Flight
import numpy as np
import datetime
from os import walk

pd.set_option('display.max_columns', None)

# t = time.time()
df = pd.read_csv('data/data_row_fulvio.csv', sep="\t")

airports = pd.read_csv("data/airports.csv", index_col=None).drop(columns="Unnamed: 0")
print(airports)

final_df = pd.DataFrame(columns=df.columns)

i = 0

for airport in airports["airport"]:
    print(airport, i)
    i += 1
    temp = df[(df["estdepartureairport"] == airport) ^ (df["estarrivalairport"] == airport)]
    final_df = pd.concat([final_df, temp])

final_df.to_csv("europe.csv")


#
#
# df = pd.read_csv('croatia_2019.csv')
#
# raf = df[['icao24', 'estdepartureairport', 'estarrivalairport', 'callsign','departureairportcandidatescount','arrivalairportcandidatescount', 'day']].copy()
# raf.columns = ['icao24', 'departure', 'arrival', 'callsign','candidate dep airports','candidate arr airports', 'day']
# raf.sort_values(by="day", inplace=True, ignore_index=True)
# raf["week day"] = raf["day"].apply(lambda d: datetime.datetime.fromtimestamp(d).weekday())
# raf["day"] = raf["day"].apply(lambda d: str(datetime.datetime.fromtimestamp(d))[:10])








