import pandas as pd
from airport import Airport
from flight import Flight
import numpy as np
import datetime
from os import walk

pd.set_option('display.max_columns', None)
import time

import csv


df = pd.read_csv('croatia_2019.csv')

raf = df[['icao24', 'estdepartureairport', 'estarrivalairport', 'callsign','departureairportcandidatescount','arrivalairportcandidatescount', 'day']].copy()
raf.columns = ['icao24', 'departure', 'arrival', 'callsign','candidate dep airports','candidate arr airports', 'day']
raf.sort_values(by="day", inplace=True, ignore_index=True)
raf["week day"] = raf["day"].apply(lambda d: datetime.datetime.fromtimestamp(d).weekday())
raf["day"] = raf["day"].apply(lambda d: str(datetime.datetime.fromtimestamp(d))[:10])



_, _, filenames = next(walk("Coppie_OD/not_moved"))

ddf = pd.read_csv("Coppie_OD/not_moved/"+filenames[0], sep="\t")
final_df = pd.DataFrame(columns=list(ddf.columns))

type_air = []

for file in filenames:
    if file[:2] == "LD" or file[5:7] =="LD":
        dddf = pd.read_csv("Coppie_OD/not_moved/" + file, sep="\t", index_col=False)
        final_df= pd.concat([final_df, dddf], ignore_index=True)

final_df.to_csv("od_cro.csv")


