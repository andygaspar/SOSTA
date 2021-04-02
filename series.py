import pandas as pd

df = pd.read_csv("data/summer_2019.csv")
airports = pd.read_csv("data/airports.csv")
print(df)

final_df = pd.DataFrame(columns=df.columns)
flights_found = []
num_series = 0
num_airport = 0
for airport in airports["airport"]:
    print(num_airport, airport)
    partial = num_series
    air = df[(df["departure"] == airport) ^ (df["arrival"] == airport)]
    num_airport += 1
    for i in range(7):
        hhh = air[air["week day"] == i]
        for cs in hhh["callsign"]:
            same_call = hhh[hhh["callsign"] == cs].copy()
            if same_call.shape[0] > 4:
                if cs not in flights_found:
                    same_call["series"] = num_series
                    final_df = pd.concat([final_df, same_call], ignore_index=True)
                    flights_found.append(cs)
                    num_series += 1
    print(airport, num_series-partial)

print(final_df)
final_df["series"] = final_df["series"].astype(int)
final_df.to_csv("data/series.csv", index=False)