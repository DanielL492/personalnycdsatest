import pandas as pd

df = pd.read_csv("pm25_temperature_monthly_graphs_through_sep_2026/nyc_pm25_temperature_monthly_through_sep_2026.csv")

print(df.isna().sum())

print(df[df["pm25_mean_ug_m3"].isna()])