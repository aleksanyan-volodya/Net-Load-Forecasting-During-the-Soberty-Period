import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import acf
from pandas.plotting import lag_plot


Data0 = pd.read_csv(
    "Data/train.csv",
    parse_dates=["Date"]
)
Data1 = pd.read_csv(
    "Data/test.csv",
    parse_dates=["Date"]
)

""""
# Basic descriptions
Data0.describe()
Data0["Date"].min(), Data0["Date"].max()
Data1["Date"].min(), Data1["Date"].max()

Data0.columns
Data0[["Date", "WeekDays"]].head()
Data0["Date"].dt.day_name().head(7)



#####################
# trend plots
plt.figure()
plt.plot(Data0["Date"], Data0["Net_demand"])
plt.xlim(Data0["Date"].min(), Data1["Date"].max())
plt.show()

fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(Data0["Date"], Data0["Load"])
axes[1].plot(Data0["Date"], Data0["Solar_power"])
axes[2].plot(Data0["Date"], Data0["Wind_power"])
plt.show()

plt.figure()
plt.plot(Data0["Date"], Data0["Load"], label="Load")
plt.plot(Data0["Date"], Data0["Wind_power"], label="Wind")
plt.plot(Data0["Date"], Data0["Solar_power"], label="Solar")
plt.legend()
plt.show()

# stat summary
Data0["Load"].mean()
Data0["Wind_power"].mean()
Data0["Solar_power"].mean()

# histogram 
plt.hist(Data0["Net_demand"], bins=100)
plt.show()

K = 7
smooth = Data0["Net_demand"].rolling(K, center=True).mean()

plt.plot(Data0["Date"], Data0["Net_demand"])
plt.plot(Data0["Date"], smooth, color="red", linewidth=2)
plt.show()


##################### yearly cycle
sel = Data0["Year"] == 2021
plt.plot(Data0.loc[sel, "Date"], Data0.loc[sel, "Net_demand"])
plt.show()

plt.scatter(
    Data0["toy"],
    Data0["Net_demand"],
    alpha=0.3,
    s=10
)
plt.show()


# Monthly boxplots 
fig, axes = plt.subplots(3, 1, sharex=True)

sns.boxplot(x="Month", y="Net_demand", data=Data0, ax=axes[0])
sns.boxplot(x="Month", y="Solar_power", data=Data0, ax=axes[1])
sns.boxplot(x="Month", y="Wind_power", data=Data0, ax=axes[2])

plt.show()


###################### weekly cycle
sel = (Data0["Month"] == 6) & (Data0["Year"] == 2021)
plt.plot(Data0.loc[sel, "Date"], Data0.loc[sel, "Net_demand"])
plt.show()

sns.boxplot(x="WeekDays", y="Net_demand", data=Data0)
plt.show()


plt.scatter(Data0["Load.1"], Data0["Load"], s=10)
plt.show()

Data0["Load.1"].corr(Data0["Load"])


# Lag scatter
plt.scatter(Data0["Load.1"], Data0["Load"], s=10)
plt.show()

Data0["Load.1"].corr(Data0["Load"])

# ACF
lags = 7 * 10
acf_vals = acf(Data0["Load"], nlags=lags, fft=True)

plt.stem(range(len(acf_vals)), acf_vals, use_line_collection=True)
plt.ylim(0, 1)
plt.show()

# Temperature effect
fig, ax1 = plt.subplots()

ax1.plot(Data0["Date"], Data0["Net_demand"], color="black")
ax2 = ax1.twinx()
ax2.plot(Data0["Date"], Data0["Temp"], color="red")

ax2.set_ylabel("Temperature", color="red")
plt.show()

plt.scatter(Data0["Temp"], Data0["Net_demand"], alpha=0.25, s=10)
plt.show()


#### Wind effect

plt.figure()
plt.plot(Data0["Date"], Data0["Wind"])
plt.show()

plt.scatter(Data0["Wind"], Data0["Wind_power"], alpha=0.25, s=10)
plt.show()

K = 7 * 4
smooth_net = Data0["Net_demand"].rolling(K, center=True).mean()
smooth_wind = Data0["Wind"].rolling(K, center=True).mean()

fig, ax1 = plt.subplots()
ax1.plot(Data0["Date"], smooth_net)
ax2 = ax1.twinx()
ax2.plot(Data0["Date"], smooth_wind, color="red")
plt.show()


##################### solar / nebulosity effect
plt.plot(Data0["Date"], Data0["Nebulosity"])
plt.show()

plt.scatter(Data0["Nebulosity"], Data0["Solar_power"], alpha=0.25, s=10)
plt.show()

sel = Data0["Date"].dt.year >= 2018
Data0.loc[sel, "Nebulosity"].corr(Data0.loc[sel, "Solar_power"])

#### lagged net demand
plt.scatter(Data0["Net_demand.1"], Data0["Net_demand"], s=10)
plt.show()

Data0["Net_demand.1"].corr(Data0["Net_demand"])

### Holiday effect
sns.boxplot(x="Christmas_break", y="Net_demand",
            data=Data0[Data0["DLS"] == 0])
plt.show()

sns.boxplot(x="DLS", y="Load", data=Data0)
plt.show()


### Train/test comparison

plt.hist(Data0["Temp"], bins=50, alpha=0.6, label="Train")
plt.hist(Data1["Temp"], bins=50, alpha=0.5, label="Test")
plt.legend()
plt.show()

"""