import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.ticker as mticker
from scipy import stats
import pandas as pd

def plot(filename):
    df = pd.read_csv("data/" + filename + ".dat", parse_dates=[1], header=None, delimiter="	", comment="#")

    fig, ax = plt.subplots()

    fig.set_size_inches(8,6)

    frmt = mdates.DateFormatter('%H:%M:%S')


    timeInSeconds = []
    for i in range(27):
        date = df[1][i]
        timeInSeconds.append(date.second+(date.minute+date.hour*60)*60)

    ax.plot(df[0], timeInSeconds, "o", label="Messdaten")
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.array([2,6,10,14,18,22,26,])*1000))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(np.array([2,4,6,8,10,12,14,16,18,20,22,24,26,28])*1000))
    labels = ax.get_xticklabels()

    slope, intercept, _, _, _ = stats.linregress(df[0], timeInSeconds)

    ax.plot(df[0], df[0]*slope+intercept, label="Lineare Regression")

    ax.set_xlabel("$N_{Training}$")
    ax.set_ylabel("Zeitdauer [s]")
    ax.legend()

    plt.savefig("images/" + filename + "_time.pdf",bbox_inches='tight')

    plt.cla()

    ax.plot(df[0], df[2], "o", label="Messdaten", )
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.array([2,6,10,14,18,22,26,])*1000))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(np.array([2,4,6,8,10,12,14,16,18,20,22,24,26,28])*1000))
    labels = ax.get_xticklabels()

    ax.set_xlabel("$N_{Training}$")
    ax.set_ylabel("MSE")
    ax.legend()

    plt.savefig("images/" + filename + "_mse.pdf",bbox_inches='tight')

plot("nn_trainlength_uv")
plot("nn_trainlength_vh")
