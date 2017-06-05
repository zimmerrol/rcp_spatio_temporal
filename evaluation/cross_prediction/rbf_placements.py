import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.ticker as mticker
from scipy import stats
import pandas as pd

def plot(filename):
    df = pd.read_csv("data/" + filename + ".dat", parse_dates=[2], header=None, delimiter="	", comment="#")

    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)

    frmt = mdates.DateFormatter('%H:%M:%S')

    timeInSeconds = []
    for i in range(1,22):
        date = df[2][i]
        timeInSeconds.append(date.second+(date.minute+date.hour*60)*60)

    ax.plot(df[0][1:], timeInSeconds, "o", label="Messdaten")
    labels = ax.get_xticklabels()

    slope, intercept, _, _, _ = stats.linregress(df[0][1:], timeInSeconds)

    ax.plot(df[0], df[0]*slope+intercept, label="Lineare Regression")

    ax.set_xlabel("Anzahl der Basisfunktionen")
    ax.set_ylabel("Zeitdauer [s]")
    ax.legend()

    plt.savefig("images/" + filename + "_time.pdf", bbox_inches='tight')

    plt.cla()


    ax.plot(df[0][1:], df[3][1:], "o", label="Messdaten", )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    #ax.set_yscale('log')


    labels = ax.get_xticklabels()

    ax.set_xlabel("Anzahl der Basisfunktionen")
    ax.set_ylabel("MSE")
    ax.legend()

    plt.savefig("images/" + filename + "_mse.pdf", bbox_inches='tight')

plot("rbf_placements_uv")
plot("rbf_placements_vh")
