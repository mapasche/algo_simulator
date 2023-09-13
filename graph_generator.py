import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime as dt
from copy import deepcopy
from matplotlib.dates import date2num, HourLocator, MinuteLocator, DayLocator, MonthLocator
from sklearn.preprocessing import PolynomialFeatures

import parametros as pr
from classes import *
from functions import *



def setear_visualizador_eje_x(ax, init_date, last_date):
    locator = AutoDateLocator()

    #setear el visualizador del eje x
    if last_date - init_date <= timedelta(hours=1):
        ax.xaxis.set_major_locator(MinuteLocator(byminute=(0, 15, 30, 45)))
    elif last_date - init_date <= timedelta(hours=2):
        ax.xaxis.set_major_locator(MinuteLocator(byminute=(0, 30)))
    elif last_date - init_date <= timedelta(hours=3):
        ax.xaxis.set_major_locator(HourLocator(byhour=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 ,20 ,21, 22, 23)))
    elif last_date - init_date <= timedelta(hours=6):
        ax.xaxis.set_major_locator(HourLocator(byhour=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18 , 20, 22)))
    elif last_date - init_date <= timedelta(hours=12):
        ax.xaxis.set_major_locator(HourLocator(byhour=(0, 3, 6, 9, 12, 15, 18, 21)))
    elif last_date - init_date <= timedelta(hours=24):
        ax.xaxis.set_major_locator(HourLocator(byhour=(0, 6, 12, 18)))
    elif last_date - init_date <= timedelta(days=4):
        ax.xaxis.set_major_locator(HourLocator(byhour=(0, 12)))
    elif last_date - init_date <= timedelta(days=7):
        ax.xaxis.set_major_locator(locator)
        #ax.xaxis.set_major_locator(DayLocator(bymonthday=(0, 2, 4, 6)))
    elif last_date - init_date <= timedelta(days=31):
        ax.xaxis.set_major_locator(DayLocator(byhour=(0, 7, 14, 21, 28)))
    else:
        ax.xaxis.set_major_locator(MontLocator())


def plot_labels ():
    plt.xlabel('Fechas')    
    plt.ylabel('Precio en USD $')
    plt.title("USD-BTC")


def plot_size (i, j):
    plt.figure(figsize=(i, j))


def create_cuadratic (df, time_backwards, index_buy_sell):

    init_date = df_buy_sell.date[index_buy_sell] - time_backwards
    last_date = df_buy_sell.date[index_buy_sell]

    data = df.loc[(last_date >= df.date) & (init_date <= df.date)]

    polynomial_features = PolynomialFeatures(degree = 2)
    X = data.loc[:, ['time']]
    X.dropna(inplace = True)
    X = polynomial_features.fit_transform(X)
    X = sm.add_constant(X)
    y = data.loc[:, 'price']

    #train en ordinary least square
    model = sm.OLS(y, X)
    results = model.fit()

    last_date_predict = last_date + time_backwards
    data = df.loc[(last_date_predict >= df.date) & (init_date <= df.date)]

    X = data.loc[:, ['time']]
    X.dropna(inplace = True)
    X = polynomial_features.fit_transform(X)
    X = sm.add_constant(X)

    y_pred = results.predict(X)
    return data, y_pred



adjust_dates = True


figure, axs = plt.subplots(4, 1, figsize = (10, 10))

#descargamos la informacion
df = pd.read_csv(pr.artificial_db_location, parse_dates=["date"])
df['time'] = np.arange(len(df.date))
df_graph = pd.read_csv("resultados/graph_info.csv", parse_dates=['date'])
df_buy_sell = pd.read_csv(pr.graph_info_buy_sell_location, parse_dates=['date'])


init_date = df_graph.date.iloc[0]
last_date = df_graph.date.iloc[-1]

if adjust_dates:
    init_date = dt.datetime(2022, 8, 14, 0)
    last_date = dt.datetime(2022, 8, 16, 18)



df = df[(df.date >= init_date) & (df.date <= last_date)]
df_graph = df_graph[(df_graph.date >= init_date) & (df_graph.date <= last_date)]
df_buy_sell = df_buy_sell[(df_buy_sell.date >= init_date) & (df_buy_sell.date <= last_date)]




axs[0].plot_date(x = df.date, y = df.price, linestyle='-', markersize = 0.01, label="Price")
axs[0].plot_date(x= df_graph.date, y = df_graph.average, linestyle='-', markersize = 0.01, label="MA")
axs[0].plot_date(x= df_graph.date, y = df_graph.low_threshold, linestyle='-', markersize = 0.01, label="MA - Interest")






axs[0].plot_date(x = df.date, y = df.price, linestyle='-', markersize = 0.001, label="Price")
#axs[0].plot_date(x= df_graph.date, y = df_graph.average, linestyle='-', markersize = 0.001, label="MA")
#axs[0].plot_date(x= df_graph.date, y = df_graph.ema_12, linestyle='-', markersize = 0.001, label="EMA 12")
#axs[0].plot_date(x= df_graph.date, y = df_graph.ema_26, linestyle='-', markersize = 0.001, label="EMA 26")
#axs[0].plot_date(x= df_graph.date, y = df_graph.ema_200, linestyle='-', markersize = 0.001, label="EMA 200")
#axs[0].plot_date(x= df_graph.date, y = df_graph.low_threshold, linestyle='-', markersize = 0.001, label="MA - Interest")
axs[0].plot_date(x= df_graph.date, y = df_graph.low_band_keltner, linestyle='-', markersize = 0.001, label="low band keltner")
axs[0].plot_date(x= df_graph.date, y = df_graph.ema_keltner, linestyle='-', markersize = 0.001, label="EMA keltner")
axs[0].plot_date(x= df_graph.date, y = df_graph.high_band_keltner, linestyle='-', markersize = 0.001, label="high band keltner")
#axs[0].fill_between(df_graph.date, df_graph.low_band_keltner, df_graph.high_band_keltner, color="blue", alpha = 0.1)


#setear_visualizador_eje_x(axs[0], init_date, last_date)



axs[1].plot_date(x= df_graph.date, y = df_graph.rsi_14, color="red", linestyle='-', markersize = 0.001, label="RSI 14")
axs[1].axhline(y=50, linewidth=1, color='black')

axs[2].plot_date(x= df_graph.date, y = df_graph.fast_stocastic_oscillator, color="red", linestyle='-', markersize = 0.001, label="fast stocastic oscillator")
axs[2].plot_date(x= df_graph.date, y = df_graph.slow_stocastic_oscillator, color="blue", linestyle='-', markersize = 0.001, label="slow stocastic oscillator")
axs[2].axhline(y=80, linewidth=1, color="gray")
axs[2].axhline(y=20, linewidth=1, color='gray')

#axs[3].plot_date(x= df_graph.date, y = df_graph.macd, color="red", linestyle='-', markersize = 0.001, label="MACD")
#axs[3].plot_date(x= df_graph.date, y = df_graph.macd_smooth, color="blue", linestyle='-', markersize = 0.001, label="MACD EMA")
axs[3].plot_date(x= df_graph.date, y = df_graph.macd_hist, color="black", linestyle='-', markersize = 0.001, label="MACD HIST")
#axs[3].fill_between(df_graph.date, 0, df_graph.macd_hist, color="black", alpha = 0.1)

axs[3].plot_date(x= df_graph.date, y = df_graph.macd_hist_2, color="orange", linestyle='-', markersize = 0.001, label="MACD HIST 2")
#axs[3].fill_between(df_graph.date, 0, df_graph.macd_hist_2, color="blue", alpha = 0.1)



#vertical lines of buy and sell
df_buy_sell.apply(lambda row: axs[0].axvline(x=row.date, linewidth=0.5, color="red") if row.action == "buy" else axs[0].axvline(x=row.date, linewidth=0.5, color="green") , axis=1)


"""for i in df_buy_sell.index:
    data, y_pred = create_cuadratic(df, dt.timedelta(minutes=10), i)
    axs[0].plot_date(x = data.date, y = y_pred, linestyle='-', markersize = 0.001, color="purple")"""



axs[2].grid(True)
axs[3].grid(True)
axs[0].grid(True)
axs[1].grid(True)


#legends
#fig = plt.gcf()
figure.legend(loc=1)
figure.tight_layout()
#plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.show()