import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime as dt
from copy import deepcopy
from matplotlib.dates import date2num, HourLocator, MinuteLocator, DayLocator, MonthLocator
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LinearRegression
from functools import reduce
import operator

import parametros as pr
from classes import *
from functions import *



#relajar la condicion cuando estamos sobre el moving average




class OscilationsMovingAverage:
    def __init__(self, info_base):
        self.info_base = info_base
        for wallet in self.info_base.wallets:
            wallet.amount_coin = 100000

    
    def set_low_threshold_in_wallets (self, type_threshold):

        if type_threshold == "average":
            self.info_base.confirm_have_coin()

            if self.info_base.have_coin:
                #obtener el valor mas bajo comprado en las wallets
                
                ###############################################
                #### Por ahora los dos casos serán iguales ####
                ###############################################
                last_date = self.info_base.last_date()
                initial_date = last_date - pr.time_backwards_of_low_threshold_analysis

                mean = self.info_base.mean_in_dates(initial_date, last_date)
                #print(f"mean: {mean}")

                low_threshold = round(mean - mean * pr.interest_percent * pr.low_threshold_amplifier / 100, 4)
                for wallet in self.info_base.wallets:
                    wallet.low_threshold = low_threshold
                ###############################################

            else:
                #obtener el promedio
                last_date = self.info_base.last_date()
                initial_date = last_date - pr.time_backwards_of_low_threshold_analysis

                mean = self.info_base.mean_in_dates(initial_date, last_date)
                #print(f"mean: {mean}")

                low_threshold = round(mean - mean * pr.interest_percent * pr.low_threshold_amplifier / 100, 4)
                for wallet in self.info_base.wallets:
                    wallet.low_threshold = low_threshold

    def high_threshold_calculator (self, type_threshold, last_exchange_price):
        if pr.high_threshold_above_interest:
            return last_exchange_price + last_exchange_price * pr.interest_percent / 100 * 2.2
        elif pr.high_threshold_above_last_price_plus_cte:
            return last_exchange_price + last_exchange_price * pr.high_threshold_constant
        
        else:
            raise Exception("error en calcular highthreshol. Fijarse en los bools")
 
    def curve_regression(self, init_date, last_date):
        data = deepcopy(self.info_base.df.loc[(last_date >= self.info_base.df.date) & (init_date <= self.info_base.df.date)])
        data['time'] = np.arange(len(data.date))
        
        polynomial_features = PolynomialFeatures(degree = 2)
        X = data.loc[:, ['time']]
        X.dropna(inplace = True)
        X = polynomial_features.fit_transform(X)
        X = sm.add_constant(X)
        y = data.loc[:, 'price']

        #train en ordinary least square
        model = sm.OLS(y, X)
        results = model.fit()
        return results

    
    def linear_regression(self, data, y_variable, init_date, last_date):
        data = deepcopy(data.loc[(last_date >= data.date) & (init_date <= data.date)])
        data['time'] = np.arange(len(data.date))
        
        polynomial_features = PolynomialFeatures(degree = 1)
        X = data.loc[:, ['time']]
        X.dropna(inplace = True)
        X = polynomial_features.fit_transform(X)
        X = sm.add_constant(X)
        y = data.loc[:, y_variable]

        #train en ordinary least square
        model = sm.OLS(y, X)
        results = model.fit()
        return results



    def porcentual_dif (self, init_date, last_date):
        data = deepcopy(self.info_base.df.loc[(last_date >= self.info_base.df.date) & (init_date <= self.info_base.df.date)])
        data['lag_1'] = data["price"].shift(1)
        data.dropna(inplace = True)
        data["porcentual_dif"] = (data["price"] - data["lag_1"]) / data["price"] + 1

        if len(data.porcentual_dif) == 0:
            return 1

        return (reduce(operator.mul, data.porcentual_dif) - 1 ) * 100
   
    def porcentual_dif_since_smallest_price (self, init_date, last_date):
        
        df = self.info_base.df.loc[(last_date >= self.info_base.df.date) & (init_date <= self.info_base.df.date)]
        date_of_min_price = df.loc[df.price.min() == df.price].iloc[0,0]
        
        return self.porcentual_dif(date_of_min_price, last_date)
      
    def moving_average (self, last_date, time_backwards = pr.time_backwards_moving_average):
        return self.info_base.mean_in_dates(last_date - time_backwards, last_date)

    def average_last_price (self, last_date):
        return self.info_base.mean_in_dates(last_date - pr.time_backwards_average_last_price, last_date)

    def ema (self, periods, previous_ema, last_price, last_date):
        #esta clase es rara pq define un parametro de info_base desde esta clase / funcion
        
        #constant considering periods
        multiplier = 2 / (periods + 1)

        if previous_ema == 0:
            ema = self.moving_average(last_date)
        else:
            #formula
            ema = (multiplier * (last_price - previous_ema)) + previous_ema   
        return ema

    def macd (self, ema_12, ema_26):
        macd = ema_12 - ema_26
        return macd

    def macd_smooth(self, periods, ema_12, ema_26, last_date):
        previous_macd_smooth = self.info_base.macd_smooth
        if previous_macd_smooth == 0:
            previous_macd_smooth = 0.001
        self.info_base.macd_smooth = self.ema(periods, previous_macd_smooth, self.info_base.macd, last_date)
        return self.info_base.macd_smooth

    def macd_hist(self, macd, macd_smooth):
        return macd - macd_smooth

    def average_true_range(self, periods, init_date, last_date, last_price):
        #usaremos ema
        previous_atr = self.info_base.atr
        tr_value = self.true_range(init_date, last_date, last_price)      
        if previous_atr == 0:
            previous_atr = tr_value
        ema_atr = self.ema(periods, previous_atr, tr_value, last_date)
        self.info_base.atr = ema_atr
        return ema_atr

    def true_range(self, init_date, last_date, last_price):
        highest_value = self.info_base.highest_price(init_date, last_date)
        lowest_value = self.info_base.lowest_price(init_date, last_date)

        tr_1 = highest_value - lowest_value
        tr_2 = highest_value -last_price
        tr_3 = last_price - lowest_value

        tr = max( [tr_1, tr_2, tr_3] )
        self.info_base.tr = tr
        return tr

    def keltner_band (self, periods, last_price, last_date):
        #the periods are in minutes

        previous_ema_keltner = self.info_base.ema_keltner
        ema_keltner = self.ema(periods, previous_ema_keltner, last_price, last_date)
        self.info_base.ema_keltner = ema_keltner

        init_date = last_date - dt.timedelta(minutes=1) * periods
        atr_value = self.average_true_range(periods, init_date, last_date, last_price)

        high_band = ema_keltner + atr_value * 0.6
        low_band = ema_keltner - atr_value * 0.6

        self.info_base.low_band_keltner = low_band
        self.info_base.high_band_keltner = high_band

        return low_band, ema_keltner, high_band
        
    def fast_stocastic_oscillator (self, periods, last_price, last_date):

        init_date = last_date - dt.timedelta(minutes = 1) * periods
        lowest_value = self.info_base.lowest_price(init_date, last_date)
        highest_value = self.info_base.highest_price(init_date, last_date)

        formula = round( ( last_price - lowest_value ) / ( highest_value - lowest_value ) * 100, 4)

        self.info_base.fast_stocastic_oscillator = formula
        return formula

    def slow_stocastic_oscillator (self, periods, last_price, last_date):

        init_date = last_date - dt.timedelta( minutes = 1) * periods
        average = self.info_base.average_fast_stocastic_oscillator(init_date, last_date)

        self.info_base.slow_stocastic_oscillator = average
        return average

    def rsi_calculator (self, periods, last_price, last_date):
        
        closing_prices_values = self.info_base.closing_prices(periods, last_date)
        dif_closing_prices_values = closing_prices_values - closing_prices_values.shift(1)
        dif_closing_prices_values = dif_closing_prices_values.dropna()

        positive_average = np.mean(list(filter(lambda x: True if x >= 0 else False, dif_closing_prices_values)))
        negative_average = np.mean(list(map(lambda y: y * -1, list(filter(lambda x: True if x < 0 else False, dif_closing_prices_values)))))

        RS_value = positive_average / negative_average
        RSI = 100 - ( 100 / ( 1 + RS_value))

        self.info_base.rsi_14 = RSI

        return RSI

        

    def evaluate (self):

          
        last_exchange_price = self.info_base.last_exchange_price
        last_date = self.info_base.last_date()

        previous_ema_5 = self.info_base.ema_5
        previous_ema_20 = self.info_base.ema_20
        previous_ema_50 = self.info_base.ema_50

        self.info_base.ema_5 = self.ema(5, previous_ema_5, last_exchange_price, last_date)
        self.info_base.ema_20 = self.ema(20, previous_ema_20, last_exchange_price, last_date)
        self.info_base.ema_50 = self.ema(50, previous_ema_50, last_exchange_price, last_date)



        print("Exchange price:", last_exchange_price)
        print("Date:", last_date)


        #without interest
        for i, wallet in enumerate(self.info_base.wallets):

            if previous_ema_5 < previous_ema_20:
                if self.info_base.ema_5 >= self.info_base.ema_20:
                    
                    if self.info_base.ema_5 >= self.info_base.ema_50:
                        if self.info_base.ema_20 >= self.info_base.ema_50:
                            if last_exchange_price >= self.info_base.ema_50:

                              
                                amount_coin = wallet.buy_coin(last_exchange_price, last_date = last_date, amount_dolar = 100)
                                print("\n\ncomprar")
                                print(amount_coin)
                                print()

                                #send the information to graph
                                send_information_2_graph(action = "buy", reason = "compra", last_date = last_date)
                                return "buy", amount_coin


            if previous_ema_5 > previous_ema_20:
                if self.info_base.ema_5 <= self.info_base.ema_20:
                    
                    if self.info_base.ema_5 <= self.info_base.ema_50:
                        if self.info_base.ema_20 <= self.info_base.ema_50:
                            if last_exchange_price <= self.info_base.ema_50:
                                
                                coin = dolar_2_coin(amount_dolar = 100, market_price_in_dolar = last_exchange_price)

                                print("\n\nvender")
                                print(coin)
                                print()
                                amount_coin = wallet.sell_coin(last_exchange_price, last_date, amount_coin = coin)
                                send_information_2_graph(action = "sell", reason = 'vender', last_date = last_date)

                                return "sell", amount_coin


        #si pasa nada
        return None, None




