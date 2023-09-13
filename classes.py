from functions import *

import datetime as dt
import pandas as pd
import parametros as pr



class StocasticCondition:

    def __init__ (self):
        self.__start_date = None
        self.__condition_fulfilled = False
        

    @property
    def start_date (self):
        return self.__start_date

    @start_date.setter
    def start_date (self, value):
        self.__start_date = value


    @property
    def condition_fulfilled (self):
        return self.__condition_fulfilled

    @condition_fulfilled.setter
    def condition_fulfilled (self, value):
        if type(value) == bool:
            self.__condition_fulfilled = value
        else:
            raise Exception("EROOOR al intentar seter condition_fulfilled, no es boolean")


    def condition_is_now_fulfilled(self):
        self.condition_fulfilled = True

    def condition_is_not_now_fulfilled(self, last_date):
        self.condition_fulfilled = False
        self.start_date = last_date - pr.time_stocastic_condition_available - dt.timedelta(minutes=1)

    def check_condition_still_available (self, last_date):
        if self.start_date == None:
            self.start_date = last_date
        
        if (last_date - self.start_date) >= pr.time_stocastic_condition_available:
            self.condition_is_not_now_fulfilled(last_date)
        else:
            self.condition_is_now_fulfilled()

    def init_timer_and_condition (self, last_date):
        self.start_date = last_date
        self.check_condition_still_available(last_date)




class Wallet:
    def __init__(self, amount_dolar = 0):
        self.__amount_dolar = amount_dolar
        self.__amount_coin = 0

    
    def buy_coin(self, exchange_price, amount_dolar):
        amount_coin = dolar_2_coin(amount_dolar, exchange_price)
        self.amount_dolar -= amount_dolar
        self.amount_coin += amount_coin

    def sell_coin(self, exchange_price, amount_coin):
        amount_dolar = coin_2_dolar(amount_coin, exchange_price)
        self.amount_dolar += amount_dolar
        self.amount_coin -= amount_coin
        

    @property
    def amount_dolar (self):
        return self.__amount_dolar
    
    @amount_dolar.setter 
    def amount_dolar(self, setter):
        if setter < 0 - pr.epsilon:
            raise Exception("Se está seteando una cantidad de USD negativa")
        elif setter < 0:
            self.__amount_dolar = 0
        else:
            self.__amount_dolar = setter

    @property
    def amount_coin (self):
        return self.__amount_coin
    
    @amount_coin.setter
    def amount_coin(self, setter):
        if setter < 0 - pr.epsilon:
            raise Exception("Se está seteando una cantidad de BTC negativa")
        elif setter < 0:
            self.__amount_coin = 0
        else:
            self.__amount_coin = setter



class InfoWallet(Wallet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.have_coin = False
        self.have_dolar = False
        self.bought_above_ma = False
        self.last_bought_price = 0
        self.high_threshold = 0
        #self.comprar_vender = False
        self.low_threshold = 0
        self.last_bought_price = -1
        self.last_exchange_date = None
        self.can_buy = True
        self.bought_stocastic_oscillator = False
        self.bought_stocastic_oscillator_time = None
        self.confirm_have_coin()
        self.counter_high_band_keltner_pass = 0


    def set_wallet_cant_buy (self):
        self.can_buy = False
    
    def set_wallet_can_buy (self):
        self.can_buy = True

    def confirm_have_coin(self):
        if self.amount_coin > pr.epsilon:
            self.have_coin = True
        else:
            self.have_coin = False
        
        if self.amount_dolar > pr.epsilon:
            self.have_dolar = True
        else:
            self.have_dolar = False

    def buy_coin(self, exchange_price, ma_value = 0, last_date = None, amount_dolar = -1):

        if amount_dolar == -1:
            amount_dolar = self.amount_dolar

        #aplicamos interes
        if pr.with_interest:
            amount_dolar_with_interest = amount_dolar - amount_dolar * pr.interest_percent / 100
        else:
            amount_dolar_with_interest = amount_dolar

        if ma_value < exchange_price:
            self.bought_above_ma = True
        
        self.counter_high_band_keltner_pass = 0
        self.last_bought_price = exchange_price
        self.last_exchange_date = last_date
        amount_coin = dolar_2_coin(amount_dolar_with_interest, exchange_price)
        self.amount_dolar -= amount_dolar
        self.amount_coin += amount_coin
        self.confirm_have_coin()


        print("bought above price:", self.bought_above_ma)
        
        return amount_coin

    def sell_coin(self,  exchange_price, last_date = None, amount_coin = -1):
        if amount_coin == -1:
            amount_coin = self.amount_coin
        
        #aplicamos interes
        if pr.with_interest:
            amount_coin_with_interest = amount_coin - amount_coin * pr.interest_percent / 100
        else:
            amount_coin_with_interest = amount_coin

        self.bought_above_ma = False
        self.bought_stocastic_oscillator = False

        amount_dolar = coin_2_dolar(amount_coin_with_interest, exchange_price)
        self.amount_dolar += amount_dolar
        self.amount_coin -= amount_coin

        self.last_bought_price = -1
        self.last_exchange_date = last_date
        self.confirm_have_coin()

        return amount_coin


class InfoBase:
    def __init__(self, amount_dolar_initially = pr.amount_dolar_initially, *args, **kwargs):
        
        self.wallets = [InfoWallet(amount_dolar_initially)]
        self.stocastic_condition = StocasticCondition()
        self.have_coin = False
        self.__df = None
        self.last_exchange_price = 0
        self.total_dolar = 0
        self.total_coin = 0
        self.total_money = self.set_total_money()
        self.last_exchange_price = None
        self.last_exchange_date = None
        self.ema_12 = 0
        self.ema_26 = 0
        self.ema_24 = 0
        self.ema_52 = 0
        self.macd_2 = 0
        self.macd_smooth_2 = 0
        self.macd_hist_2 = 0
        self.ema_200 = 0
        self.macd = 0
        self.macd_smooth = 0
        self.macd_hist = 0
        self.tr = 0
        self.atr = 0
        self.ema_keltner = 0
        self.high_band_keltner = 0
        self.low_band_keltner = 0
        self.fast_stocastic_oscillator = 0
        self.slow_stocastic_oscillator = 0
        self.rsi_14 = 0
        self.bought_macd_rsi_up_trend = False
        self.need_2_pass_high_keltner = False
        self.ema_5 = 0
        self.ema_20 = 0
        self.ema_50 = 0


    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, value):
        if not type(value) == pd.core.frame.DataFrame:
            print("MEGA ERROR, el df que estas intentando setter en el INFOBASE no es dataframe")

        self.__df = value
        self.__df.date = pd.to_datetime(self.__df.date)
        self.set_last_exchange_price()
    
    def set_total_money (self):
        self.set_total_coin()
        self.set_total_dolar()
        self.total_money = self.total_dolar + self.total_coin * self.last_exchange_price

    def set_total_dolar(self):
        total = 0
        for wallet in self.wallets:
            total += wallet.amount_dolar
        self.total_dolar = total

    def set_total_coin (self):
        total = 0
        for wallet in self.wallets:
            total += wallet.amount_coin
        self.total_coin = total

    def set_last_exchange_price (self):
        try:
            self.last_exchange_price = self.df.iloc[-1].price
        except Exception as e:
            print(e)
            #se mantiene el precio anterior

    def set_last_exchange_date (self):
        try:
            self.last_exchange_date = self.df.iloc[-1].date
        except Exception as e:
            return e
            #se mantiene la anterior
    
    def mean_in_dates (self, init_date, last_date = dt.datetime.now()):
        try:
            return round(self.df.loc[(last_date >= self.df.date) & (self.df.date >= init_date)]["price"].mean(), 4)
        except Exception as e:
            print(e)
            return 0

    def last_date (self):
        try:
            date = self.df.iloc[-1].date
        except Exception as e:
            print(e)
            #se mantiene la fecha anterior
        return date

    def confirm_have_coin (self):

        for wallet in self.wallets:
            if wallet.have_coin:
                self.have_coin = True
                break
        
        self.have_coin = False

    def lowest_threshold (self):  
        return min(self.wallets, key= lambda x: x.low_threshold).low_threshold

    def highest_price (self, init_date, last_date = dt.datetime.now()):
        try:
            return round(self.df.loc[(last_date >= self.df.date) & (self.df.date >= init_date)]["price"].max(), 4)
        except Exception as e:
            print(e)
            return 0

    def lowest_price (self, init_date, last_date = dt.datetime.now()):
        try:
            return round(self.df.loc[(last_date >= self.df.date) & (self.df.date >= init_date)]["price"].min(), 4)
        except Exception as e:
            print(e)
            return 0

    def check_wallet_can_buy(self):
        self.set_last_exchange_date()

        for wallet in self.wallets:

            if not wallet.last_exchange_date == None:
                amount_time_pass = self.last_exchange_date - wallet.last_exchange_date

                if amount_time_pass >= pr.time_not_buy:
                    wallet.set_wallet_can_buy()
                else:
                    wallet.set_wallet_cant_buy()

    def close_price (self, init_date, last_date):
        small_df = self.df[(last_date >= self.df.date) & (init_date <= self.df.date)]
        try:
            price_1 = small_df["price"].iloc[-1]
        except Exception as e:
            price_1 = self.last_exchange_price

        try:
            price_2 = small_df["price"].iloc[-2]
        except Exception as e:
            price_2 = self.last_exchange_price
        
        average = ( price_1 + price_2) / 2
        return average

    def closing_prices (self, periods, last_date):

        length_period = dt.timedelta(minutes = 1)
        last_date_without_seconds = last_date.replace(second = 0)
        init_date = last_date_without_seconds - length_period * periods

        closing_prices_values = []

        for i in range(periods):
            first_date = init_date + length_period * i
            second_date = init_date + length_period * (i + 1)
            close_price_value = self.close_price(first_date, second_date)
            closing_prices_values.append(close_price_value)

        return pd.Series(closing_prices_values)

    def average_fast_stocastic_oscillator (self, init_date, last_date = dt.datetime.now()):
        df_stocastic = pd.read_csv(pr.graph_info_location, parse_dates=['date'])
        try:
            return round(df_stocastic.loc[(last_date >= df_stocastic.date) & (df_stocastic.date >= init_date)]["fast_stocastic_oscillator"].mean(), 4)
        except Exception as e:
            print(e)
            return 0

    def sto_macd_rsi_start_timer(self, last_date):
        self.stocastic_condition.init_timer_and_condition(last_date)

    def sto_macd_rsi_quit_stocastic_condition(self, last_date):
        self.stocastic_condition.condition_is_not_now_fulfilled(last_date)

    def check_cross_high_keltner (self, keltner_high_band_value):
        try:
            price_1 = self.df['price'].iloc[-1]
        except Exception as e:
            price_1 = self.last_exchange_price

        try:
            price_2 = self.df['price'].iloc[-2]
        except Exception as e:
            price_2 = self.last_exchange_price
            
        
        
       

        if ( price_1 >= keltner_high_band_value and price_2 <= keltner_high_band_value) or (price_1 <= keltner_high_band_value and price_2 >= keltner_high_band_value):
            return True
        else:
            return False






