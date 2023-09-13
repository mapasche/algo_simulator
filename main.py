from classes import *
from functions import *
import pandas as pd
import threading
import parametros as pr
import modelo_2 as md
from binance.spot import Spot as Client
from binance.lib.utils import config_logging
from binance.error import ClientError
import logging



class LogicMoney(threading.Thread):
    def __init__(self , db_location, event_turn_of_db, event_turn_of_logic, *arg, **kwargs):
        super().__init__(**kwargs)

        self.db_location = db_location
        self.event_turn_of_db = event_turn_of_db
        self.event_turn_of_logic = event_turn_of_logic
        self.info_base = InfoBase()
        self.model = md.OscilationsMovingAverage(self.info_base)     


    def run(self):
        
        while True:
            #recibe el event
            self.event_turn_of_logic.wait()
            self.event_turn_of_logic.clear()

            self.download_db()
            
            action, amount = self.model.evaluate()

            last_exchange_price = self.info_base.last_exchange_price

            if not action == None:
                self.info_base.set_total_money()
                self.upload_action_to_history(action)

                self.print_sell_buy(action)
                if action == "buy":
                    self.buy_coin(amount)

                
                elif action == "sell":
                    self.sell_coin(amount)

            self.upload_graph_info()

            #Manda event
            self.event_turn_of_db.set()  

    

    def buy_coin(self, amount_dolar):
        pass


    def sell_coin(self, amount_btc):
        pass


    def print_sell_buy(self, action):
        print(f"Action: {action} \t Total money: {self.info_base.total_money} \t Date: {self.info_base.last_date()}\n")
        print(f"Total dolar: {round(self.info_base.total_dolar, 4)} \t Total coin: {round(self.info_base.total_coin, 4)}\n")
        for i, wallet in enumerate(self.info_base.wallets):
            print(f"Wallet {i}: \t Dolar: {round(self.info_base.total_dolar, 4)} \t Coin: {round(self.info_base.total_coin, 4)}\n")
        print(f"\n\n")


    def download_db(self):
        self.info_base.df = pd.read_csv(pr.artificial_db_location)
        return self.info_base.df

    def upload_action_to_history (self, action):
        last_date = self.info_base.last_date()
        with open(pr.buy_sell_history_location, "a", encoding="utf-8") as archive:
            archive.write(f"Action: {action} \t Total money: {self.info_base.total_money} \t Date: {last_date}\n")
            archive.write(f"Total dolar: {round(self.info_base.total_dolar, 4)} \t Total coin: {round(self.info_base.total_coin, 4)}\n")
            for i, wallet in enumerate(self.info_base.wallets):
                archive.write(f"Wallet {i}: \t Dolar: {round(self.info_base.total_dolar, 4)} \t Coin: {round(self.info_base.total_coin, 4)}\n")
            archive.write(f"\n\n")

        
    def upload_graph_info (self):
        last_date = self.info_base.last_date()
        initial_date = last_date - pr.time_backwards_of_low_threshold_analysis

        last_date = self.info_base.last_date()
        mean = self.info_base.mean_in_dates(initial_date, last_date)
        lowest_threshold = round(self.info_base.lowest_threshold(), 5)
        ema_12 = round(self.info_base.ema_12, 5)
        ema_26 = round(self.info_base.ema_26, 5)
        ema_200 = round(self.info_base.ema_200, 5)
        macd = round(self.info_base.macd, 5)
        macd_smooth = round(self.info_base.macd_smooth, 5)
        macd_hist = round(self.info_base.macd_hist, 5)
        ema_24 = round(self.info_base.ema_24, 5)
        ema_52 = round(self.info_base.ema_52, 5)
        macd_2 = round(self.info_base.macd_2, 5)
        macd_smooth_2 = round(self.info_base.macd_smooth_2, 5)
        macd_hist_2 = round(self.info_base.macd_hist_2, 5)
        ema_keltner = round(self.info_base.ema_keltner, 5)
        low_band_keltner = round(self.info_base.low_band_keltner, 5)
        high_band_keltner = round(self.info_base.high_band_keltner, 5)
        fast_stocastic_oscillator = round(self.info_base.fast_stocastic_oscillator, 5)
        slow_stocastic_oscillator = round(self.info_base.slow_stocastic_oscillator, 5)
        rsi_14 = round(self.info_base.rsi_14, 5)

        with open(pr.graph_info_location, "a", encoding="utf-8") as archive:
            archive.write(f"{last_date},{mean},{lowest_threshold},{ema_12},{ema_26},{macd},{macd_smooth},{macd_hist},{ema_24},{ema_52},{macd_2},{macd_smooth_2},{macd_hist_2},{ema_200},{low_band_keltner},{ema_keltner},{high_band_keltner},{fast_stocastic_oscillator},{slow_stocastic_oscillator},{rsi_14}\n")





