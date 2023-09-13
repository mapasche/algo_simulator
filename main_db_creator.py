#muy importnate esto para usar ctrl+c para cerrar el programa
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


import parametros as pr
import pandas as pd
import time
import threading
import main
import datetime as dt
from functions import *
from copy import deepcopy
from binance.spot import Spot


#crear los threads
class MainDataBaseCreator(threading.Thread):

    def __init__ (self, event_turn_of_db, event_turn_of_logic, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

        self.client = Spot(key = pr.api_key)
        self.event_turn_of_db = event_turn_of_db
        self.event_turn_of_logic = event_turn_of_logic
        self.df = None


    def run (self):

        """#Create graph info db
        data_2 = {'date' : [], 'average' : [], 'low_threshold' : []}
        df_graph_info = pd.DataFrame(data_2)
        df_graph_info.date = pd.to_datetime(df_graph_info.date)
        df_graph_info.to_csv(pr.graph_info_location, index=None)"""


        #Erase data in buy sell history
        with open(pr.buy_sell_history_location, "w", encoding="utf-8") as archive:
            archive.write("")
        
        """#Create graph info buy sell
        data_3 = {'date' : [], 'action' : [], "param1" : [], "param2" : [], "param3" : []}
        df_graph_info_3 = pd.DataFrame(data_3)
        df_graph_info_3.date = pd.to_datetime(df_graph_info_3.date)
        df_graph_info_3.to_csv(pr.graph_info_buy_sell_location, index=None)"""
        time_between_loops = 1
        
        #init loop
        while True:
            time.sleep(1)


            price = round(self.price_coin(), 2)
            print(f"Price:\t{price}")

            self.upload_coin(price)

            self.event_turn_of_logic.set()
            self.event_turn_of_db.wait()
            self.event_turn_of_db.clear()
        
            


    def price_coin(self):
        return float(self.client.ticker_price("BTCUSDT")["price"])
    
    def upload_coin(self, price):
        #entrega el tiempo al momento de calcular los datos
        now = dt.datetime.now().replace(microsecond=0)

        df = pd.read_csv(pr.main_db_location, index_col=None)
        df_2 = df.append(pd.DataFrame({
            "date": [now], 
            "price" : [price]}, index=None))
        df_2.to_csv(pr.main_db_location,index=None)





print("se esta iniciando")
#crear events
event_turn_of_logic = threading.Event()
event_turn_of_db = threading.Event()

#instanciar clases de otros modulos
logic_money = main.LogicMoney(pr.artificial_db_location, event_turn_of_db, event_turn_of_logic, daemon= True)
main_db_creator = MainDataBaseCreator(event_turn_of_db, event_turn_of_logic, daemon=True)


#comenzar los Threads
logic_money.start()
main_db_creator.start()



try:
    while main_db_creator.is_alive():
        main_db_creator.join(0.5)


except KeyboardInterrupt as e:
    print("Se apretaron las teclas para cerrar la app, vendiendo cryptos")
    #vender monedas -------------------------------------------

except Exception as e:
    print(e)
    #vender monedas -------------------------------------------

finally:
    print("\nfin del programa")
    #vender monedas -------------------------------------------

    pass


