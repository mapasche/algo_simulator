import parametros as pr
import pandas as pd
import threading
import main
import datetime as dt
from functions import *
from copy import deepcopy
import winsound





#crear los threads
class ArtificialDataBaseCreator(threading.Thread):

    def __init__ (self, init_date, end_date, event_turn_of_db, event_turn_of_logic, *arg, **kwargs):
        super().__init__(**kwargs)

        #confirmar que dates esten en datetime
        if not type(init_date) == dt.datetime or not type(end_date) == dt.datetime:
            raise Exception("init o end date no son clase datetime")

        #dates in datetime
        self.init_date = init_date
        self.end_date = end_date
        self.event_turn_of_db = event_turn_of_db
        self.event_turn_of_logic = event_turn_of_logic
        self.df = None
        self.df_art = None


    def run (self):

        #Create artificial db
        data = {'date' : [], 'price' : []}
        self.df_art = pd.DataFrame(data)
        self.df_art.date = pd.to_datetime(self.df_art.date)
        self.df_art.to_csv(pr.artificial_db_location,index=None)


        #Create graph info db
        data_2 = {'date' : [],
            'average' : [],
            'low_threshold' : [],
            'ema_12' : [],
            'ema_26' : [],
            'macd' : [],
            'macd_smooth' : [],
            'macd_hist' : [],
            'ema_24' : [],
            'ema_52' : [],
            'macd_2' : [],
            'macd_smooth_2' : [],
            'macd_hist_2' : [],
            'ema_200' : [],
            'low_band_keltner' : [],
            'ema_keltner' : [],
            'high_band_keltner' : [],
            'fast_stocastic_oscillator' : [],
            'slow_stocastic_oscillator' : [],
            'rsi_14' : [],
            }
        df_graph_info = pd.DataFrame(data_2)
        df_graph_info.date = pd.to_datetime(df_graph_info.date)
        df_graph_info.to_csv(pr.graph_info_location, index=None)


        #Erase data in buy sell history
        with open(pr.buy_sell_history_location, "w", encoding="utf-8") as archive:
            archive.write("")
        
        #Create graph info buy sell
        data_3 = {'date' : [], 'action' : [], 'reason' : [] , "param0" : [], "param1" : [], "param2" : [], "porcentual_dif" : []}
        df_graph_info_3 = pd.DataFrame(data_3)
        df_graph_info_3.date = pd.to_datetime(df_graph_info_3.date)
        df_graph_info_3.to_csv(pr.graph_info_buy_sell_location, index=None)


        #considerar primer caso
        self.download_db()
        first_index = self.df[self.df.date > self.init_date].index[0]
        index_start_art_db = self.df[self.df.date > self.init_date - pr.time_backwards_for_art_db].index[0]
        index = first_index + 1

        try:

            #init loop
            while self.df.loc[index].date < self.end_date:           

                self.update_db_art(index_start_art_db, index)

                self.event_turn_of_logic.set()
                self.event_turn_of_db.wait()
                self.event_turn_of_db.clear()

                index += 1
                #index_start_art_db += 1
            
        except Exception as e:
            print(e)


    def download_db (self):
        self.df = pd.read_csv(pr.db_location)
        self.df.date = pd.to_datetime(self.df.date)
        return self.df

    def update_db_art (self, first_index, second_index):
        self.df_art = deepcopy(self.df.iloc[first_index:second_index])
        self.df_art.to_csv(pr.artificial_db_location, index = None)



#crear events
event_turn_of_logic = threading.Event()
event_turn_of_db = threading.Event()



#instanciar clases de otros modulos
logic_money = main.LogicMoney(pr.artificial_db_location, event_turn_of_db, event_turn_of_logic, daemon= True)
art_db_creator = ArtificialDataBaseCreator(pr.initial_date, pr.final_date, event_turn_of_db, event_turn_of_logic, daemon=True)

 
#comenzar los Threads
logic_money.start()
art_db_creator.start()


art_db_creator.join()





duration = 100 #mili seconds
for _ in range(1):

    for __ in range(3):
        frequency = 200
        for i in range(1, 5):
            frequency *= i
            winsound.Beep(frequency, duration)

    for __ in range(3):
        frequency = 200
        for i in range(4, 0, -1):
            frequency *= i
            winsound.Beep(frequency, duration)




#imprimir gráficos con la información
show_final_graph()

print("fin del programa")