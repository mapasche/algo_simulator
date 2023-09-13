from binance.spot import Spot
from statistics import mean
import pandas as pd
import datetime
import time

api_key = 'PrRyANe2JGfKdI8doWUbJ4QTqSGb534D10u6R5PNuArbQlwtvr9g9hnzJ05tD8X3'
client = Spot(key = api_key)

#entrega el precio promedio en 5 min
#client.avg_price("BTCUSDT")

#entrega el ultimo precio
#client.ticker_price("BTCUSDT")

def price_btc():
    return float(client.ticker_price("BTCUSDT")["price"])

def volume_trades_btc():
    dic = client.klines(symbol= "BTCUSDT", interval="1m")
    #el valor 0 del dic es para los datos en general, el 1 es para todos los candle sticks
    #el valor dentro del 0, num 5 es el volumen y el num 8 la cantidad de trades

    volume = round( float( dic[0][5] ), 4)
    trades = int( dic[0][8] )

    return volume, trades



def upload_btc(price, volume, trades):
        #entrega el tiempo al momento de calcular los datos
        now = datetime.datetime.now().replace(microsecond=0)

        df = pd.read_csv("info.csv", index_col=None)
        df_2 = df.append(pd.DataFrame({
            "date": [now], 
            "price" : [price], 
            "volume" : [volume],
            "trades" : [trades]}, index=None))
        df_2.to_csv("info.csv",index=None)



###########################################################
print("El programa esta corriendo bien")
###########################################################


#hacemos un loop que se cierra al usar ctrl+c
#subimos el valor cada 30 segundos
#hacemos que tome valores cada 1/3 del tiempo y luego promediamos
time_price_avg = 60
amount_price_values = 4

#volumen y trades se obtienen cada 1m, 5m, 1h, 1d etc


time_between_values = time_price_avg / amount_price_values
try:
    while True:

        list_prices = []
        volume = 0
        trades = 0

        for i in range(amount_price_values):
            time.sleep(time_between_values)
            price = round(price_btc(), 2)
            list_prices.append(price)

            print(f"Price:\t\t{price}")

        avg_price = round( mean( list_prices), 2)
        volume, trades = volume_trades_btc()

        print("-------------------------------------------")
        print(f"Avg price:\t{avg_price}\t Volumen:\t{volume}\t Trades:\t{trades}")
        print("-------------------------------------------")
        
        upload_btc(avg_price, volume, trades)


except Exception as e:
    print(e)