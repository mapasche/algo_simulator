import datetime as dt

api_key = 'PrRyANe2JGfKdI8doWUbJ4QTqSGb534D10u6R5PNuArbQlwtvr9g9hnzJ05tD8X3'


amount_dolar_initially = 10000

results_OLS_2_buy_enable = False
results_OLS_2_sell_enable = True
interest_percent = 0.075
with_interest = False
with_high_threshold = True
high_threshold_above_interest = False
high_threshold_above_last_price_plus_cte = True



multiple_graphs = True 
with_rsi = True
with_macd = True
with_stocastic = True
amount_graphs = 4


#location of archives
db_location = "datos/info_2.csv"

buy_sell_history_location = "resultados/buy_sell_history.txt"
graph_info_location = "resultados/graph_info.csv"
graph_info_buy_sell_location = "resultados/graph_info_buy_sell.csv"
graph_location = "resultados/graph.png"

main_db_location = "datos/info_2.csv"
#artificial_db_location = "datos/info_2.csv"
artificial_db_location = "datos/info_art.csv"

#año - mes - dia - hora - minuto
initial_date = dt.datetime(2022, 12, 22, 18, 0)
final_date = dt.datetime(2022, 12, 26, 12, 0)








time_backwards_for_art_db = dt.timedelta(hours=2)
epsilon = 0.0000001




#parameters model of oscilations

type_of_low_threshold = "average"
low_threshold_amplifier = 2
time_backwards_of_low_threshold_analysis = dt.timedelta(minutes= 45)
min_value_x2_curve_regression_buy = 0.005
min_value_x2_curve_regression_sell = 0.005
time_backwards_of_cond_low_curve_regression = dt.timedelta(minutes=10)
time_backwards_of_cond_high_curve_regression = dt.timedelta(minutes=10)
time_backwards_of_cond_low_high_curve_regression_small = dt.timedelta(minutes = 1)
min_value_slope_regression_buy =  0.000001
min_value_slope_regression_sell = 0.000001


time_backwards_of_cond_porcentual_dif = dt.timedelta(minutes=1)
time_not_buy = dt.timedelta(minutes = 10)
high_threshold_constant = 0.1 / 100 / 5

stop_loss_proportion = 0.1 / 100


#moving average
time_backwards_moving_average = dt.timedelta(minutes = 60)
time_backwards_average_last_price = dt.timedelta(minutes = 1)


#EMA exponential moving average
min_macd_hist = 20

#buy above macd
macd_buy_price = 40


#stocastic oscillator
stocastic_oscillator_low_value_buy = 15
stocastic_oscillator_high_value_sell = 80


#stocastic condition
time_stocastic_condition_available = dt.timedelta(minutes = 10)

time_backwards_linear_regression_rsi = dt.timedelta(minutes= 5)
time_backwards_linear_regression_ma = dt.timedelta(minutes= 5)

slow_stocastic_oscillator_low_value_for_condition = 20
fast_stocastic_oscillator_high_value_for_condition = 75









