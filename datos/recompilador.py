from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import urllib.request
import json
import time
import pandas as pd 
import datetime
from statistics import mean





data_path = "info.csv"
driver = webdriver.Chrome("chromedriver.exe")
driver.get("https://currency.com/trading/platform/trade")
delay = 15

#sec
time_between_data = 30
time_refresh = 5


try:
    myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'txt__link')))
    print("Page 1 is ready!")
    myElem.click()

except TimeoutException:
    print("Loading took too much time!")




try:
    username = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.NAME, 'username')))
    password = driver.find_element_by_name("password")
    print("Page 2 is ready!")
    username.send_keys("ignacio.pasche@gmail.com")
    password.send_keys("Sug@rglider0680")
    password.submit()

except TimeoutException:
    print("Loading took too much time!")



try:
    time.sleep(10)
    base_element = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'trade-instruments-panel')))

    row_elements = base_element.find_elements_by_class_name("trade-instrument")

    #BTC
    row_element = row_elements[-1]

    element_number = row_element.find_element_by_class_name("number")
    print(element_number.text)
    print("Page 3 is ready!")

except TimeoutException:
    print("Loading took too much time!")


time.sleep(1)

try:

    while True:
        
        driver.refresh()
        time.sleep(time_refresh)
        base_element = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'trade-instruments-panel')))
        row_elements = base_element.find_elements_by_class_name("trade-instrument")

        if len(row_elements) == 0:
            print("error al capturar los row elements, se entregaron cero elementos")
            continue

        #BTC
        row_element = row_elements[-1]
        element_number = row_element.find_element_by_class_name("number")

        #sacar promedio
        number_1 = float(row_element.find_element_by_class_name("number").text.replace(",", "."))
        print(number_1)
        time.sleep(time_between_data / 3)
        number_2 = float(row_element.find_element_by_class_name("number").text.replace(",", "."))
        print(number_2)
        time.sleep(time_between_data / 3)
        number_3 = float(row_element.find_element_by_class_name("number").text.replace(",", "."))
        print(number_3)

        number = round(mean([number_1, number_2, number_3]), 2)
        print("avg:", number)

        now = datetime.datetime.now()
        
        df = pd.read_csv("info.csv", index_col=None)
        df_2 = df.append(pd.DataFrame({"date": [now], "price" : [number]}, index=None))
        df_2.to_csv("info.csv",index=None)

        time.sleep(time_between_data / 3)
        


except Exception as e:
    print(e)
    driver.quit()
