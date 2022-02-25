#!/usr/bin/env python
# coding: utf-8

# In[25]:


def AQ_sensor(year):
    switcher = {
        '2020': "https://www.dati.lombardia.it/download/88sp-5tmj/application%2Fzip",
        '2019': "https://www.dati.lombardia.it/download/j2mz-aium/application%2Fzip",
        '2018': "https://www.dati.lombardia.it/download/4t9j-fd8z/application%2Fzip",
        '2017': "https://www.dati.lombardia.it/download/fdv6-2rbs/application%2Fzip"
    }
    return switcher.get(year, "Invalid year. For current year data use the API request.")

def meteo_sensor(year):
    switcher = {
        '2021': "https://www.dati.lombardia.it/download/49n9-866s/application%2Fzip",
        '2020': "https://www.dati.lombardia.it/download/erjn-istm/application%2Fzip",
        '2019': "https://www.dati.lombardia.it/download/wrhf-6ztd/application%2Fzip",
        '2018': "https://www.dati.lombardia.it/download/sfbe-yqe8/application%2Fzip",
        '2017': "https://www.dati.lombardia.it/download/vx6g-atiu/application%2Fzip"
    }
    return switcher.get(year, "Invalid year. For current year data use the API request.")

