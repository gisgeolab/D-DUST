#!/usr/bin/env python
# coding: utf-8


def select_grid(grid):
    """
    A function to select paths for grids
    """
    switcher = {
        '0_1': '/grid/grid_0_1.gpkg',
        '0_066': '/grid/grid_0_066.gpkg',
        '0_01': '/grid/grid_0_01.gpkg'
    }
    return switcher.get(grid, "Invalid grid")


def manuring_periods(year, custom_w):
    """
    Function used to select the time range for the processing and 
    visualize the calendar with the corresponding selected week.
    """

    import datetime
    from collections import defaultdict
    import pandas as pd
    import numpy as np
    from plotly_calplot import calplot
    date = datetime.date(year, 1, 1)
    enddate = datetime.date(year, 12, 31)
    weeks = defaultdict(list)
    custom_week_start = datetime.datetime.strptime((str(year)+'-'+custom_w[0]), "%Y-%m-%d").date()
    custom_week_end = datetime.datetime.strptime((str(year)+'-'+custom_w[1]), "%Y-%m-%d").date()
    all_dates = []
    while date < enddate:
        date += datetime.timedelta(days=1)
        if date >= custom_week_start and date <= custom_week_end:
            all_dates.append(date.strftime("%Y-%m-%d"))
    dictionary = dict(weeks)
    calendar = dict(dictionary)
    all_dates = pd.DataFrame(pd.Series(all_dates),columns=['date'])
    all_dates.date = pd.to_datetime(all_dates.date)
    all_dates['value'] = 1
    # creating the plot
    fig = calplot(all_dates, x="date", y="value")
    fig.show()
    return calendar


def AQ_sensor(year):
    """
    Function for selecting the correct link for downloading zipped .csv air quality data from ARPA sensors
    """
    switcher = {
        '2021': "https://www.dati.lombardia.it/download/wzmx-9k7n/application%2Fx-zip-compressed",
        '2020': "https://www.dati.lombardia.it/download/88sp-5tmj/application%2Fzip",
        '2019': "https://www.dati.lombardia.it/download/j2mz-aium/application%2Fzip",
        '2018': "https://www.dati.lombardia.it/download/4t9j-fd8z/application%2Fzip",
        '2017': "https://www.dati.lombardia.it/download/fdv6-2rbs/application%2Fzip"
    }
    return switcher.get(year, "Invalid year. For current year data use the API request.")


def meteo_sensor(year):
    """
    Function for selecting the correct link for downloading zipped .csv meteorological data from ARPA sensors
    """
    switcher = {
        '2021': "https://www.dati.lombardia.it/download/49n9-866s/application%2Fzip",
        '2020': "https://www.dati.lombardia.it/download/erjn-istm/application%2Fzip",
        '2019': "https://www.dati.lombardia.it/download/wrhf-6ztd/application%2Fzip",
        '2018': "https://www.dati.lombardia.it/download/sfbe-yqe8/application%2Fzip",
        '2017': "https://www.dati.lombardia.it/download/vx6g-atiu/application%2Fzip"
    }
    return switcher.get(year, "Invalid year. For current year data use the API request.")
