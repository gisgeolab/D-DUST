#!/usr/bin/env python
# coding: utf-8


def select_grid(grid):
    """
    A function to select paths for grids
    """
    switcher = {
        'cams': '/grid/grid_cams.gpkg',
        's5p': '/grid/grid_s5p.gpkg',
        'arpa': '/grid/grid_arpa.gpkg'
    }
    return switcher.get(grid, "Invalid grid")


def manuring_periods(year, mais_w, rice_w, cereal_w, custom_w):
    """
    This function returns a dictionary where each key corresponds
    to the week number in a given year.
    The corresponding value is empty if there is no manuring in that week,
    while a list of days if there is manuring during that week is returned.
    It's required to pass the weeks corresponding to mais, rice and cereal manuring.
    """

    import datetime
    from collections import defaultdict
    import pandas as pd
    import numpy as np
    from plotly_calplot import calplot
    date = datetime.date(year, 1, 1)
    enddate = datetime.date(year, 12, 31)
    weeks = defaultdict(list)
    mais_week = mais_w
    rice_week = rice_w
    cereal_week = cereal_w
    custom_week = custom_w
    all_dates = []
    while date < enddate:
        weeks[date.isocalendar()[1]]
        date += datetime.timedelta(days=1)
        if date.isocalendar()[1] in mais_week:
            weeks[date.isocalendar()[1]].append(date.strftime("%Y-%m-%d"))
            all_dates.append(date.strftime("%Y-%m-%d"))
        elif date.isocalendar()[1] in rice_week:
            weeks[date.isocalendar()[1]].append(date.strftime("%Y-%m-%d"))
            all_dates.append(date.strftime("%Y-%m-%d"))
        elif date.isocalendar()[1] in cereal_week:
            weeks[date.isocalendar()[1]].append(date.strftime("%Y-%m-%d")) 
            all_dates.append(date.strftime("%Y-%m-%d"))
        elif date.isocalendar()[1] in custom_week:
            weeks[date.isocalendar()[1]].append(date.strftime("%Y-%m-%d")) 
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

