"""
This function returns a dictionary where each key corresponds
to the week number in a given year.
The corresponding value is empty if there is no manuring in that week,
while a list of days if there is manuring during that week is returned.
It's required to pass the weeks corresponding to mais, rice and cereal manuring.
"""

def manuring_periods(year, mais_w, rice_w, cereal_w):
    import datetime
    from collections import defaultdict
    date = datetime.date(year, 1, 1)
    enddate = datetime.date(year, 12, 31)
    weeks = defaultdict(list)
    mais_week = mais_w
    rice_week = rice_w
    cereal_week = cereal_w

    while date < enddate:
        weeks[date.isocalendar()[1]]
        date += datetime.timedelta(days=1)
        if date.isocalendar()[1]==mais_week:
            weeks[date.isocalendar()[1]].append(date.strftime("%d/%m/%Y"))
        elif date.isocalendar()[1]==rice_week:
            weeks[date.isocalendar()[1]].append(date.strftime("%d/%m/%Y"))
        elif date.isocalendar()[1]==cereal_week:
            weeks[date.isocalendar()[1]].append(date.strftime("%d/%m/%Y"))    
    print(weeks)
    return dict(weeks)
