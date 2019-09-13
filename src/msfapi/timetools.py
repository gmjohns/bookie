from datetime import datetime, timedelta
import pytz

def format_date(input_date):
    tz_east = pytz.timezone('America/New_York')
    utc = datetime.strptime(input_date, '%Y-%m-%dT%H:%M:%S.%fZ')
    east = utc.astimezone(tz_east)
    return str(east.date()).replace('-', '')

def get_previous_season_end(curr_season):
    if curr_season == '2017-regular':
        last = ('20160930', '2016-regular')
    elif curr_season == '2018-regular':
        last = ('20170930', '2017-regular')
    elif curr_season == '2019-regular':
        last = ('20180930', '2016-regular')
    else:
        return None
    return last

def get_previous_day(input_date):
    tz_east = pytz.timezone('America/New_York')
    utc = datetime.strptime(input_date, '%Y-%m-%dT%H:%M:%S.%fZ')
    east = utc.astimezone(tz_east)
    return str(east.date() - timedelta(days=1)).replace('-', '')

def prev_in_range(input_date, start, end):
    tz_east = pytz.timezone('America/New_York')
    start = datetime.strptime(start, '%Y-%m-%dZ').isoformat()
    end = datetime.strptime(end, '%Y-%m-%dZ')
    utc = datetime.strptime(input_date, '%Y-%m-%dT%H:%M:%S.%fZ')
    prev_date = utc.astimezone(tz_east) - timedelta(days=1)
    return start < prev_date < end
