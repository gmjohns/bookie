from datetime import datetime, timedelta
from dateutil import tz

def convert_utc_to_est(utc_date, format_string):
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/New_York')
    utc = datetime.strptime(utc_date, format_string)
    utc = utc.replace(tzinfo=from_zone)
    east = utc.astimezone(to_zone)
    return east

def format_datetime(input_date):
    east = convert_utc_to_est(input_date, '%Y-%m-%dT%H:%M:%S.%fZ')
    return str(east.date()).replace('-', '')

def format_date(input_date):
    east = convert_utc_to_est(input_date, '%Y-%m-%dZ')
    return str(east.date()).replace('-', '')

def get_previous_day(gameday):
    gameday = datetime.strptime(gameday, '%Y%m%d')
    prev_day = gameday - timedelta(days=1)
    return str(prev_day.date()).replace('-', '')

def prev_in_range(gameday, start, end):
    gameday = datetime.strptime(gameday, '%Y%m%d')
    prev_day = gameday - timedelta(days=1)
    start = datetime.strptime(start, '%Y-%m-%dZ')
    end = datetime.strptime(end, '%Y-%m-%dZ')
    return start <= prev_day <= end
