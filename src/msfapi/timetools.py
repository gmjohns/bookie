from datetime import datetime
import pytz

def format_date(input_date):
    tz_east = pytz.timezone('America/New_York')
    utc = datetime.strptime(input_date, '%Y-%m-%dT%H:%M:%S.%fZ')
    east = utc.astimezone(tz_east)
    return str(east.date()).replace('-', '')