from apisetup import MySportsFeeds
import time


class GetRawData:
    def __init__(self, sleep_time):
        self.msf = MySportsFeeds(version="2.1", store_type='file', store_location='Results/')
        self.msf.authenticate("2adef7f2-54b9-4983-88b2-678277", "MYSPORTSFEEDS")
        self.sleep_time = sleep_time

    def get_games(self, season_year, team_id=None, date_curr=None):
        print('waiting on games')
        time.sleep(self.sleep_time)
        return self.msf.msf_get_data(league='mlb',season=season_year,feed='seasonal_games',team=team_id,date=date_curr,format='json')
    
    def get_lineup(self, season_year, game_id):
        print('waiting on lineup')
        time.sleep(self.sleep_time)
        return self.msf.msf_get_data(league='mlb',season=season_year,feed='game_lineup',game=game_id,format='json')

    def get_boxscore(self, season_year, game_id):
        print('waiting on boxscore')
        time.sleep(self.sleep_time)
        return self.msf.msf_get_data(league='mlb',season=season_year,feed='game_boxscore',game=game_id,format='json')

    def get_season_player(self, season_year, player_list, date_curr):
        print('waiting on season player')
        time.sleep(self.sleep_time)
        return self.msf.msf_get_data(league='mlb',season=season_year,feed='seasonal_player_stats',player=player_list,date=date_curr,format='json')

    def get_team_stats(self, team_id, season_year, date_curr):
        print('waiting on team stats')
        time.sleep(self.sleep_time)
        return self.msf.msf_get_data(league='mlb',season=season_year,feed='seasonal_team_stats',team=team_id,date=date_curr,format='json')

    def get_current_season(self, date_curr):
        print('waiting on current season')
        time.sleep(self.sleep_time)
        return self.msf.msf_get_data(league='mlb',feed='current_season',date=date_curr,format='json')

    def get_daily_games(self, season_year, date_curr):
        print('waiting on daily games')
        time.sleep(self.sleep_time)
        return self.msf.msf_get_data(league='mlb', feed='daily_games',season=season_year,date=date_curr,format='json')
