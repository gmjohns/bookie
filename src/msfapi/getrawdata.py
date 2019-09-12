from apisetup import MySportsFeeds


class GetRawData:
    def __init__(self):
        self.msf = MySportsFeeds(version="2.1", store_type='file', store_location='Results/')
        self.msf.authenticate("2adef7f2-54b9-4983-88b2-678277", "MYSPORTSFEEDS")

    def get_games(self, season_year):
        return self.msf.msf_get_data(league='mlb',season=season_year,feed='seasonal_games',format='json')
    
    def get_lineup(self, season_year, game_id):
        return self.msf.msf_get_data(league='mlb',season=season_year,feed='game_lineup',game=game_id,format='json')

    def get_boxscore(self, season_year, game_id):
        return self.msf.msf_get_data(league='mlb',season=season_year,feed='game_boxscore',game=game_id,format='json')

    def get_season_player(self, season_year, player_id, date_curr):
        return self.msf.msf_get_data(league='mlb',season=season_year,feed='seasonal_player_stats',player=player_id,date=date_curr,format='json')

    def get_team_stats(self, team_id, season_year, date_curr):
        return self.msf.msf_get_data(league='mlb',season=season_year,feed='seasonal_team_stats',team=team_id,date=date_curr,format='json')

    def get_season_games(self, team_id, season_year, date_curr):
        return self.msf.msf_get_data(league='mlb',season=season_year,feed='seasonal_games',team=team_id,date=date_curr,format='json')
