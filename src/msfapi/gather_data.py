import numpy as np
import timetools
from getrawdata import GetRawData

    
class GetStats:

    raw = GetRawData()
    
    def get_game(self, season):
        games = {}
        # api call to get json formatted list of all games 
        season_games = self.raw.get_games(season)

        for game in season_games['games']:
            id = game['schedule']['id']
            date = game['schedule']['startTime']
            games[id] = timetools.format_date(date)
        
        return games

    def get_game_stats(self, season, game):
        glu = self.raw.get_lineup(season, game)
        home_team = glu['game']['homeTeam']['abbreviation']
        away_team = glu['game']['awayTeam']['abbreviation']
        
        for team in glu['teamLineups']:
            if team['team']['abbreviation'] == home_team:
                home_lineup = team['expected']
            if team['team']['abbreviation'] == away_team:
                away_lineup = team['expected']
        
        for player in home_lineup['lineupPositions']:
            if player['position'] == 'P':
                home_pitcher = player['player']['firstName']
                home_pitcher += '-'
                home_pitcher += player['player']['lastName']
                home_pitcher += '-'
                home_pitcher += str(player['player']['id'])
        
        for player in away_lineup['lineupPositions']:
            if player['position'] == 'P':
                away_pitcher = player['player']['firstName']
                away_pitcher += '-'
                away_pitcher += player['player']['lastName']
                away_pitcher += '-'
                away_pitcher += str(player['player']['id'])
        
        return [home_pitcher, away_pitcher]
    
    def get_current_era(self, season, pitcher, date):
        print(season, pitcher, date)
        sps = self.raw.get_season_player(season, pitcher, date)
        print(sps)

if __name__ == "__main__":
    stats = GetStats()
    season = '2017-regular'
    games = stats.get_game(season)
    for game_id in games:
        pitchers = stats.get_game_stats(season, int(game_id))
        for pitcher in pitchers:
            stats.get_current_era(season, pitcher, games[game_id])


