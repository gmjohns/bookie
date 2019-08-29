import numpy as np
from getrawdata import GetRawData

    
class GetStats:

    raw = GetRawData()
    
    def get_ids(self, season):
        ids = np.array([])
        # api call to get json formatted list of all games 
        games = self.raw.get_games(season)

        for game in games['games']:
            ids = np.append(ids, game['schedule']['id'])
        
        return ids

    def get_game_stats(self, season, id):
        glu = self.raw.get_lineup(season, id)
        home_team = glu['game']['homeTeam']['abbreviation']
        away_team = glu['game']['awayTeam']['abbreviation']
        
        for team in glu['teamLineups']:
            if team['team']['abbreviation'] == home_team:
                home_lineup = team['actual']
            if team['team']['abbreviation'] == away_team:
                away_lineup = team['actual']
        
        


    
if __name__ == "__main__":
    stats = GetStats()
    season = '2017-regular'
    ids = stats.get_ids(season)
    
    for id in ids[:1]:
        stats.get_game_stats(season, int(id))
