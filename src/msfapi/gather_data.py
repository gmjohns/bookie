import numpy as np
import pandas as pd
import timetools
from getrawdata import GetRawData

    
class GetStats:

    raw = GetRawData()
    def get_season(self, date):
        season_stats = {}
        s = self.raw.get_current_season(date)
        for season in s['seasons']:
            season_stats['title'] = season['slug']
            season_stats['start'] = season['startDate']
            season_stats['end'] = season['endDate']
        return season_stats

    def get_game(self, season):
        games = pd.DataFrame(columns=['id', 'date', 'prev_day'])
        # api call to get json formatted list of all games 
        season_games = self.raw.get_games(season['title'])

        for game in season_games['games']:
            start_time = game['schedule']['startTime']
            id = game['schedule']['id']
            date = timetools.format_datetime(start_time)
            # check if previous day falls within season range
            if timetools.prev_in_range(start_time, season['start'], season['end']):
                prev = timetools.get_previous_day(start_time)
            else:
                prev = None
            games = games.append({'date': date, 'id': id, 'prev_day': prev}, ignore_index=True)   
        return games

    def get_game_stats(self, season, game):
        glu = self.raw.get_lineup(season, game)
        lineup = {}
        home_team = glu['game']['homeTeam']['abbreviation']
        away_team = glu['game']['awayTeam']['abbreviation']
        lineup['home_team'] = home_team
        lineup['away_team'] = away_team
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
                lineup['home_pitcher'] = home_pitcher
        
        for player in away_lineup['lineupPositions']:
            if player['position'] == 'P':
                away_pitcher = player['player']['firstName']
                away_pitcher += '-'
                away_pitcher += player['player']['lastName']
                away_pitcher += '-'
                away_pitcher += str(player['player']['id'])
                lineup['away_pitcher'] = away_pitcher       
        return lineup
    
    def get_pitcher_stats(self, season, pitcher, date):
        pitcher_stats = {'ERA': None, 'IP': None}
        if date is not None:      
            sps = self.raw.get_season_player(season, pitcher, date)
            for stats in sps['playerStatsTotals']:
                pitcher_stats['ERA'] = stats['stats']['pitching']['earnedRunAvg']
                pitcher_stats['IP'] = stats['stats']['pitching']['inningsPitched']
        return pitcher_stats

    def get_team_stats(self, team, season, date):
        team_stats = {'AB': None, 'HR': None, 'WP': None}
        if date is not None:
            team = self.raw.get_team_stats(team, season, date)
            for stats in team['teamStatsTotals']:
                team_stats['AB'] = stats['stats']['batting']['atBats']
                team_stats['HR'] = stats['stats']['batting']['homeruns']
                team_stats['WP'] = stats['stats']['standings']['winPct']
        return team_stats

    def get_team_fir(self, team, season, date):
        fir = 0
        if date is not None:
            sg = self.raw.get_season_games(team, season, date)

def main():
    stats = GetStats()
    season = stats.get_season('20170701')
    # get previous season name and last day of previous season
    # some further logic could be implemented to save prev season values per team so as to limit the api calls per game
    prev_season = stats.get_season('20160701')
    prev_end_date = timetools.format_date(prev_season['end'])
    games = stats.get_game(season)
    stat_list = [
        'home_pitcher_curr_era', 
        'home_pitcher_prev_era', 
        'home_pitcher_curr_ip', 
        'home_pitcher_prev_ip', 
        'away_pitcher_curr_era', 
        'away_pitcher_prev_era', 
        'away_pitcher_curr_ip', 
        'away_pitcher_prev_ip', 
        'home_team_curr_win_pct',
        'home_team_prev_win_pct',
        'home_team_curr_fir',
        'home_team_prev_fir', 
        'home_team_curr_hr',
        'home_team_prev_hr',
        'away_team_curr_win_pct',
        'away_team_prev_win_pct',
        'away_team_curr_fir',
        'away_team_prev_fir',
        'away_team_curr_hr',
        'away_team_prev_hr'
        ]
    games_data = pd.DataFrame(columns=stat_list)
    for idx, game in games.head(10).iterrows():
        lineup = stats.get_game_stats(season['title'], game['id'])
        # get home and away pitcher statistics
        home_pitcher_stats = stats.get_pitcher_stats(season['title'], lineup['home_pitcher'], game['prev_day'])
        away_pitcher_stats = stats.get_pitcher_stats(season['title'], lineup['home_pitcher'], game['prev_day'])
        ls_home_pitcher_stats = stats.get_pitcher_stats(prev_season['title'], lineup['home_pitcher'], prev_end_date)
        ls_away_pitcher_stats = stats.get_pitcher_stats(prev_season['title'], lineup['away_pitcher'], prev_end_date)
        # get home and away team statistics
        home_team_stats = stats.get_team_stats(lineup['home_team'], season['title'], game['prev_day'])
        away_team_stats = stats.get_team_stats(lineup['away_team'], season['title'], game['prev_day'])
        ls_home_team_stats = stats.get_team_stats(lineup['home_team'], prev_season['title'], prev_end_date)
        ls_away_team_stats = stats.get_team_stats(lineup['away_team'], prev_season['title'], prev_end_date)
        # get first inning runs
        stats.get_team_fir(lineup['home_team'], season['title'], game['prev_day'])
        
        games_data = games_data.append({
        'home_pitcher_curr_era': home_pitcher_stats['ERA'], 
        'home_pitcher_prev_era': ls_home_pitcher_stats['ERA'],
        'home_pitcher_curr_ip': home_pitcher_stats['IP'], 
        'home_pitcher_prev_ip': ls_home_pitcher_stats['IP'],
        'away_pitcher_curr_era': away_pitcher_stats['ERA'], 
        'away_pitcher_prev_era': ls_away_pitcher_stats['ERA'],
        'away_pitcher_curr_ip': away_pitcher_stats['IP'],
        'away_pitcher_prev_ip': ls_away_pitcher_stats['IP'],
        'home_team_curr_win_pct': home_team_stats['WP'],
        'home_team_prev_win_pct': ls_home_team_stats['WP'],
        'home_team_curr_hr': home_team_stats['HR'],
        'home_team_prev_hr': ls_home_team_stats['HR'],
        'away_team_curr_win_pct': away_team_stats['WP'],
        'away_team_prev_win_pct': ls_away_team_stats['WP'],
        'away_team_curr_hr': away_team_stats['HR'],
        'away_team_prev_hr': ls_away_team_stats['HR']
        }, ignore_index=True)
    games_data.to_csv(season['title'] + '.csv', sep=',', index=False)

if __name__ == "__main__":
    main()
