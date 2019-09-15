import numpy as np
import pandas as pd
import time
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
                prev = 'NA'
            games = games.append({'date': date, 'id': id, 'prev_day': prev}, ignore_index=True)   
        return games

    def get_game_stats(self, season, game):
        glu = self.raw.get_lineup(season, game)
        lineup = {}
        home_team = glu['game']['homeTeam']['abbreviation']
        away_team = glu['game']['awayTeam']['abbreviation']
        lineup['home_team'] = home_team
        lineup['away_team'] = away_team
        
        try:
            for team in glu['teamLineups']:
                if team['team']['abbreviation'] == home_team:
                    home_lineup = team['actual']
                if team['team']['abbreviation'] == away_team:
                    away_lineup = team['actual']

            for player in home_lineup['lineupPositions']:
                if player['position'] == 'P':
                    # remove all non-character values from string
                    home_pitcher = player['player']['firstName'].replace('.', '').replace(' ', '')
                    home_pitcher += '-'
                    home_pitcher += player['player']['lastName'].replace('.', '').replace(' ', '')
                    lineup['home_pitcher'] = home_pitcher
            
            for player in away_lineup['lineupPositions']:
                if player['position'] == 'P':
                    away_pitcher = player['player']['firstName'].replace('.', '').replace(' ', '')
                    away_pitcher += '-'
                    away_pitcher += player['player']['lastName'].replace('.', '').replace(' ', '')
                    lineup['away_pitcher'] = away_pitcher   
        except:
            lineup['home_pitcher'] = 'NA'
            lineup['away_pitcher'] = 'NA'

        return lineup
    
    def get_pitcher_stats(self, season, pitcher, date=None):
        pitcher_stats = {'ERA': 0, 'IP': 0}
        try:
            if date != 'NA':      
                sps = self.raw.get_season_player(season, pitcher, date)
                for stats in sps['playerStatsTotals']:
                    # should add check of player-id here
                    pitcher_stats['ERA'] = stats['stats']['pitching']['earnedRunAvg']
                    pitcher_stats['IP'] = stats['stats']['pitching']['inningsPitched']
        except:
            pitcher_stats['ERA'] = -1
            pitcher_stats['IP'] = -1
            
        return pitcher_stats

    def get_team_stats(self, team, season, date=None):
        team_stats = {'AB': 0, 'SLG': 0, 'WP': 0.5}
        try:
            if date != 'NA':
                team = self.raw.get_team_stats(team, season, date)
                for stats in team['teamStatsTotals']:
                    team_stats['AB'] = stats['stats']['batting']['atBats']
                    team_stats['SLG'] = stats['stats']['batting']['batterSluggingPct']
                    team_stats['WP'] = stats['stats']['standings']['winPct']
        except:
            team_stats['AB'] = -1
            team_stats['SLG'] = -1
            team_stats['WP'] = -1

        return team_stats

    def get_team_fir(self, team, season, date=None):
        fir = []
        count = 0
        try:
            if date != 'NA':
                if date is not None:
                    date = 'until-' + date
                sg = self.raw.get_games(season, team, date)
                for count, game in enumerate(sg['games']):
                    if game['schedule']['awayTeam']['abbreviation'] == team:
                        status = 'away'
                    else:
                        status = 'home'
                    for inning in game['score']['innings']:
                        if inning['inningNumber'] == 1:
                            score = inning[status + 'Score']
                            fir.append(score)
        except:
            return -1
        return sum(fir)/(count+1)


def main():
    stats = GetStats()
    season = stats.get_season('20170701')
    # some further logic could be implemented to save prev season values per team so as to limit the api calls per game
    prev_season = stats.get_season('20160701')
    prev_end_date = timetools.format_date(prev_season['end'])
    default_era = 6.00
    inn_min = 15
    games = stats.get_game(season)
    stat_list = [
        'date',
        'home_team',
        'away_team',
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
        'home_team_curr_fir_pct',
        'home_team_prev_fir_pct', 
        'home_team_curr_slg',
        'home_team_prev_slg',
        'away_team_curr_win_pct',
        'away_team_prev_win_pct',
        'away_team_curr_fir_pct',
        'away_team_prev_fir_pct',
        'away_team_curr_slg',
        'away_team_prev_slg'
        ]
    games_data = pd.DataFrame(columns=stat_list)
    for idx, game in games.iterrows():
        print(idx)
        lineup = stats.get_game_stats(season['title'], game['id'])
        print(lineup)
        # get home and away pitcher statistics
        home_pitcher_stats = stats.get_pitcher_stats(season['title'], lineup['home_pitcher'], game['prev_day'])
        away_pitcher_stats = stats.get_pitcher_stats(season['title'], lineup['home_pitcher'], game['prev_day'])
        ls_home_pitcher_stats = stats.get_pitcher_stats(prev_season['title'], lineup['home_pitcher'])
        ls_away_pitcher_stats = stats.get_pitcher_stats(prev_season['title'], lineup['away_pitcher'])
        if 0 <= ls_home_pitcher_stats['IP'] < inn_min:
            ls_home_pitcher_stats['ERA'] = default_era
        if 0 <= ls_away_pitcher_stats['IP'] < inn_min:
            ls_away_pitcher_stats['ERA'] = default_era
        if 0 <= home_pitcher_stats['IP'] < inn_min:
            home_pitcher_stats['ERA'] = ls_home_pitcher_stats['ERA']
        if 0 <= away_pitcher_stats['IP'] < inn_min:
            away_pitcher_stats['ERA'] = ls_away_pitcher_stats['ERA']
        # get home and away team statistics
        home_team_stats = stats.get_team_stats(lineup['home_team'], season['title'], game['prev_day'])
        away_team_stats = stats.get_team_stats(lineup['away_team'], season['title'], game['prev_day'])
        ls_home_team_stats = stats.get_team_stats(lineup['home_team'], prev_season['title'])
        ls_away_team_stats = stats.get_team_stats(lineup['away_team'], prev_season['title'])
        # get first inning runs
        home_team_fir = stats.get_team_fir(lineup['home_team'], season['title'], game['prev_day'])
        away_team_fir = stats.get_team_fir(lineup['away_team'], season['title'], game['prev_day'])
        ls_home_team_fir = stats.get_team_fir(lineup['home_team'], prev_season['title'])
        ls_away_team_fir = stats.get_team_fir(lineup['away_team'], prev_season['title'])

        games_data = games_data.append({
            'date': game['date'],
            'home_team': lineup['home_team'],
            'away_team': lineup['away_team'],
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
            'home_team_curr_fir_pct': home_team_fir,
            'home_team_prev_fir_pct': ls_home_team_fir,
            'home_team_curr_slg': home_team_stats['SLG'],
            'home_team_prev_slg': ls_home_team_stats['SLG'],
            'away_team_curr_win_pct': away_team_stats['WP'],
            'away_team_prev_win_pct': ls_away_team_stats['WP'],
            'away_team_curr_fir_pct': away_team_fir,
            'away_team_prev_fir_pct': ls_away_team_fir,
            'away_team_curr_slg': away_team_stats['SLG'],
            'away_team_prev_slg': ls_away_team_stats['SLG']
        }, ignore_index=True)
        if idx % 20 == 0 and idx != 0:
            time.sleep(80)
    games_data.to_csv(season['title'] + 'Raw.csv', sep=',', index=False)

if __name__ == "__main__":
    main()
