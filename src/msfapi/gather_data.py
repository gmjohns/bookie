import numpy as np
import pandas as pd
import time
import timetools
from collections import defaultdict
from getrawdata import GetRawData
import pprint

pp = pprint.PrettyPrinter(indent=4)

class GetStats:
    def __init__(self, sleep_time):
        self.raw = GetRawData(sleep_time)

    def get_season(self, date):
        season_stats = {}
        s = self.raw.get_current_season(date)
        for season in s['seasons']:
            season_stats['title'] = season['slug']
            season_stats['start'] = season['startDate']
            season_stats['end'] = season['endDate']
        return season_stats

    def get_game(self, season):
        games = pd.DataFrame(columns=['date', 'id', 'fir_label'])
        # api call to get json formatted list of all games 
        season_games = self.raw.get_games(season['title'])
        for game in season_games['games']:
            start_time = game['schedule']['startTime']
            id = game['schedule']['id']
            date = timetools.format_datetime(start_time)
            first_inning = game['score']['innings'][0]
            if first_inning['awayScore'] + first_inning['homeScore'] >= 1:
                fir_label = 1
            else:
                fir_label = 0
            # check if previous day falls within season range
            games = games.append({'date': date, 'id': id, 'fir_label': fir_label}, ignore_index=True)   
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
                    home_pitcher = str(player['player']['id'])
                    lineup['home_pitcher'] = home_pitcher
            
            for player in away_lineup['lineupPositions']:
                if player['position'] == 'P':
                    away_pitcher = str(player['player']['id'])
                    lineup['away_pitcher'] = away_pitcher   
        except:
            lineup['home_pitcher'] = 'NA'
            lineup['away_pitcher'] = 'NA'

        return lineup
    
    def get_pitcher_stats(self, season, pitchers, date, lineups, data):  
        pitcher_str = ','.join(pitchers)   
        if date != 'NA':      
            sps = self.raw.get_season_player(season, pitcher_str, date)
            for player in sps['playerStatsTotals']:
                # should add check of player-id here
                try:
                    pitcher = str(player['player']['id'])
                    
                    for idx in range(len(lineups[pitcher]['game_id'])):
                        game_id = lineups[pitcher]['game_id'][idx]
                        data[game_id][lineups[pitcher]['title'][idx]]['ERA'] = player['stats']['pitching']['earnedRunAvg']
                        data[game_id][lineups[pitcher]['title'][idx]]['IP'] = player['stats']['pitching']['inningsPitched']
                except:
                    continue

    def get_prev_pitcher_stats(self, season, pitchers, lineups, data):
        pitcher_str = ','.join(pitchers)
        sps = self.raw.get_season_player(season, pitcher_str, None)
        for player in sps['playerStatsTotals']:
            # should add check of player-id here
            try:
                pitcher = str(player['player']['id'])

                for idx in range(len(lineups[pitcher]['game_id'])):
                    game_id = lineups[pitcher]['game_id'][idx]
                    data[game_id][lineups[pitcher]['title'][idx]]['prev_ERA'] = player['stats']['pitching']['earnedRunAvg']
                    data[game_id][lineups[pitcher]['title'][idx]]['prev_IP'] = player['stats']['pitching']['inningsPitched']
            except:
                continue

    def get_team_stats(self, teams, season, date, lineups, data):
        team_str = ','.join(teams)
        if date != 'NA':
            team = self.raw.get_team_stats(team_str, season, date)
            for team in team['teamStatsTotals']:
                try:
                    team_abr = team['team']['abbreviation']

                    for idx in range(len(lineups[team_abr]['game_id'])):
                        game_id = lineups[team_abr]['game_id'][idx]
                        data[game_id][lineups[team_abr]['title'][idx]]['AB'] = team['stats']['batting']['atBats']
                        data[game_id][lineups[team_abr]['title'][idx]]['SLG'] = team['stats']['batting']['batterSluggingPct']
                        data[game_id][lineups[team_abr]['title'][idx]]['WP'] = team['stats']['standings']['winPct']
                except:
                    continue

    def get_prev_team_stats(self, teams, season, lineups, data):
        team_str = ','.join(teams)
        prev = self.raw.get_team_stats(team_str, season, None)
        for team in prev['teamStatsTotals']:
            try:
                team_abr = team['team']['abbreviation']
                for idx in range(len(lineups[team_abr]['game_id'])):
                    game_id = lineups[team_abr]['game_id'][idx]
                    data[game_id][lineups[team_abr]['title'][idx]]['prev_SLG'] = team['stats']['batting']['batterSluggingPct']
                    data[game_id][lineups[team_abr]['title'][idx]]['prev_WP'] = team['stats']['standings']['winPct']
            except:
                continue

    def get_team_fir(self, teams, season, date, lineups, data):
        fir = {}
        team_str = ','.join(teams)
        if date != 'NA' and date is not None:
            date = 'until-' + date
            for team in teams:
                fir[team] = []
            sg = self.raw.get_games(season, team_str, date)
            for game in sg['games']:
                try:
                    away = game['schedule']['awayTeam']['abbreviation']
                    home = game['schedule']['homeTeam']['abbreviation']
                    for inning in game['score']['innings']:
                        if inning['inningNumber'] == 1:
                            if away in fir.keys():
                                fir[away].append(inning['awayScore'])
                            if home in fir.keys():
                                fir[home].append(inning['homeScore'])
                except:
                    continue
            for team in teams:
                try:
                    for idx in range(len(lineups[team]['game_id'])):
                        game_id = lineups[team]['game_id'][idx]
                        data[game_id][lineups[team]['title'][idx]]['FIR'] = sum(fir[team]) / len(fir[team])
                except:
                    continue

    def get_prev_team_fir(self, teams, season, lineups, data):
        fir = {}
        team_str = ','.join(teams)     
        for team in teams:
            fir[team] = []
        sg = self.raw.get_games(season, team_str, None)
        for game in sg['games']:
            try:
                away = game['schedule']['awayTeam']['abbreviation']
                home = game['schedule']['homeTeam']['abbreviation']
                for inning in game['score']['innings']:
                    if inning['inningNumber'] == 1:
                        if away in fir.keys():
                            fir[away].append(inning['awayScore'])
                        if home in fir.keys():
                            fir[home].append(inning['homeScore'])
            except:
                continue

        for team in teams:
            try:
                for idx in range(len(lineups[team]['game_id'])):
                    game_id = lineups[team]['game_id'][idx]
                    data[game_id][lineups[team]['title'][idx]]['prev_FIR'] = sum(fir[team]) / len(fir[team])
            except:
                continue

def main():
    stats = GetStats(sleep_time=2)
    season = stats.get_season('20180701')
    # some further logic could be implemented to save prev season values per team so as to limit the api calls per game
    prev_season = stats.get_season('20170701')
    prev_end_date = timetools.format_date(prev_season['end'])

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
        'home_team_curr_fir_avg',
        'home_team_prev_fir_avg', 
        'home_team_curr_slg',
        'home_team_prev_slg',
        'away_team_curr_win_pct',
        'away_team_prev_win_pct',
        'away_team_curr_fir_avg',
        'away_team_prev_fir_avg',
        'away_team_curr_slg',
        'away_team_prev_slg',
        'fir_result'
        ]
    games_data = pd.DataFrame(columns=stat_list)
    dates = games['date'].unique()
    game_day = games.groupby(['date'])
    data = {}
    for idx, date in enumerate(dates):
        if timetools.prev_in_range(date, season['start'], season['end']):
            prev = timetools.get_previous_day(date)
            if timetools.prev_in_range(prev, season['start'], season['end']):
                prev_2 = timetools.get_previous_day(prev)
            else:
                prev_2 = 'NA'
        else:
            prev = 'NA'
            prev_2 = 'NA'
        daily_pitchers = []
        daily_teams = []
        lineups = defaultdict(lambda: defaultdict(list))
        day_games = game_day.get_group(date)
        print(idx, date)
        for num, game in day_games.iterrows():
            lineup = stats.get_game_stats(season['title'], game['id'])
            daily_pitchers.extend([lineup['home_pitcher'], lineup['away_pitcher']])
            daily_teams.extend([lineup['home_team'], lineup['away_team']])
            lineups[lineup['home_pitcher']]['title'].append('home_pitcher')
            lineups[lineup['home_pitcher']]['game_id'].append(game['id'])
            lineups[lineup['away_pitcher']]['title'].append('away_pitcher')
            lineups[lineup['away_pitcher']]['game_id'].append(game['id'])
            lineups[lineup['home_team']]['title'].append('home_team')
            lineups[lineup['home_team']]['game_id'].append(game['id'])
            lineups[lineup['away_team']]['title'].append('away_team')
            lineups[lineup['away_team']]['game_id'].append(game['id'])

            data[game['id']] = {}
            data[game['id']]['home_team'] = {}
            data[game['id']]['away_team'] = {}
            data[game['id']]['home_pitcher'] = {}
            data[game['id']]['away_pitcher'] = {}
            data[game['id']]['home_team']['abr'] = lineup['home_team']
            data[game['id']]['away_team']['abr'] = lineup['away_team']
            # initialize stats to -1
            data[game['id']]['home_pitcher']['ERA'] = -1
            data[game['id']]['home_pitcher']['prev_ERA'] = -1
            data[game['id']]['home_pitcher']['IP'] = -1
            data[game['id']]['home_pitcher']['prev_IP'] = -1
            data[game['id']]['away_pitcher']['ERA'] = -1
            data[game['id']]['away_pitcher']['prev_ERA'] = -1
            data[game['id']]['away_pitcher']['IP'] = -1
            data[game['id']]['away_pitcher']['prev_IP'] = -1
            data[game['id']]['home_team']['AB'] = -1
            data[game['id']]['home_team']['SLG'] = -1
            data[game['id']]['home_team']['prev_SLG'] = -1
            data[game['id']]['home_team']['WP'] = -1
            data[game['id']]['home_team']['prev_WP'] = -1
            data[game['id']]['home_team']['FIR'] = -1
            data[game['id']]['home_team']['prev_FIR'] = -1
            data[game['id']]['away_team']['AB'] = -1
            data[game['id']]['away_team']['SLG'] = -1
            data[game['id']]['away_team']['prev_SLG'] = -1
            data[game['id']]['away_team']['WP'] = -1
            data[game['id']]['away_team']['prev_WP'] = -1
            data[game['id']]['away_team']['FIR'] = -1
            data[game['id']]['away_team']['prev_FIR'] = -1
        stats.get_pitcher_stats(season['title'], daily_pitchers, prev_2, lineups, data)
        stats.get_prev_pitcher_stats(prev_season['title'], daily_pitchers, lineups, data)
        stats.get_team_stats(daily_teams, season['title'], prev_2, lineups, data)
        stats.get_prev_team_stats(daily_teams, prev_season['title'], lineups, data)
        stats.get_team_fir(daily_teams, season['title'], prev, lineups, data)
        stats.get_prev_team_fir(daily_teams, prev_season['title'], lineups, data)

        for idx, game in day_games.iterrows():
            data[game['id']]
            games_data = games_data.append({
                'date': date,
                'home_team': data[game['id']]['home_team']['abr'],
                'away_team': data[game['id']]['away_team']['abr'],
                'home_pitcher_curr_era': data[game['id']]['home_pitcher']['ERA'], 
                'home_pitcher_prev_era': data[game['id']]['home_pitcher']['prev_ERA'], 
                'home_pitcher_curr_ip': data[game['id']]['home_pitcher']['IP'], 
                'home_pitcher_prev_ip': data[game['id']]['home_pitcher']['prev_IP'],
                'away_pitcher_curr_era': data[game['id']]['away_pitcher']['ERA'],
                'away_pitcher_prev_era': data[game['id']]['away_pitcher']['prev_ERA'],
                'away_pitcher_curr_ip': data[game['id']]['away_pitcher']['IP'],
                'away_pitcher_prev_ip': data[game['id']]['away_pitcher']['prev_IP'],
                'home_team_curr_win_pct': data[game['id']]['home_team']['WP'],
                'home_team_prev_win_pct': data[game['id']]['home_team']['prev_WP'],
                'home_team_curr_fir_avg': data[game['id']]['home_team']['FIR'],
                'home_team_prev_fir_avg': data[game['id']]['home_team']['prev_FIR'],
                'home_team_curr_slg': data[game['id']]['home_team']['SLG'],
                'home_team_prev_slg': data[game['id']]['home_team']['prev_SLG'],
                'away_team_curr_win_pct': data[game['id']]['away_team']['WP'],
                'away_team_prev_win_pct': data[game['id']]['away_team']['prev_WP'],
                'away_team_curr_fir_avg': data[game['id']]['away_team']['FIR'],
                'away_team_prev_fir_avg': data[game['id']]['away_team']['prev_FIR'],
                'away_team_curr_slg': data[game['id']]['away_team']['SLG'],
                'away_team_prev_slg': data[game['id']]['away_team']['prev_SLG'],
                'fir_result': game['fir_label']
            }, ignore_index=True)

    games_data.to_csv(season['title'] + 'LabeledRaw.csv', sep=',', index=False)

if __name__ == "__main__":
    main()
