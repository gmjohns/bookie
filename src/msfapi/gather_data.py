import numpy as np
import pandas as pd
import time
import timetools
from bs4 import BeautifulSoup
import lxml
import urllib.request
from collections import defaultdict
from getrawdata import GetRawData
import pprint
import re

pp = pprint.PrettyPrinter(indent=4)

class GetStats:
    convert = {
        'Braves':'ATL',
        'Marlins':'MIA',
        'Mets':'NYM',
        'Phillies':'PHI',
        'Nationals':'WAS',
        'Cubs':'CHC',
        'Reds':'CIN',
        'Brewers':'MIL',
        'Pirates':'PIT',
        'Cardinals':'STL',
        'Diamondbacks':'ARI',
        'Rockies':'COL',
        'Dodgers':'LAD',
        'Padres':'SD',
        'Giants':'SF',
        'Orioles':'BAL',
        'Red Sox':'BOS',
        'Yankees':'NYY',
        'Rays':'TB',
        'Blue Jays':'TOR',
        'White Sox':'CWS',
        'Indians':'CLE',
        'Tigers':'DET',
        'Royals':'KC',
        'Twins':'MIN',
        'Astros':'HOU',
        'Angels':'LAA',
        'Athletics':'OAK',
        'Mariners':'SEA',
        'Rangers':'TEX'
    }

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
        games = pd.DataFrame(columns=['date', 'time', 'id', 'fir_label'])
        # api call to get json formatted list of all games 
        season_games = self.raw.get_games(season['title'])
        for game in season_games['games']:
            start_time = game['schedule']['startTime']
            id = game['schedule']['id']
            time = timetools.format_time(start_time)
            date = timetools.format_datetime(start_time)
            first_inning = game['score']['innings'][0]
            if first_inning['awayScore'] + first_inning['homeScore'] >= 1:
                fir_label = 1
            else:
                fir_label = 0
            # check if previous day falls within season range
            games = games.append({'date': date, 'time': time, 'id': id, 'fir_label': fir_label}, ignore_index=True)   
        return games

    def get_game_stats(self, season, game, date, time):
        glu = self.raw.get_lineup(season, game)
        lineup = {}
        home_team = glu['game']['homeTeam']['abbreviation']
        away_team = glu['game']['awayTeam']['abbreviation']
        lineup['home_team'] = home_team
        lineup['away_team'] = away_team
        home_bo, away_bo = self.get_batting_order(date, time, home_team, away_team)
        for idx, player in enumerate(home_bo):
            lineup['home_BO' + str(idx+1)] = player
        for idx, player in enumerate(away_bo):
            lineup['away_BO' + str(idx+1)] = player
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
    
    
    def get_batting_order(self, date, time, home_team, away_team):
        response = urllib.request.urlopen('https://www.baseballpress.com/lineups/' + date)
        html = response.read()
        soup = BeautifulSoup(html, "lxml")
        home_bo = []
        away_bo = []
        for box in soup.find_all(class_='lineup-card'):
            teams = []
            for team in box.find_all(class_='mlb-team-logo bc'):
                teams.append(self.convert[team.get_text().strip()])
            datetime = box.find_all(class_='col col--min c')[1]
            datetime = pd.to_datetime(datetime.get_text().strip().splitlines()[1])
            if (teams == [away_team, home_team]):
                away = box.find_all(class_='col col--min')[0]
                home = box.find_all(class_='col col--min')[1]
                for player in away.find_all(class_='player'):
                    fn = player.get_text().strip().splitlines()[0].split()[1].replace('.', '')
                    ln = player.get_text().strip().splitlines()[0].split()[2]
                    ln = re.sub(r"(\w)([A-Z])(\.)", r"\1 \2", ln).split()[0]
                    away_bo.append(fn + '-' + ln)
                for player in home.find_all(class_='player'):
                    fn = player.get_text().strip().splitlines()[0].split()[1].replace('.', '')
                    ln = player.get_text().strip().splitlines()[0].split()[2]
                    ln = re.sub(r"(\w)([A-Z])(\.)", r"\1 \2", ln).split()[0]
                    home_bo.append(fn + '-' + ln)

        return home_bo, away_bo

    
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
                        data[game_id][lineups[pitcher]['title'][idx]]['HAND'] = player['player']['handedness']['throws']
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

    def get_batter_stats(self, season, batters, date, lineups, data):  
        batter_str = ','.join(batters)   
        if date != 'NA':      
            sps = self.raw.get_season_player(season, batter_str, date)
            for player in sps['playerStatsTotals']:
                # should add check of player-id here
                try:
                    fn = str(player['player']['firstName']).replace('.', '')
                    ln = str(player['player']['lastName']).replace('.', '')
                    batter = fn + '-' + ln
                    
                    for idx in range(len(lineups[batter]['game_id'])):
                        game_id = lineups[batter]['game_id'][idx]
                        data[game_id][lineups[batter]['title'][idx]]['OPS'] = player['stats']['batting']['batterOnBasePlusSluggingPct']
                        data[game_id][lineups[batter]['title'][idx]]['HAND'] = player['player']['handedness']['bats']
                except:
                    continue

    def get_prev_batter_stats(self, season, batters, lineups, data):  
        batter_str = ','.join(batters)   
        sps = self.raw.get_season_player(season, batter_str, None)
        for player in sps['playerStatsTotals']:
            # should add check of player-id here
            try:
                fn = str(player['player']['firstName']).replace('.', '')
                ln = str(player['player']['lastName']).replace('.', '')
                batter = fn + '-' + ln
                
                for idx in range(len(lineups[batter]['game_id'])):
                    game_id = lineups[batter]['game_id'][idx]
                    data[game_id][lineups[batter]['title'][idx]]['prev_OPS'] = player['stats']['batting']['batterOnBasePlusSluggingPct']
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
    
    def get_average_bo_stats(self, game):
        home_key_list = ['home_BO1', 'home_BO2', 'home_BO3', 'home_BO4', 'home_BO5']
        away_key_list = ['away_BO1', 'away_BO2', 'away_BO3', 'away_BO4', 'away_BO5']
        home_prev_key_list = ['home_BO1', 'home_BO2', 'home_BO3', 'home_BO4', 'home_BO5']
        away_prev_key_list = ['away_BO1', 'away_BO2', 'away_BO3', 'away_BO4', 'away_BO5']
        home_total = 0
        home_count = 0
        away_total = 0
        away_count = 0
        home_prev_total = 0
        home_prev_count = 0
        away_prev_total = 0
        away_prev_count = 0
        home_left_hand = 0
        home_right_hand = 0
        away_left_hand = 0
        away_right_hand = 0
        for key in home_key_list:
            if game[key]['OPS'] > -1:
                home_total += game[key]['OPS']
                home_count += 1
            if game[key]['HAND'] == 'L':
                home_left_hand += 1
            if game[key]['HAND'] == 'R':
                home_right_hand += 1

        
        for key in away_key_list:
            if game[key]['OPS'] > -1:
                away_total += game[key]['OPS']
                away_count += 1
            if game[key]['HAND'] == 'L':
                away_left_hand += 1
            if game[key]['HAND'] == 'R':
                away_right_hand += 1

        
        for key in home_prev_key_list:
            if game[key]['OPS'] > -1:
                home_prev_total += game[key]['prev_OPS']
                home_prev_count += 1
        
        for key in away_prev_key_list:
            if game[key]['OPS'] > -1:
                away_prev_total += game[key]['prev_OPS']
                away_prev_count += 1

        if home_count == 0:
            home_ops = -1
        else:
            home_ops = home_total / home_count
        if away_count == 0:
            away_ops = -1
        else:
            away_ops = away_total / away_count
        if home_prev_count == 0:
            home_prev_ops = -1
        else:
            home_prev_ops = home_prev_total / home_prev_count
        if away_prev_count == 0:
            away_prev_ops = -1
        else:
            away_prev_ops = home_prev_total / home_prev_count

        return home_ops, away_ops, home_prev_ops, away_prev_ops, home_left_hand, home_right_hand, away_left_hand, away_right_hand

def main():
    stats = GetStats(sleep_time=2)
    season = stats.get_season('20170701')
    # some further logic could be implemented to save prev season values per team so as to limit the api calls per game
    prev_season = stats.get_season('20160701')
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
        'home_curr_ops',
        'home_prev_ops',
        'away_curr_ops',
        'away_prev_ops',
        'home_pitcher_adv',
        'away_pitcher_adv',
        'home_team_curr_win_pct',
        'home_team_prev_win_pct',
        'home_team_curr_fir_avg',
        'home_team_prev_fir_avg', 
        'away_team_curr_win_pct',
        'away_team_prev_win_pct',
        'away_team_curr_fir_avg',
        'away_team_prev_fir_avg',
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
        daily_batters = []
        daily_teams = []
        lineups = defaultdict(lambda: defaultdict(list))
        day_games = game_day.get_group(date)
        print(idx, date)
        for num, game in day_games.iterrows():
            lineup = stats.get_game_stats(season['title'], game['id'], game['date'], game['time'])
            daily_pitchers.extend([lineup['home_pitcher'], lineup['away_pitcher']])
            daily_batters.extend([lineup['home_BO1'], 
                                lineup['home_BO2'], 
                                lineup['home_BO3'], 
                                lineup['home_BO4'], 
                                lineup['home_BO5'], 
                                lineup['away_BO1'], 
                                lineup['away_BO2'], 
                                lineup['away_BO3'], 
                                lineup['away_BO4'], 
                                lineup['away_BO5']])
            daily_teams.extend([lineup['home_team'], lineup['away_team']])
            lineups[lineup['home_pitcher']]['title'].append('home_pitcher')
            lineups[lineup['home_pitcher']]['game_id'].append(game['id'])
            lineups[lineup['away_pitcher']]['title'].append('away_pitcher')
            lineups[lineup['away_pitcher']]['game_id'].append(game['id'])
            lineups[lineup['home_team']]['title'].append('home_team')
            lineups[lineup['home_team']]['game_id'].append(game['id'])
            lineups[lineup['away_team']]['title'].append('away_team')
            lineups[lineup['away_team']]['game_id'].append(game['id'])
            lineups[lineup['home_BO1']]['title'].append('home_BO1')
            lineups[lineup['home_BO1']]['game_id'].append(game['id'])
            lineups[lineup['home_BO2']]['title'].append('home_BO2')
            lineups[lineup['home_BO2']]['game_id'].append(game['id'])
            lineups[lineup['home_BO3']]['title'].append('home_BO3')
            lineups[lineup['home_BO3']]['game_id'].append(game['id'])
            lineups[lineup['home_BO4']]['title'].append('home_BO4')
            lineups[lineup['home_BO4']]['game_id'].append(game['id']) 
            lineups[lineup['home_BO5']]['title'].append('home_BO5')
            lineups[lineup['home_BO5']]['game_id'].append(game['id'])
            lineups[lineup['away_BO1']]['title'].append('away_BO1')
            lineups[lineup['away_BO1']]['game_id'].append(game['id'])
            lineups[lineup['away_BO2']]['title'].append('away_BO2')
            lineups[lineup['away_BO2']]['game_id'].append(game['id'])
            lineups[lineup['away_BO3']]['title'].append('away_BO3')
            lineups[lineup['away_BO3']]['game_id'].append(game['id'])
            lineups[lineup['away_BO4']]['title'].append('away_BO4')
            lineups[lineup['away_BO4']]['game_id'].append(game['id'])
            lineups[lineup['away_BO5']]['title'].append('away_BO5')
            lineups[lineup['away_BO5']]['game_id'].append(game['id'])
            data[game['id']] = {}
            data[game['id']]['home_team'] = {}
            data[game['id']]['away_team'] = {}
            data[game['id']]['home_pitcher'] = {}
            data[game['id']]['away_pitcher'] = {}
            data[game['id']]['home_BO1'] = {}
            data[game['id']]['home_BO2'] = {}
            data[game['id']]['home_BO3'] = {}
            data[game['id']]['home_BO4'] = {}
            data[game['id']]['home_BO5'] = {}
            data[game['id']]['away_BO1'] = {}
            data[game['id']]['away_BO2'] = {}
            data[game['id']]['away_BO3'] = {}
            data[game['id']]['away_BO4'] = {}
            data[game['id']]['away_BO5'] = {}
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
            data[game['id']]['home_pitcher']['HAND'] = -1
            data[game['id']]['away_pitcher']['HAND'] = -1
            data[game['id']]['home_BO1']['OPS'] = -1
            data[game['id']]['home_BO1']['HAND'] = -1
            data[game['id']]['home_BO2']['OPS'] = -1
            data[game['id']]['home_BO2']['HAND'] = -1
            data[game['id']]['home_BO3']['OPS'] = -1
            data[game['id']]['home_BO3']['HAND'] = -1
            data[game['id']]['home_BO4']['OPS'] = -1
            data[game['id']]['home_BO4']['HAND'] = -1
            data[game['id']]['home_BO5']['OPS'] = -1
            data[game['id']]['home_BO5']['HAND'] = -1
            data[game['id']]['away_BO1']['OPS'] = -1
            data[game['id']]['away_BO1']['HAND'] = -1
            data[game['id']]['away_BO2']['OPS'] = -1
            data[game['id']]['away_BO2']['HAND'] = -1
            data[game['id']]['away_BO3']['OPS'] = -1
            data[game['id']]['away_BO3']['HAND'] = -1
            data[game['id']]['away_BO4']['OPS'] = -1
            data[game['id']]['away_BO4']['HAND'] = -1
            data[game['id']]['away_BO5']['OPS'] = -1
            data[game['id']]['away_BO5']['HAND'] = -1
            data[game['id']]['home_BO1']['prev_OPS'] = -1
            data[game['id']]['home_BO1']['prev_HAND'] = -1
            data[game['id']]['home_BO2']['prev_OPS'] = -1
            data[game['id']]['home_BO2']['prev_HAND'] = -1
            data[game['id']]['home_BO3']['prev_OPS'] = -1
            data[game['id']]['home_BO3']['prev_HAND'] = -1
            data[game['id']]['home_BO4']['prev_OPS'] = -1
            data[game['id']]['home_BO4']['prev_HAND'] = -1
            data[game['id']]['home_BO5']['prev_OPS'] = -1
            data[game['id']]['home_BO5']['prev_HAND'] = -1
            data[game['id']]['away_BO1']['prev_OPS'] = -1
            data[game['id']]['away_BO1']['prev_HAND'] = -1
            data[game['id']]['away_BO2']['prev_OPS'] = -1
            data[game['id']]['away_BO2']['prev_HAND'] = -1
            data[game['id']]['away_BO3']['prev_OPS'] = -1
            data[game['id']]['away_BO3']['prev_HAND'] = -1
            data[game['id']]['away_BO4']['prev_OPS'] = -1
            data[game['id']]['away_BO4']['prev_HAND'] = -1
            data[game['id']]['away_BO5']['prev_OPS'] = -1
            data[game['id']]['away_BO5']['prev_HAND'] = -1
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
        daily_batters1 = daily_batters[:len(daily_batters)//2]
        daily_batters2 = daily_batters[len(daily_batters)//2:]
        stats.get_pitcher_stats(season['title'], daily_pitchers, prev_2, lineups, data)
        stats.get_batter_stats(season['title'], daily_batters1, prev_2, lineups, data)
        stats.get_batter_stats(season['title'], daily_batters2, prev_2, lineups, data)
        stats.get_prev_pitcher_stats(prev_season['title'], daily_pitchers, lineups, data)
        stats.get_prev_batter_stats(prev_season['title'], daily_batters1, lineups, data)
        stats.get_prev_batter_stats(prev_season['title'], daily_batters2, lineups, data)
        stats.get_team_stats(daily_teams, season['title'], prev_2, lineups, data)
        stats.get_prev_team_stats(daily_teams, prev_season['title'], lineups, data)
        stats.get_team_fir(daily_teams, season['title'], prev, lineups, data)
        stats.get_prev_team_fir(daily_teams, prev_season['title'], lineups, data)

        for idx, game in day_games.iterrows():
            home_ops, away_ops, home_prev_ops, away_prev_ops, home_left_hand, home_right_hand, away_left_hand, away_right_hand, = stats.get_average_bo_stats(data[game['id']])
            if (home_right_hand >= 3 and data[game['id']]['home_pitcher']['HAND'] == 'R') or (home_left_hand >= 3 and data[game['id']]['home_pitcher']['HAND'] == 'L'):
                home_pitcher_adv = 1
            else:
                home_pitcher_adv = 0
            if data[game['id']]['home_pitcher']['HAND'] != 'L' and data[game['id']]['home_pitcher']['HAND'] != 'R':
                home_pitcher_adv = 1
            if (away_right_hand >= 3 and data[game['id']]['away_pitcher']['HAND'] == 'R') or (away_left_hand >= 3 and data[game['id']]['away_pitcher']['HAND'] == 'L'):
                away_pitcher_adv = 1
            else:
                away_pitcher_adv = 0
            if data[game['id']]['away_pitcher']['HAND'] != 'L' and data[game['id']]['away_pitcher']['HAND'] != 'R':
                away_pitcher_adv = 1

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
                'home_curr_ops': home_ops,
                'home_prev_ops': home_prev_ops,
                'away_curr_ops': away_ops,
                'away_prev_ops': away_prev_ops,
                'home_pitcher_adv': home_pitcher_adv,
                'away_pitcher_adv': away_pitcher_adv,
                'home_team_curr_win_pct': data[game['id']]['home_team']['WP'],
                'home_team_prev_win_pct': data[game['id']]['home_team']['prev_WP'],
                'home_team_curr_fir_avg': data[game['id']]['home_team']['FIR'],
                'home_team_prev_fir_avg': data[game['id']]['home_team']['prev_FIR'],
                'away_team_curr_win_pct': data[game['id']]['away_team']['WP'],
                'away_team_prev_win_pct': data[game['id']]['away_team']['prev_WP'],
                'away_team_curr_fir_avg': data[game['id']]['away_team']['FIR'],
                'away_team_prev_fir_avg': data[game['id']]['away_team']['prev_FIR'],
                'fir_result': game['fir_label']
            }, ignore_index=True)

    games_data.to_csv('data/' + season['title'] + 'LabeledRaw.csv', sep=',', index=False)

if __name__ == "__main__":
    main()
