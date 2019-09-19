import os
import csv
import requests
from datetime import datetime
import simplejson as json
import platform
import base64

import ohmysportsfeedspy
from .v1_0 import API_v1_0


# API class for dealing with v2.1 of the API
class API_v2_1(API_v1_0):

    # Constructor
    def __init__(self, verbose, store_type=None, store_location=None):
        super().__init__(verbose, store_type, store_location)

        self.base_url = "https://api.mysportsfeeds.com/v2.0/pull"

        self.valid_feeds = [
            'seasonal_games',
            'daily_games',
            'weekly_games',
            'seasonal_dfs',
            'daily_dfs',
            'weekly_dfs',
            'seasonal_player_gamelogs',
            'daily_player_gamelogs',
            'weekly_player_gamelogs',
            'seasonal_team_gamelogs',
            'daily_team_gamelogs',
            'weekly_team_gamelogs',
            'game_boxscore',
            'game_playbyplay',
            'game_lineup',
            'current_season',
            'player_injuries',
            'latest_updates',
            'seasonal_team_stats',
            'seasonal_player_stats',
            'seasonal_venues',
            'players',
            'seasonal_standings',
            'seasonal_game_lines',
            'daily_game_lines',
            'daily_futures'
        ]

    # Feed URL
    def determine_url(self, league, season, feed, output_format, params):
        if feed == "seasonal_games":
            if season == "":
                raise AssertionError("You must specify a season for this request.")

            return "{base_url}/{league}/{season}/games.{output}".format(base_url=self.base_url, league=league, season=season, output=output_format)

        elif feed == "daily_games":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            if not "date" in params:
                raise AssertionError("You must specify a 'date' param for this request.")

            return "{base_url}/{league}/{season}/date/{date}/games.{output}".format(base_url=self.base_url, league=league, season=season, date=params["date"], output=output_format)

        elif feed == "weekly_games":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            if not "week" in params:
                raise AssertionError("You must specify a 'week' param for this request.")

            return "{base_url}/{league}/{season}/week/{week}/games.{output}".format(base_url=self.base_url, league=league, season=season, week=params["week"], output=output_format)

        elif feed == "seasonal_dfs":
            if season == "":
                raise AssertionError("You must specify a season for this request.")

            return "{base_url}/{league}/{season}/dfs.{output}".format(base_url=self.base_url, league=league, season=season, output=output_format)

        elif feed == "daily_dfs":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            if not "date" in params:
                raise AssertionError("You must specify a 'date' param for this request.")

            return "{base_url}/{league}/{season}/date/{date}/dfs.{output}".format(base_url=self.base_url, league=league, season=season, date=params["date"], output=output_format)

        elif feed == "weekly_dfs":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            if not "week" in params:
                raise AssertionError("You must specify a 'week' param for this request.")

            return "{base_url}/{league}/{season}/week/{week}/dfs.{output}".format(base_url=self.base_url, league=league, season=season, week=params["week"], output=output_format)

        elif feed == "seasonal_player_gamelogs":
            if season == "":
                raise AssertionError("You must specify a season for this request.")

            return "{base_url}/{league}/{season}/player_gamelogs.{output}".format(base_url=self.base_url, league=league, season=season, output=output_format)

        elif feed == "daily_player_gamelogs":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            if not "date" in params:
                raise AssertionError("You must specify a 'date' param for this request.")

            return "{base_url}/{league}/{season}/date/{date}/player_gamelogs.{output}".format(base_url=self.base_url, league=league, season=season, date=params["date"], output=output_format)

        elif feed == "weekly_player_gamelogs":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            if not "week" in params:
                raise AssertionError("You must specify a 'week' param for this request.")

            return "{base_url}/{league}/{season}/week/{week}/player_gamelogs.{output}".format(base_url=self.base_url, league=league, season=season, week=params["week"], output=output_format)

        elif feed == "seasonal_team_gamelogs":
            if season == "":
                raise AssertionError("You must specify a season for this request.")

            return "{base_url}/{league}/{season}/team_gamelogs.{output}".format(base_url=self.base_url, league=league, season=season, output=output_format)

        elif feed == "daily_team_gamelogs":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            if not "date" in params:
                raise AssertionError("You must specify a 'date' param for this request.")

            return "{base_url}/{league}/{season}/date/{date}/team_gamelogs.{output}".format(base_url=self.base_url, league=league, season=season, date=params["date"], output=output_format)

        elif feed == "weekly_team_gamelogs":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            if not "week" in params:
                raise AssertionError("You must specify a 'week' param for this request.")

            return "{base_url}/{league}/{season}/week/{week}/team_gamelogs.{output}".format(base_url=self.base_url, league=league, season=season, week=params["week"], output=output_format)

        elif feed == "game_boxscore":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            if not "game" in params:
                raise AssertionError("You must specify a 'game' param for this request.")

            return "{base_url}/{league}/{season}/games/{game}/boxscore.{output}".format(base_url=self.base_url, league=league, season=season, game=params["game"], output=output_format)

        elif feed == "game_playbyplay":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            if not "game" in params:
                raise AssertionError("You must specify a 'game' param for this request.")

            return "{base_url}/{league}/{season}/games/{game}/playbyplay.{output}".format(base_url=self.base_url, league=league, season=season, game=params["game"], output=output_format)

        elif feed == "game_lineup":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            if not "game" in params:
                raise AssertionError("You must specify a 'game' param for this request.")
            
            return "{base_url}/{league}/{season}/games/{game}/lineup.{output}".format(base_url=self.base_url, league=league, season=season, game=params["game"], output=output_format)

        elif feed == "current_season":
            return "{base_url}/{league}/current_season.{output}".format(base_url=self.base_url, league=league, output=output_format)

        elif feed == "player_injuries":
            return "{base_url}/{league}/injuries.{output}".format(base_url=self.base_url, league=league, output=output_format)

        elif feed == "latest_updates":
            if season == "":
                raise AssertionError("You must specify a season for this request.")

            return "{base_url}/{league}/{season}/latest_updates.{output}".format(base_url=self.base_url, league=league, season=season, output=output_format)

        elif feed == "seasonal_team_stats":
            if season == "":
                raise AssertionError("You must specify a season for this request.")

            return "{base_url}/{league}/{season}/team_stats_totals.{output}".format(base_url=self.base_url, league=league, season=season, output=output_format)

        elif feed == "seasonal_player_stats":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            return "{base_url}/{league}/{season}/player_stats_totals.{output}".format(base_url=self.base_url, league=league, season=season, output=output_format)

        elif feed == "seasonal_venues":
            if season == "":
                raise AssertionError("You must specify a season for this request.")

            return "{base_url}/{league}/{season}/venues.{output}".format(base_url=self.base_url, league=league, season=season, output=output_format)

        elif feed == "players":
            return "{base_url}/{league}/players.{output}".format(base_url=self.base_url, league=league, output=output_format)

        elif feed == "seasonal_standings":
            if season == "":
                raise AssertionError("You must specify a season for this request.")

            return "{base_url}/{league}/{season}/standings.{output}".format(base_url=self.base_url, league=league, season=season, output=output_format)

        elif feed == "seasonal_game_lines":
            if season == "":
                raise AssertionError("You must specify a season for this request.")

            return "{base_url}/{league}/{season}/odds_gamelines.{output}".format(base_url=self.base_url, league=league, season=season, output=output_format)

        elif feed == "daily_game_lines":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            if "date" not in params:
                raise AssertionError("You must specify a 'date' param for this request.")
            return "{base_url}/{league}/{season}/date/{date}/odds_gamelines.{output}".format(base_url=self.base_url, league=league,
                                                                                             season=season, date=params["date"], output=output_format)

        elif feed == "daily_futures":
            if season == "":
                raise AssertionError("You must specify a season for this request.")
            if "date" not in params:
                raise AssertionError("You must specify a 'date' param for this request.")
            return "{base_url}/{league}/{season}/date/{date}/odds_futures.{output}".format(base_url=self.base_url, league=league,
                                                                                           season=season, date=params["date"], output=output_format)

        else:
            return ""