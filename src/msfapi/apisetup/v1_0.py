import os
import csv
import requests
from datetime import datetime
import simplejson as json
import platform
import base64

import ohmysportsfeedspy


# API class for dealing with v1.0 of the API
class API_v1_0(object):

    # Constructor
    def __init__(self, verbose, store_type=None, store_location=None):
        self.base_url = "https://api.mysportsfeeds.com/v1.0/pull"
        self.headers = {
            'Accept-Encoding': 'gzip',
            'User-Agent': 'MySportsFeeds Python/{} ({})'.format(ohmysportsfeedspy.__version__, platform.platform())
        }

        self.verbose = verbose
        self.store_type = store_type
        self.store_location = store_location

        self.valid_feeds = [
            'cumulative_player_stats',
            'full_game_schedule',
            'daily_game_schedule',
            'daily_player_stats',
            'game_boxscore',
            'scoreboard',
            'game_playbyplay',
            'player_gamelogs',
            'team_gamelogs',
            'roster_players',
            'game_startinglineup',
            'active_players',
            'overall_team_standings',
            'conference_team_standings',
            'division_team_standings',
            'playoff_team_standings',
            'player_injuries',
            'daily_dfs',
            'current_season',
            'latest_updates',
        ]

    # Verify a feed
    def __verify_feed(self, feedName):
        is_valid = False

        for feed in self.valid_feeds:
            if feed == feedName:
                is_valid = True
                break

        return is_valid

    # Verify output format
    def __verify_format(self, format):
        is_valid = True

        if format != 'json' and format != 'xml' and format != 'csv':
            is_valid = False

        return is_valid

    # Feed URL
    def determine_url(self, league, season, feed, output_format, params):
        if feed == "current_season":
            return "{base_url}/{league}/{feed}.{output}".format(base_url=self.base_url, feed=feed, league=league, season=season, output=output_format)
        else:
            return "{base_url}/{league}/{season}/{feed}.{output}".format(base_url=self.base_url, feed=feed, league=league, season=season, output=output_format)

    # Generate the appropriate filename for a feed request
    def __make_output_filename(self, league, season, feed, output_format, params):
        filename = "{feed}-{league}-{season}".format(league=league.lower(),
            season=season,
            feed=feed)

        if "gameid" in params:
            filename += "-" + params["gameid"]

        if "fordate" in params:
            filename += "-" + params["fordate"]

        filename += "." + output_format

        return filename

    # Save a feed response based on the store_type
    def __save_feed(self, response, league, season, feed, output_format, params):
        # Save to memory regardless of selected method
        if output_format == "json":
            store_output = response.json()
        elif output_format == "xml":
            store_output = response.text
        elif output_format == "csv":
            #store_output = response.content.split('\n')
            store_output = response.content.decode('utf-8')
            store_output = csv.reader(store_output.splitlines(), delimiter=',')
            store_output = list(store_output)

        if self.store_type == "file":
            if not os.path.isdir(self.store_location):
                os.mkdir(self.store_location)

            filename = self.__make_output_filename(league, season, feed, output_format, params)

            with open(self.store_location + filename, "w") as outfile:
                if output_format == "json":  # This is JSON
                    json.dump(store_output, outfile)

                elif output_format == "xml":  # This is xml
                    outfile.write(store_output)

                elif output_format == "csv":  # This is csv
                    writer = csv.writer(outfile)
                    for row in store_output:
                        writer.writerow([row])

                else:
                    raise AssertionError("Could not interpret feed output format")

    # Indicate this version does support BASIC auth
    def supports_basic_auth(self):
        return True

    # Establish BASIC auth credentials
    def set_auth_credentials(self, username, password):
        self.auth = (username, password)
        self.headers['Authorization'] = 'Basic ' + base64.b64encode('{}:{}'.format(username,password).encode('utf-8')).decode('ascii')

    # Request data (and store it if applicable)
    def get_data(self, **kwargs):
        if not self.auth:
            raise AssertionError("You must authenticate() before making requests.")

        # establish defaults for all variables
        league = ""
        season = ""
        feed = ""
        output_format = ""
        params = {}

        # iterate over args and assign vars
        for key, value in kwargs.items():
            if str(key) == 'league':
                league = value
            elif str(key) == 'season':
                if kwargs['feed'] == 'players':
                    params['season'] = value
                else:
                    season = value
            elif str(key) == 'feed':
                feed = value
            elif str(key) == 'format':
                output_format = value
            else:
                params[key] = value

        if self.__verify_feed(feed) == False:
            raise ValueError("Unknown feed '" + feed + "'.  Known values are: " + str(self.valid_feeds))

        if self.__verify_format(output_format) == False:
            raise ValueError("Unsupported format '" + output_format + "'.")

        url = self.determine_url(league, season, feed, output_format, params)

        if self.verbose:
            print("Making API request to '{}'.".format(url))
            print("  with headers:")
            print(self.headers)
            print(" and params:")
            print(params)

        r = requests.get(url, params=params, headers=self.headers)

        if r.status_code == 200:
            if self.store_type != None:
                self.__save_feed(r, league, season, feed, output_format, params)

            if output_format == "json":
                data = json.loads(r.content)
            elif output_format == "xml":
                data = str(r.content)
            else:
                data = r.content.splitlines()

        elif r.status_code == 304:
            if self.verbose:
                print("Data hasn't changed since last call")

            filename = self.__make_output_filename(league, season, feed, output_format, params)

            with open(self.store_location + filename) as f:
                if output_format == "json":
                    data = json.load(f)
                elif output_format == "xml":
                    data = str(f.readlines()[0])
                else:
                    data = f.read().splitlines()

        else:
            raise Warning("API call failed with error: {error}".format(error=r.status_code))

        return data