from apisetup import MySportsFeeds


class GameBoxscore():

    def get_boxscore(self, game_id):
        msf = MySportsFeeds(version="1.2")
        msf.authenticate("a91b62dd-084f-4eb8-b1ed-cdcd1c")

        return msf.msf_get_data(league='mlb',season='latest',feed='game_boxscore',gameid='51265',format='json')
