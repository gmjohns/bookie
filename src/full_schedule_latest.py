from ohmysportsfeedspy import MySportsFeeds

msf = MySportsFeeds(version="1.2")
msf.authenticate("a91b62dd-084f-4eb8-b1ed-cdcd1c", "Thebigbob11")

output = msf.msf_get_data(league='mlb',season='latest',feed='full_game_schedule',format='csv')
