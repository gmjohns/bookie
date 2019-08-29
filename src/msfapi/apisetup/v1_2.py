import os
import csv
import requests
from datetime import datetime
import simplejson as json
import platform
import base64

from .v1_0 import API_v1_0


# API class for dealing with v1.2 of the API
class API_v1_2(API_v1_0):

    # Constructor
    def __init__(self, verbose, store_type=None, store_location=None):
        super().__init__(verbose, store_type, store_location)

        self.base_url = "https://api.mysportsfeeds.com/v1.2/pull"