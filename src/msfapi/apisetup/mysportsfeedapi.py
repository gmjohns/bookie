from apisetup.v2_1 import API_v2_1


### Main class for all interaction with the MySportsFeeds API
class MySportsFeeds(object):

    # Constructor
    def __init__(self, version='2.1', verbose=False, store_type=None, store_location=None):
        self.version = version
        self.verbose = verbose
        self.store_type = None
        self.store_location = None

        # Instantiate an instance of the appropriate API depending on version
        if self.version == '2.1':
            self.api_instance = API_v2_1(self.verbose, self.store_type, self.store_location)

    # Make sure the version is supported
    def __verify_version(self, version):
        if version != '2.1':
            raise ValueError("Unrecognized version specified.  Supported versions are: '2.1'")

    # Verify the type and location of the stored data
    def __verify_store(self, store_type, store_location):
        if store_type != None and store_type != 'file':
            raise ValueError("Unrecognized storage type specified.  Supported values are: None,'file'")

        if store_type == 'file':
            if store_location == None:
                raise ValueError("Must specify a location for stored data.")

    # Authenticate against the API (for v1.0)
    def authenticate(self, apikey, password='Thebigbob11'):
        if not self.api_instance.supports_basic_auth():
            raise ValueError("BASIC authentication not supported for version " + self.version)

        self.api_instance.set_auth_credentials(apikey, password)

    # Request data (and store it if applicable)
    def msf_get_data(self, **kwargs):
        return self.api_instance.get_data(**kwargs)
        