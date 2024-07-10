
class Network:  # TODO
    # this might need to be an xarray dataarray with dims of time, lat, lon, site, (pollutant?)
    # but lat/lon of mobile is not standardized
    # could be a variable with dims of time, site
    'aggregate pollutant data from multiple sites into a single dataframe (wide or long format)'
    def __init__(self, pollutant: str):
        self.pollutant = pollutant
        self.sites = {}
        self.instruments = set()

    def add_site(self, site: Site):
        self.sites[site.SID] = site
        self.instruments.update(site.instruments)

    def remove_site(self, SID: str):
        del self.sites[SID]
        # remove instruments from set FIXME
        self.instruments = set()
        for site in self.sites.values():
            self.instruments.update(site.instruments)

    def get_obs(self, format='wide', group=None, time_range=None, num_processes=1):
        'get observations for all sites in network'
        # should I allow for sites to be removed from network here or does it need to be done beforehand
        # want to be able to specify subnetworks such as SLV or uinta etc.
        # maybe even subseting by lat/lon bounds but would be complicated with mobile...
        # probably just let user decide which sites
        # 
        pass