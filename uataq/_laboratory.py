"""
lair.uataq._laboratory
~~~~~~~~~~~~~~~~~~~~~

This module provides a factory class for creating site objects with instruments from a configuration file.
"""

from copy import deepcopy
import json
import os

from lair.config import LAIR_DIR, vprint
from lair.uataq.instruments import InstrumentEnsemble
from lair.uataq import sites


class Laboratory:
    '''
    Factory class for creating site objects from config file.

    This class provides methods for creating site objects from a configuration file.
    The configuration file should contain information about the sites and their instruments.

    Attributes:
        site_config (str): The path to the site configuration file.

    Methods:
        get_site(SID): Returns a site object for the specified site ID.
    '''
    def __init__(self, config: str | dict = os.path.join(LAIR_DIR, 'uataq', 'config.json')):
        '''
        Initialize the Laboratory class.

        Parameters
        ----------
        config : str | dict
            The path to the configuration file or a dictionary containing the configuration data.

        Raises
        ------
        ValueError
            If the configuration data is invalid.
        '''
        if isinstance(config, str):
            # config is a file path
            with open(config) as config_file:
                self.config = json.load(config_file)
        elif isinstance(config, dict):
            # config is a dictionary
            self.config = config
        else:
            raise ValueError("Invalid configuration data. Must be a file path or dictionary.")

        self.sites = list(self.config.keys())
        self.instruments = list(set([instrument for SID in self.sites
                                     for instrument in self.config[SID].get('instruments', {}).keys()]))

    def __repr__(self):
        config = json.dumps(self.config, indent=4)
        return f"Laboratory(config={config})"

    def __str__(self):
        return "UATAQ Laboratory"

    def get_site(self, SID: str):
        '''
        Return site object from config file

        Parameters
        ----------
        SID : str
            The site ID.

        Returns
        -------
        Site
            A site object.

        Raises
        ------
        ValueError
            If the site ID is not found in the configuration file.
        ValueError
            If no instruments are found for the specified site.
        '''
        SID = SID.upper()
        config = deepcopy(self.config).get(SID)
        if config is None:
            raise ValueError(f"Site '{SID}' not found in the configuration file.")

        # Get site class based on config
        is_mobile = config.get('is_mobile', False)
        SiteClass = sites.MobileSite if is_mobile else sites.Site

        # If logger are specified at the site level, then pass them to the instrument ensemble.
        # If both are specified, the instrument ensemble will use the loggers specified at the instrument level.
        loggers = config.pop('loggers', None)

        # Build instrument ensemble based on config
        instrument_configs = config.pop('instruments', None)
        if instrument_configs is None:
            raise ValueError(f"No instruments found for site '{SID}'.")
        instruments = InstrumentEnsemble(SID, instrument_configs, loggers)

        return SiteClass(SID, config, instruments)


laboratory = Laboratory()

def get_site(SID: str) -> sites.Site:
    return laboratory.get_site(SID)
