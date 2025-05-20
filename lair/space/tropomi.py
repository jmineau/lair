import os
from sentinelsat import SentinelAPI

from lair.config import GROUP_DIR

TROPOMI_DIR = os.path.join(GROUP_DIR)


class TROPOMI:
    'https://www.star.nesdis.noaa.gov/atmospheric-composition-training/python_tropomi_level2_download.php'

    def __init__(self, product, latency,
                 bbox=None, extent=None):
        self.product = product
        self.latency = latency

        if not any([bbox, extent]):
            raise ValueError("Either 'bbox' or 'extent' must be provided.")
        if bbox:
            self.xmin = bbox[0]
            self.xmax = bbox[2]
            self.ymin = bbox[1]
            self.ymax = bbox[3]
        elif extent:
            self.xmin = extent[0]
            self.xmax = extent[1]
            self.ymin = extent[2]
            self.ymax = extent[3]

    @staticmethod
    def get_tropomi_product_abbreviation(product: str):
        abbreviations = {
            'CO': 'L2__CO____',
            'NO2': 'L2__NO2___',
            'SO2': 'L2__SO2___',
            'HCHO': 'L2__HCHO__',
            'AI': 'L2__AER_AI',
            'ALH': 'L2__AER_LH'
        }

        if not product in abbreviations:
            raise ValueError(f"Product '{product}' not recognized. Available products: {', '.join(abbreviations.keys())}")
        return abbreviations[product]


    def list_files(start_date, end_date, product_abbreviation, latency,
                   bbox=None, extent=None):
        """
        # TODO update this
        # Create list of TROPOMI data file names for user-entered product, latency, search region, and date range
        # "product_abbreviation": parameter variable from "get_tropomi_product_abbreviation(product)" function
        # "start_date", "end_date": parameter variables from "convert_date_sentinel_api_format(year, month, day)" function
        # "west_lon", "east_lon", "south_lat", "north_lat", "latency": parameter variables from widget menus, set in main function
        """
        if not any([bbox, extent]):
            raise ValueError("Either 'bbox' or 'extent' must be provided.")

        # Access S5P Data Hub using guest login credentials
        api = SentinelAPI('s5pguest', 's5pguest', 'https://s5phub.copernicus.eu/dhus')
    
        # Query API for specified region, start/end dates, data product
        footprint = "POLYGON((' + west_lon + ' ' + south_lat + ',' + east_lon + ' ' + south_lat + ',' + east_lon + ' ' + north_lat + ',' + west_lon + ' ' + north_lat + ',' + west_lon + ' ' + south_lat + '))"
        try:
            products = api.query(area=footprint, date=(start_date + 'T00:00:00Z', end_date + 'T23:59:59Z'), producttype=product_abbreviation, processingmode=latency)
        except:
            print('Error connecting to SciHub server. This happens periodically. Run code again.')
        
        # Convert query output to pandas dataframe (df) (part of Sentinelsat library)
        products_df = api.to_dataframe(products)
        
        # Extract data file names from dataframe to list
        if len(products_df) > 0:
            file_name_list = products_df['filename'].tolist()
            file_size_list = products_df['size'].tolist()
        else:
            file_name_list = []
            file_size_list = []
        
        return file_name_list, file_size_list, products

    def download(self, download_dir, product, 
                 start_date, end_date,
                 bbox=None, extent=None,
                 ):
        


class TROPOMI_GOSAT:
    """
    TROPOMI-GOSAT blended CH4 product

    https://doi.org/10.5194/amt-16-3787-2023
    """
