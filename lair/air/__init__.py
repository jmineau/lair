import os
from lair.utils.records import ftp_download, unzip


def setup_ccg_filter():
    """
    Setup the CCG filter module from NOAA GML.
    Downloads the necessary files from the FTP server and unzip them if not already present.
    """
    # Define the path for the CCG filter file
    air_dir = os.path.dirname(__file__)
    ccg_filter_file = os.path.join(air_dir, '_ccg_filter.py')

    # Check if the CCG filter file already exists
    if not os.path.exists(ccg_filter_file):

        remote_zf = 'user/thoning/ccgcrv/ccg_filter.zip'
        zf = os.path.join(air_dir, 'ccg_filter.zip')

        # Download the zip file from the FTP server
        ftp_download('ftp.gml.noaa.gov', remote_zf, air_dir)

        # Unzip the downloaded file
        unzip(zf, air_dir)

        # Cleanup: remove the downloaded zip and unnecessary files
        os.remove(zf)
        os.remove(os.path.join(air_dir, 'ccg_dates.py'))
        os.remove(os.path.join(air_dir, 'ccgcrv.py'))

        # Rename the original file with leading underscore (private module)
        os.rename(os.path.join(air_dir, 'ccg_filter.py'), ccg_filter_file)


setup_ccg_filter()
