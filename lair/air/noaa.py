"""
lair.air.noaa
~~~~~~~~~~~~~

Module to get NOAA greenhouse gas data.
"""

def get_GML_ghg_data(site, parameter, type, level, filetype, download_dir,
                     lab_ID_num=1, measurement_group=None, qualifiers=None):

    host = 'ftp.gml.noaa.gov'

    category = 'greenhouse_gases'
    project = f'{level}-{type}'

    filename = f'{parameter}_{site}_{project}_{lab_ID_num}'

    if measurement_group is not None:
        filename += f'_{measurement_group}'

    if qualifiers is not None:
        filename += f'_{qualifiers}'

    filename = f'{filename}.{filetype}'

    path = f'data/{category}/{parameter}/{type}/{level}/{filetype}/{filename}'

    download_ftp_files(host, path, download_dir)

    local_file = os.path.join(download_dir, filename)

    if filetype == 'nc':
        data = xr.open_dataset(local_file)

        times = data.time.values
        data = data.drop_vars('time').assign_coords(time=('obs',times))

    elif filetype == 'txt':
        # Get number of lines to skip
        with open(local_file) as f:
            header_lines = int(f.readline().split(':')[1].strip())
            header_lines -= 1  # include column headers

        data = pd.read_csv(local_file, sep=' ', skiprows=header_lines,
                           parse_dates=['datetime'])
        data['datetime'] = data.datetime.dt.tz_localize(None)

    else:
        raise ValueError('filetype not implemented!')

    # Thought that I could save to a tmpdir and read from there, but the tmpdir
    #   tries to close while the data is loaded into xarray/pandas
    #   Possible solutions would be to make this function a context manager
    #   Or change the downloa_ftp_files function to read to an IO stream

    # with tempfile.TemporaryDirectory() as tmpdir:
    #     download_ftp_files(host, path, tmpdir)

    #     tmp_filepath = os.path.join(tmpdir, filename)
    #     if filetype == 'nc':
    #         data = xr.open_dataset(tmp_filepath)

    return data
