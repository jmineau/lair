from dataclasses import dataclass

@dataclass
class Observation:
    name: str
    units: str
    long_name: str
    latex: str
    variants: list = None
    exts: list = None
    prefixes: list = None
    alt_names: list = None
    pollutant: bool = False

    def get_regx(self):
        prefix = '|'.join(self.prefixes or '')
        name = '|'.join(self.alt_names or [self.name])
        variant = '|'.join(self.variants or '')
        ext = '|'.join(self.exts or '')

        pattern = fr"""
            ^(?:{prefix}){'_' if prefix else ''}
            (?:{name})(?:{variant})?_{self.units}
            {'_' if ext else ''}(?:{ext})$
            """

        return pattern

observations = {
    'CO2': Observation(
        name='CO2',
        units='ppm',
        long_name='Carbon Dioxide',
        latex='CO$_{2}$',
        variants=['d'],
        exts=['raw', 'cal']
        pollutant=True
    ),
}

observations = {
    'CO2': {
        'latex': 'CO_${2}$',
        'units': 'ppm',
        'long_name': 'Carbon Dioxide',
        'variants': ['d'],
        'exts': ['raw', 'cal']
    },
    'CH4': {
        'latex': 'CH$_{4}$',
        'units': 'ppm',
        'long_name': 'Methane',
        'variants': ['d'],
        'exts': ['raw', 'cal']
    },
    'O3': {
        'latex': 'O$_{3}$',
        'units': 'ppb',
        'long_name': 'Ozone',
    },
    'NO2': {
        'latex': 'NO$_{2}$',
        'units': 'ppb',
        'long_name': 'Nitrogen Dioxide',
    },
    'NO': {
        'units': 'ppb',
        'long_name': 'Nitric Oxide',
    },
    'NOx': {
        'latex': 'NO$_{x}$',
        'units': 'ppb',
        'long_name': 'Nitrogen Oxides',
    },
    'PM1': {
        'latex': 'PM$_{1}$',
        'units': 'ugm3',
        'units_latex': 'ug/m$^{3}$',
        'long_name': 'Particulate Matter < 1 micrometer',
        'long_name_latex': 'Particulate Matter$_{1}$',
    },
    'PM2.5': {
        'alt_names': ['PM_25'],
        'latex': 'PM$_{2.5}$',
        'units': 'ugm3',
        'units_latex': 'ug/m$^{3}$',
        'long_name': 'Particulate Matter < 2.5 micrometers',
        'long_name_latex': 'Particulate Matter$_{2.5}$',
        
    },
    'PM4': {
        'latex': 'PM$_{4}$',
        'units': 'ugm3',
        'units_latex': 'ug/m$^{3}$',
        'long_name': 'Particulate Matter < 4 micrometers',
        'long_name_latex': 'Particulate Matter$_{4}$',
    },
    'PM10': {
        'latex': 'PM$_{10}$',
        'units': 'ugm3',
        'units_latex': 'ug/m$^{3}$',
        'long_name': 'Particulate Matter < 10 micrometers',
        'long_name_latex': 'Particulate Matter$_{10}$',
    },
    'CO': {
        'units': 'ppb',
        'long_name': 'Carbon Monoxide',
    },
    'BC': {
        'units': 'ugm3',
        'units_latex': 'ug/m$^{3}$',
        'long_name': 'Black Carbon',
        'variants': ['1', '2', '3', '4', '5', '6', '7']
    },
    'T': {
        'units': 'C',
        'units_latex': 'C$^{\\circ}$',
        'long_name': 'Temperature',
        'prefixes': ['Ambient']
    },
    'P': {
        'alt_names': ['pres'],
        'units': 'hPa',
        'long_name': 'Pressure',
        'prefixes': ['Ambient']
    },
    'RH': {
        'units': 'pct',
        'units_latex': '%',
        'long_name': 'Relative Humidity',
        'prefixes': ['Ambient']
    },
}

def get_regx(ob: str):
    d = observations[ob.upper()]
    prefix = '|'.join(d.get('prefixes', ''))
    name = '|'.join(d.get('alt_names', ''))
    name = '|'.join([ob, name]) if name else ob
    variant = '|'.join(d.get('variants', ''))
    ext = '|'.join(d.get('exts', ''))

    pattern = fr"""
        ^(?:{prefix}){'_' if prefix else ''}
        (?:{name})(?:{variant})?_{d['units']}
        {'_' if ext else ''}(?:{ext})$
        """

    return pattern