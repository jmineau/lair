"""
lair.uataq.errors
~~~~~~~~~~~~~~~~~

This module provides custom exceptions for UATAQ.
"""

class DataFileInitializationError(Exception):
    # Exception to catch custom data file initialization errors
    pass

class ParserError(Exception):
    # Exception to catch custom parsing errors
    pass


class ReaderError(Exception):
    # Exception to catch custom reading errors
    pass


class InactiveInstrumentError(ReaderError):
    def __init__(self, instrument):
        msg = f'{instrument} is inactive in given time_range'
        super().__init__(msg)


class InvalidGroupError(ReaderError):
    # Exception to catch invalid groups for an instrument
    pass


class InstrumentNotFoundError(Exception):
    def __init__(self, instrument, ensemble):
        msg = f'{instrument} not found in {ensemble}'
        super().__init__(msg)


class PollutantNotMeasured(Exception):
    def __init__(self, SID, pollutant):
        msg = f'{pollutant} not measured at {SID}'
        super().__init__(msg)
