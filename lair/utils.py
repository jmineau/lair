"""
General utility functions and classes.
"""


def updating_print(msg):
    """
    Print a message that updates in place.
    
    Parameters
    ----------
    msg : str
        Message to print.
    """
    print(f'\r{msg}', end='')


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(*args):
        val = dict.__getitem__(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return dir(dict) + list(self.keys())
