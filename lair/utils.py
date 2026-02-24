"""
General utility functions and classes.
"""


def updating_print(msg):
    """
    Print a message that updates in place by overwriting the previous line.
    
    Uses carriage return to move cursor to beginning of line without newline.
    Useful for showing progress updates without cluttering the terminal.
    
    Parameters
    ----------
    msg : str
        Message to print.
        
    Examples
    --------
    >>> for i in range(100):
    ...     updating_print(f"Processing: {i}%")
    ...     time.sleep(0.1)
    
    Note
    ----
    Terminal compatibility may vary. Works best in standard terminals.
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
