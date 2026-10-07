"""Small helpers used across lair."""


def updating_print(msg):
    """Print *msg* over the current line (a carriage return, no newline)."""
    print(f"\r{msg}", end="")


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            # AttributeError keeps hasattr, getattr(..., default), copy working
            raise AttributeError(key) from None
        if type(val) is dict:
            # Store the wrapped dict back so writes like `d.a.b = 2` stick
            val = DotDict(val)
            self[key] = val
        return val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return dir(dict) + list(self.keys())
