"""
Metadata for common trace gases and aerosols: names, LaTeX labels, units.

A small registry of :class:`Pollutant` records for labeling plots and sanity
checking values. It knows nothing about any instrument or archive; how a
data source names its columns (e.g. UATAQ's ``CO2d_ppm_cal``) belongs with
that data source.

Examples
--------
>>> from lair.pollutants import get_pollutant
>>> ch4 = get_pollutant("ch4")
>>> ch4.label()
'$\\mathrm{CH_4}$ [ppm]'
>>> ax.set_ylabel(get_pollutant("PM25").label())  # doctest: +SKIP
"""

from dataclasses import dataclass

import pint

from lair import units as ureg
from lair._optional import import_optional_dependency

_UG_M3 = r"$\mu$g m$^{-3}$"
_NG_M3 = r"ng m$^{-3}$"


@dataclass(frozen=True)
class Pollutant:
    """
    Metadata for one pollutant.

    Attributes
    ----------
    name : str
        Canonical short name, e.g. ``'CH4'``, ``'NOx'``, ``'PM2.5'``.
    long_name : str
        Plain-language name, e.g. ``'methane'``.
    latex : str
        Matplotlib mathtext for the name, e.g. ``r'$\\mathrm{CH_4}$'``.
    units : str
        Usual units for ambient values, as plain text (``'ppm'``, ``'ppb'``,
        ``'ug/m3'``, ``'ng/m3'``). Gases are dry-air mole fractions.
    latex_units : str
        ``units`` as matplotlib mathtext.
    expected_range : tuple[float, float]
        Loose bounds on surface ambient values anywhere in the world, in
        ``units``: from clean background (and daytime CO2 drawdown over
        vegetation) up to polluted megacities, wildfire smoke and dust events.
        Meant for plot limits and quick sanity checks, not as QC thresholds.
    formula : str | None
        Chemical formula used for the molar mass; None for aerosols. NOx is
        given as NO2, the usual reporting basis for NOx mass.
    """

    name: str
    long_name: str
    latex: str
    units: str
    latex_units: str
    expected_range: tuple[float, float]
    formula: str | None = None

    @property
    def molar_mass(self) -> pint.Quantity:
        """
        Molar mass in g/mol. Needs the optional ``molmass`` dependency.

        Raises
        ------
        ValueError
            If the pollutant has no chemical formula (an aerosol).
        """
        if self.formula is None:
            raise ValueError(f"{self.name} has no chemical formula.")
        molmass = import_optional_dependency("molmass")
        return molmass.Formula(self.formula).mass * ureg("g/mol")

    def label(self, units: bool = True, latex: bool = True) -> str:
        """
        Build an axis label, e.g. ``'$\\mathrm{CH_4}$ [ppm]'``.

        Parameters
        ----------
        units : bool
            Append the units in brackets. Default True.
        latex : bool
            Use the mathtext forms. Default True; False gives ``'CH4 [ppm]'``.

        Returns
        -------
        str
            The label.
        """
        name = self.latex if latex else self.name
        if not units:
            return name
        return f"{name} [{self.latex_units if latex else self.units}]"


def _gas(
    name: str,
    long_name: str,
    latex: str,
    units: str,
    expected_range: tuple[float, float],
    formula: str | None = None,
) -> Pollutant:
    """A gas: its units need no mathtext, and its formula defaults to its name."""
    return Pollutant(
        name, long_name, latex, units, units, expected_range, formula or name
    )


#: Registry of known pollutants, keyed by canonical name.
POLLUTANTS: dict[str, Pollutant] = {
    p.name: p
    for p in [
        _gas("CO2", "carbon dioxide", r"$\mathrm{CO_2}$", "ppm", (350, 1000)),
        _gas("CH4", "methane", r"$\mathrm{CH_4}$", "ppm", (1.7, 5.0)),
        _gas("CO", "carbon monoxide", r"$\mathrm{CO}$", "ppb", (0, 5000)),
        _gas("H2O", "water vapor", r"$\mathrm{H_2O}$", "ppm", (0, 40000)),
        _gas("O3", "ozone", r"$\mathrm{O_3}$", "ppb", (0, 200)),
        _gas("NO", "nitric oxide", r"$\mathrm{NO}$", "ppb", (0, 500)),
        _gas("NO2", "nitrogen dioxide", r"$\mathrm{NO_2}$", "ppb", (0, 200)),
        _gas("NOx", "nitrogen oxides", r"$\mathrm{NO_x}$", "ppb", (0, 700), "NO2"),
        Pollutant("PM1", "PM1", r"$\mathrm{PM_{1}}$", "ug/m3", _UG_M3, (0, 300)),
        Pollutant("PM2.5", "PM2.5", r"$\mathrm{PM_{2.5}}$", "ug/m3", _UG_M3, (0, 500)),
        Pollutant("PM4", "PM4", r"$\mathrm{PM_{4}}$", "ug/m3", _UG_M3, (0, 700)),
        Pollutant("PM10", "PM10", r"$\mathrm{PM_{10}}$", "ug/m3", _UG_M3, (0, 1000)),
        Pollutant("BC", "black carbon", r"$\mathrm{BC}$", "ng/m3", _NG_M3, (0, 30000)),
    ]
}

# Spellings that don't survive upper-casing to a canonical name
_ALIASES = {"PM25": "PM2.5", "PM2_5": "PM2.5"}
_LOOKUP = {name.upper(): name for name in POLLUTANTS} | _ALIASES


def get_pollutant(name: str) -> Pollutant:
    """
    Look up a pollutant by name, case-insensitively.

    Parameters
    ----------
    name : str
        The pollutant, e.g. ``'CH4'``, ``'nox'``, ``'PM25'``.

    Returns
    -------
    Pollutant
        Its metadata.

    Raises
    ------
    KeyError
        If the pollutant is not in :data:`POLLUTANTS`.
    """
    canonical = _LOOKUP.get(name.strip().upper())
    if canonical is None:
        raise KeyError(f"Unknown pollutant '{name}'. Known: {list(POLLUTANTS)}.")
    return POLLUTANTS[canonical]
