"""Tests for lair.plotter.

Plotting functions are smoke-exercised on a headless (Agg) backend: they build
a figure on synthetic data and we assert they return an Axes without error.
NCL_cmap (network) and the HandlerDashedLines legend artist are not covered.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

plotter = pytest.importorskip("lair.plotter")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestColormapsAndFormatters:
    def test_log10formatter(self):
        assert plotter.log10formatter(2, None) == r"$10^{2}$"
        assert plotter.log10formatter(2.5, None, deci=1) == r"$10^{2.5}$"

    def test_truncate_colormap_from_name(self):
        cmap = plotter.truncate_colormap("viridis", 0.2, 0.8)
        assert isinstance(cmap, matplotlib.colors.LinearSegmentedColormap)

    def test_truncate_colormap_from_object(self):
        base = plt.get_cmap("plasma")
        cmap = plotter.truncate_colormap(base, 0.1, 0.9)
        assert isinstance(cmap, matplotlib.colors.LinearSegmentedColormap)

    def test_terrain_cmap(self):
        assert isinstance(plotter.terrain_cmap(), matplotlib.colors.LinearSegmentedColormap)


class TestPolarHelpers:
    def test_create_polar_ax(self):
        ax = plotter.create_polar_ax()
        assert ax.name == "polar"


class TestPlots:
    def test_diurnal_plot(self, rng):
        df = pd.DataFrame(
            {"CH4": rng.normal(2, 0.3, 240)},
            index=pd.date_range("2024-01-01", periods=240, freq="h"),
        )
        # Default freq must parse on pandas >= 3.0 (uppercase '1H' does not)
        ax = plotter.diurnalPlot(df, "CH4")
        assert ax.has_data()

    def test_diurnal_plot_does_not_mutate_stats(self, rng):
        df = pd.DataFrame(
            {"CH4": rng.normal(2, 0.3, 48)},
            index=pd.date_range("2024-01-01", periods=48, freq="h"),
        )
        stats = ["mean"]
        plotter.diurnalPlot(df, "CH4", stats=stats)
        assert stats == ["mean"]
        # Repeated default calls must not start plotting 'count'
        plotter.diurnalPlot(df, "CH4")
        ax = plotter.diurnalPlot(df, "CH4")
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "count" not in labels

    def test_diurnal_plot_accepts_str_stats_and_color(self, rng):
        df = pd.DataFrame(
            {"CH4": rng.normal(2, 0.3, 48)},
            index=pd.date_range("2024-01-01", periods=48, freq="h"),
        )
        ax = plotter.diurnalPlot(df, "CH4", stats="median", colors="red")
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["median"]

    def test_seasonal_plot(self, rng):
        df = pd.DataFrame(
            {"CH4": rng.normal(2, 0.2, 36)},
            index=pd.date_range("2022-01-31", periods=36, freq="ME"),
        )
        ax = plotter.seasonalPlot(df, "CH4")
        assert ax is not None

    def test_polar_plot(self, rng):
        df = pd.DataFrame(
            {
                "ws": rng.uniform(0, 10, 500),
                "wd": rng.uniform(0, 360, 500),
                "CH4": rng.normal(2, 0.3, 500),
            }
        )
        ax = plotter.polarPlot(df, "CH4")
        assert ax.name == "polar"

    def test_polar_freq(self, rng):
        df = pd.DataFrame(
            {"ws": rng.uniform(0, 10, 500), "wd": rng.uniform(0, 360, 500)}
        )
        ax = plotter.polarFreq(df)
        assert ax.name == "polar"

    def test_windvector_plot(self, rng):
        df = pd.DataFrame(
            {"WD": rng.uniform(0, 360, 24), "WS": rng.uniform(0, 10, 24)},
            index=pd.date_range("2024-01-01", periods=24, freq="h"),
        )
        ax = plotter.windvectorPlot(df)
        assert ax.has_data()
