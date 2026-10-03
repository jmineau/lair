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
        assert isinstance(
            plotter.terrain_cmap(), matplotlib.colors.LinearSegmentedColormap
        )


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

    @pytest.mark.parametrize("min_bin", [1, 3])
    def test_polar_plot_min_bin_is_inclusive(self, rng, monkeypatch, min_bin):
        # min_bin is the minimum count a bin needs to be plotted: bins with
        # exactly min_bin observations are kept.
        import lair.air

        df = pd.DataFrame(
            {
                "ws": rng.uniform(0, 10, 500),
                "wd": rng.uniform(0, 360, 500),
                "CH4": rng.normal(2, 0.3, 500),
            }
        )
        captured = {}
        circularize = lair.air.circularize_radial_data

        def spy(agg):
            captured["agg"] = agg
            return circularize(agg)

        monkeypatch.setattr(lair.air, "circularize_radial_data", spy)
        plotter.polarPlot(df, "CH4", min_bin=min_bin)

        counts = (
            lair.air.bin_polar(df, xbins=30)
            .groupby(["radian_bin", "x_bin"], observed=True)["CH4"]
            .count()
        )
        assert (counts == min_bin).any()  # the boundary case is exercised
        n_plotted = int(captured["agg"].notna().to_numpy().sum())
        assert n_plotted == int((counts >= min_bin).sum())

    def test_polar_freq(self, rng):
        df = pd.DataFrame(
            {"ws": rng.uniform(0, 10, 500), "wd": rng.uniform(0, 360, 500)}
        )
        ax = plotter.polarFreq(df)
        assert ax.name == "polar"

    @staticmethod
    def _spy_grid(monkeypatch):
        """Capture the (direction, speed) grid each polar plot contours."""
        import lair.air

        captured = {}
        circularize = lair.air.circularize_radial_data

        def spy(agg):
            captured["agg"] = agg
            return circularize(agg)

        monkeypatch.setattr(lair.air, "circularize_radial_data", spy)
        return captured

    @staticmethod
    def _spy_contourf(monkeypatch):
        """Capture the (theta, r, c) arrays each polar plot passes to contourf."""
        from matplotlib.projections.polar import PolarAxes

        captured = {}
        contourf = PolarAxes.contourf

        def spy(self, theta, r, c, *args, **kwargs):
            captured.update(theta=theta, r=r, c=c)
            return contourf(self, theta, r, c, *args, **kwargs)

        monkeypatch.setattr(PolarAxes, "contourf", spy)
        return captured

    #: The 16 direction sectors (N, NNE, ..., NNW) in radians
    _sectors = np.deg2rad(np.arange(16) * 22.5)

    # Speed edges 0-1-2-3-4: the (1, 2] bin is empty. Directions only N and E.
    _sparse = {
        "ws": [0.5, 0.0, 0.5, 2.5, 2.5, 4.0],
        "wd": [90.0, 90.0, 0.0, 0.0, 90.0, 0.0],
        "CH4": [2.0, 6.0, 1.0, 3.0, 4.0, 5.0],
    }

    @pytest.mark.filterwarnings("error::FutureWarning")
    def test_polar_plot_grid_keeps_empty_speed_bins(self, monkeypatch):
        # pandas 3 defaults to groupby(observed=True), which dropped the empty
        # speed bin from the grid; the plotted grid must not depend on pandas
        captured = self._spy_grid(monkeypatch)
        df = pd.DataFrame(self._sparse)
        before = df.copy()
        plotter.polarPlot(df, "CH4", xbins=[0, 1, 2, 3, 4])
        pd.testing.assert_frame_equal(df, before)  # caller's frame untouched
        agg = captured["agg"]
        np.testing.assert_allclose(agg.index.to_numpy(dtype=float), self._sectors)
        assert agg.columns.tolist() == [1, 2, 3, 4]
        expected = np.full((16, 4), np.nan)
        expected[0] = [1.0, np.nan, 3.0, 5.0]  # N
        expected[4] = [4.0, np.nan, 4.0, np.nan]  # E
        np.testing.assert_array_equal(agg.to_numpy(dtype=float), expected)

    @pytest.mark.filterwarnings("error::FutureWarning")
    def test_polar_freq_grid_keeps_empty_speed_bins(self, monkeypatch):
        captured = self._spy_grid(monkeypatch)
        df = pd.DataFrame(self._sparse)
        before = df.copy()
        plotter.polarFreq(df, xbins=[0, 1, 2, 3, 4])
        pd.testing.assert_frame_equal(df, before)  # no 'count' etc. added
        agg = captured["agg"]
        np.testing.assert_allclose(agg.index.to_numpy(dtype=float), self._sectors)
        assert agg.columns.tolist() == [1, 2, 3, 4]
        expected = np.zeros((16, 4))
        expected[0] = [1, 0, 1, 1]  # N
        expected[4] = [2, 0, 1, 0]  # E
        np.testing.assert_array_equal(agg.to_numpy(dtype=float), expected)

    @pytest.mark.parametrize("plot", ["polarPlot", "polarFreq"])
    def test_polar_grid_wraps_at_north_with_empty_sectors(self, monkeypatch, plot):
        # Data only at N and E: the 14 empty sectors must stay in the grid, or
        # the wrap-around row lands at 180 deg instead of closing at 360 (#58)
        captured = self._spy_contourf(monkeypatch)
        df = pd.DataFrame(self._sparse)
        if plot == "polarPlot":
            plotter.polarPlot(df, "CH4", xbins=[0, 1, 2, 3, 4])
        else:
            plotter.polarFreq(df, xbins=[0, 1, 2, 3, 4])
        theta, c = captured["theta"], captured["c"]
        assert theta.shape == (17, 4)  # 16 sectors + the closing row
        np.testing.assert_allclose(theta[:, 0], np.append(self._sectors, 2 * np.pi))
        np.testing.assert_array_equal(c[-1], c[0])  # closes back onto N
        assert np.isfinite(c[[0, 4]]).any(axis=1).all()  # N and E have data
        assert np.isnan(np.delete(c[:-1], [0, 4], axis=0)).all()  # others empty

    def test_windvector_plot(self, rng):
        df = pd.DataFrame(
            {"WD": rng.uniform(0, 360, 24), "WS": rng.uniform(0, 10, 24)},
            index=pd.date_range("2024-01-01", periods=24, freq="h"),
        )
        ax = plotter.windvectorPlot(df)
        assert ax.has_data()
