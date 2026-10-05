"""Tests for lair.plotter.

Plotting functions run on a headless (Agg) backend against synthetic data;
tests check what is drawn (line data, fill bounds, labels, legend entries,
arrow components). NCL_cmap runs with pandas.read_csv stubbed to a local
table, so no network is needed.
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

    def test_truncate_colormap_spans_the_requested_range(self):
        base = plt.get_cmap("viridis")
        cmap = plotter.truncate_colormap(base, 0.2, 0.8, n=61)
        np.testing.assert_allclose(cmap(0.0), base(0.2), atol=1e-6)
        np.testing.assert_allclose(cmap(1.0), base(0.8), atol=1e-6)
        assert cmap.name == "trunc(viridis,0.20,0.80)"

    def test_ncl_cmap_reads_table_from_ncl_site(self, monkeypatch):
        import io

        # NCL .rgb tables: an ncolors line, a '# r g b' header, then 0-255 rows
        table = "ncolors= 3\n# r g b\n255 0 0\n0 255 0\n0 0 255\n"
        read_csv = pd.read_csv
        urls = []

        def fake_read_csv(path, **kwargs):
            urls.append(path)
            return read_csv(io.StringIO(table), **kwargs)

        monkeypatch.setattr(pd, "read_csv", fake_read_csv)
        cmap = plotter.NCL_cmap("BlueRed")
        assert urls == [
            "https://www.ncl.ucar.edu/Document/Graphics/ColorTables/Files/BlueRed.rgb"
        ]
        assert cmap.name == "BlueRed"
        assert cmap.N == 100
        np.testing.assert_allclose(cmap(0.0), (1, 0, 0, 1))
        np.testing.assert_allclose(cmap(0.5), (0, 1, 0, 1), atol=0.02)
        np.testing.assert_allclose(cmap(1.0), (0, 0, 1, 1))

    def test_terrain_cmap(self):
        assert isinstance(
            plotter.terrain_cmap(), matplotlib.colors.LinearSegmentedColormap
        )


class TestPolarHelpers:
    def test_create_polar_ax(self):
        ax = plotter.create_polar_ax()
        assert ax.name == "polar"

    @pytest.mark.parametrize("angle, ha", [(45.0, "left"), (270.0, "right")])
    def test_format_radial_axis(self, angle, ha):
        ax = plotter.create_polar_ax()
        ax.set_ylim(0, 8)
        plotter.format_radial_axis(ax, "WS [m/s]", angle)
        assert ax.get_rlabel_position() == pytest.approx(angle)
        (label,) = [t for t in ax.texts if t.get_text() == "WS [m/s]"]
        theta, r = label.get_position()
        assert theta == pytest.approx(np.deg2rad(angle))
        assert r == pytest.approx(max(ax.get_yticks()))
        assert label.get_ha() == ha

    def test_format_radial_axis_keeps_default_angle(self):
        ax = plotter.create_polar_ax()
        before = ax.get_rlabel_position()
        plotter.format_radial_axis(ax, "WS", None)
        assert ax.get_rlabel_position() == pytest.approx(before)


class TestDiurnalPlotValues:
    """diurnalPlot draws the hourly statistics of the data it is given."""

    @staticmethod
    def _two_days():
        # Day 1 = hour of day, day 2 = hour + 2 -> mean = median = hour + 1,
        # std (ddof=1) = sqrt(2) at every hour
        idx = pd.date_range("2024-01-01", periods=48, freq="h")
        return pd.DataFrame({"CH4": idx.hour + 2.0 * (idx.day - 1)}, index=idx)

    @staticmethod
    def _lines(ax):
        return {line.get_color(): line for line in ax.get_lines()}

    def test_mean_median_and_std_band(self):
        ax = plotter.diurnalPlot(self._two_days(), "CH4", units="ppm", tz="MST")
        lines = self._lines(ax)
        hours = np.arange(24)
        np.testing.assert_allclose(lines["black"].get_ydata(), hours + 1)  # mean
        np.testing.assert_allclose(lines["blue"].get_ydata(), hours + 1)  # median
        # x is the time of day (on a dummy date)
        x = pd.DatetimeIndex(lines["black"].get_xdata())
        assert list(x.hour) == list(hours)

        (band,) = ax.collections
        y = band.get_paths()[0].vertices[:, 1]
        assert y.min() == pytest.approx(1 - np.sqrt(2))
        assert y.max() == pytest.approx(24 + np.sqrt(2))

        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["median", r"mean $\pm$1$\sigma$"]
        assert ax.get_ylabel() == "CH4 [ppm]"
        assert ax.get_xlabel() == "Time [MST]"

    def test_count_is_plotted_when_asked_for(self):
        ax = plotter.diurnalPlot(
            self._two_days(), "CH4", stats=["count"], colors={"count": "red"}
        )
        (line,) = ax.get_lines()
        np.testing.assert_array_equal(line.get_ydata(), np.full(24, 2))
        assert ax.get_ylabel() == "CH4"  # no units given

    def test_min_count_blanks_sparse_hours(self):
        df = self._two_days()
        df = df[~((df.index.hour == 5) & (df.index.day == 2))]  # hour 5: 1 value
        ax = plotter.diurnalPlot(df, "CH4", stats=["mean"], min_count=2)
        y = ax.get_lines()[0].get_ydata()
        assert np.isnan(y[5])
        assert np.isfinite(np.delete(y, 5)).all()


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

    def test_seasonal_plot_djf_at_january_year(self):
        # Constant within each DJF so the line value identifies the season-year
        idx = pd.date_range("2023-01-31", "2024-12-31", freq="ME")
        djf_year = idx.year + (idx.month == 12)
        df = pd.DataFrame({"CH4": djf_year.astype(float)}, index=idx)
        ax = plotter.seasonalPlot(df, "CH4")
        lines = {line.get_label(): line for line in ax.get_lines()}
        x, y = lines["DJF"].get_data()
        assert list(x) == [2023, 2024, 2025]
        assert list(y) == [2023.0, 2024.0, 2025.0]

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

    def test_windvector_components(self):
        # From the north at 4 m/s blows toward -v; from the west at 2 m/s
        # toward +u. Arrows sit at (time, speed).
        idx = pd.date_range("2024-01-01", periods=2, freq="h")
        df = pd.DataFrame({"WD": [0.0, 270.0], "WS": [4.0, 2.0]}, index=idx)
        (q,) = plotter.windvectorPlot(df).collections
        np.testing.assert_allclose(q.U, [0.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(q.V, [-4.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(q.Y, [4.0, 2.0])

    def test_windvector_unit_length(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="h")
        df = pd.DataFrame({"WD": [0.0, 135.0, 270.0], "WS": [4.0, 7.0, 2.0]}, index=idx)
        (q,) = plotter.windvectorPlot(df, unit_length=True).collections
        np.testing.assert_allclose(np.hypot(q.U, q.V), 1.0)
        np.testing.assert_allclose(q.V[0], -1.0, atol=1e-12)  # direction kept


class TestHandlerDashedLines:
    """The legend handler draws one line per LineCollection segment."""

    @staticmethod
    def _artists(lc):
        from matplotlib.transforms import IdentityTransform

        fig, ax = plt.subplots()
        ax.add_collection(lc)
        handler = plotter.HandlerDashedLines()
        legend = ax.legend([lc], ["two styles"], handler_map={type(lc): handler})
        return handler.create_artists(
            legend, lc, 0.0, 0.0, 20.0, 9.0, 10.0, IdentityTransform()
        )

    def test_one_line_per_segment_with_its_style(self):
        from matplotlib.collections import LineCollection

        lc = LineCollection(
            [[(0, 0), (1, 0)], [(0, 1), (1, 1)]],
            colors=["red", "blue"],
            linestyles=["solid", "dashed"],
            linewidths=[3.0, 1.0],
        )
        top, bottom = self._artists(lc)
        assert matplotlib.colors.to_hex(top.get_color()) == "#ff0000"
        assert matplotlib.colors.to_hex(bottom.get_color()) == "#0000ff"
        assert top.get_linewidth() == 3.0
        assert bottom.get_linewidth() == 1.0
        # get_linestyle() reports '--' for any tuple style, so compare the
        # dash patterns: none (solid) on top, the collection's dashes below
        assert top._dash_pattern[1] is None
        assert bottom._dash_pattern[1] == pytest.approx(lc.get_dashes()[1][1])
        # Segments split the 9-unit-high box into thirds, first one on top
        assert set(top.get_ydata()) == {6.0}
        assert set(bottom.get_ydata()) == {3.0}

    def test_dash_length_matches_a_thick_dashed_segment(self):
        # The collection's dashes are already scaled by its linewidth; the
        # legend line must not scale them again (lw=3 gave 3x longer dashes)
        from matplotlib.collections import LineCollection

        lc = LineCollection(
            [[(0, 0), (1, 0)]], colors=["k"], linestyles=["dashed"], linewidths=[3.0]
        )
        (line,) = self._artists(lc)
        assert line._dash_pattern[1] == pytest.approx(lc.get_dashes()[0][1])

    def test_single_style_is_reused_for_every_segment(self):
        from matplotlib.collections import LineCollection

        lc = LineCollection(
            [[(0, 0), (1, 0)], [(0, 1), (1, 1)], [(0, 2), (1, 2)]],
            colors=["green"],
            linewidths=[2.0],
        )
        lines = self._artists(lc)
        assert len(lines) == 3
        assert {matplotlib.colors.to_hex(line.get_color()) for line in lines} == {
            "#008000"
        }
        assert {line.get_linewidth() for line in lines} == {2.0}

    def test_legend_renders(self):
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots()
        lc = LineCollection([[(0, 0), (1, 0)], [(0, 1), (1, 1)]], colors=["k", "r"])
        ax.add_collection(lc)
        legend = ax.legend(
            [lc], ["pair"], handler_map={LineCollection: plotter.HandlerDashedLines()}
        )
        fig.canvas.draw()
        assert [t.get_text() for t in legend.get_texts()] == ["pair"]
