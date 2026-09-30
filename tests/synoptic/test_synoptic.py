"""Tests for lair.synoptic, with the HTTP layer replaced by canned API payloads."""

import numpy as np
import pandas as pd
import pytest

from lair import synoptic

pytest.importorskip("requests")


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


@pytest.fixture
def api(monkeypatch):
    """Serve ``api.payload``; record each request's endpoint and params."""
    import requests

    class Api:
        payload: dict = {}
        calls: list = []

    def fake_get(url, params=None, timeout=None):
        Api.calls.append((url, dict(params)))
        return FakeResponse(Api.payload)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setenv("SYNOPTIC_TOKEN", "test-token")
    Api.calls = []
    return Api


OK = {"RESPONSE_CODE": 1, "RESPONSE_MESSAGE": "OK"}


def _station(stid, times, speed, direction):
    return {
        "STID": stid,
        "OBSERVATIONS": {
            "date_time": times,
            "wind_speed_set_1": speed,
            "wind_direction_set_1": direction,
        },
    }


class TestRequests:
    def test_token_from_env_and_time_format(self, api):
        api.payload = {"SUMMARY": OK, "STATION": []}
        synoptic.timeseries("2024-01-01", pd.Timestamp("2024-01-02 06:30"), stid="WBB")
        url, params = api.calls[0]
        assert url.endswith("stations/timeseries")
        assert params["token"] == "test-token"
        assert params["start"] == "202401010000" and params["end"] == "202401020630"
        assert params["stid"] == "WBB"

    def test_missing_token(self, api, monkeypatch):
        monkeypatch.delenv("SYNOPTIC_TOKEN")
        with pytest.raises(ValueError, match="SYNOPTIC_TOKEN"):
            synoptic.metadata(stid="WBB")

    def test_api_error_raises(self, api):
        api.payload = {"SUMMARY": {"RESPONSE_CODE": 2, "RESPONSE_MESSAGE": "no access"}}
        with pytest.raises(synoptic.SynopticError, match="no access"):
            synoptic.timeseries("2024-01-01", "2024-01-02", stid="KU42")


class TestTimeseries:
    def test_long_frame_across_stations(self, api):
        t = ["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"]
        api.payload = {
            "SUMMARY": OK,
            "STATION": [
                _station("A", t, [1.0, 2.0], [90, 90]),
                _station("B", t, [3.0, None], [180, None]),
                {"STID": "EMPTY", "OBSERVATIONS": {}},
            ],
        }
        df = synoptic.timeseries("2024-01-01", "2024-01-02", bbox="x")
        assert list(df.stid.unique()) == ["A", "B"]
        assert str(df.Time.dt.tz) == "UTC"
        assert len(df) == 4

    def test_numeric_strings_become_numbers_text_stays(self, api):
        t = ["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"]
        st = _station("A", t, [1.0, 2.0], [90, 90])
        st["OBSERVATIONS"]["air_temp_set_1"] = [1.5, "0.51"]  # seen from a real station
        st["OBSERVATIONS"]["wind_gust_set_1"] = ["", 3.0]  # '' = missing, also seen
        st["OBSERVATIONS"]["wind_cardinal_direction_set_1d"] = ["E", "E"]
        api.payload = {"SUMMARY": OK, "STATION": [st]}
        df = synoptic.timeseries("2024-01-01", "2024-01-02", stid="A")
        assert df.air_temp_set_1.tolist() == [1.5, 0.51]
        assert np.isnan(df.wind_gust_set_1[0]) and df.wind_gust_set_1[1] == 3.0
        assert df.wind_cardinal_direction_set_1d.tolist() == ["E", "E"]


class TestMetadata:
    def test_rows_and_period(self, api):
        api.payload = {
            "SUMMARY": OK,
            "STATION": [
                {
                    "STID": "WBB",
                    "NAME": "U of U",
                    "LATITUDE": "40.766",
                    "LONGITUDE": "-111.848",
                    "ELEVATION": "4806",
                    "MNET_ID": "153",
                    "STATUS": "ACTIVE",
                    "PERIOD_OF_RECORD": {
                        "start": "1997-01-01T00:00:00Z",
                        "end": "2026-09-30T00:00:00Z",
                    },
                }
            ],
        }
        meta = synoptic.metadata(stid="WBB")
        assert meta.loc["WBB", "latitude"] == pytest.approx(40.766)
        assert meta.loc["WBB", "record_start"] == pd.Timestamp("1997-01-01")


class TestHourlyMean:
    def frame(self):
        # hour 0: 350 and 10 deg at 5 m/s -> vector mean from the north
        # hour 1: dead calm; hour 2: nothing (dropped)
        t = pd.to_datetime(
            [
                "2024-01-01 00:00",
                "2024-01-01 00:30",
                "2024-01-01 01:00",
                "2024-01-01 01:30",
            ],
            utc=True,
        )
        return pd.DataFrame(
            {
                "stid": "A",
                "Time": t,
                "wind_speed_set_1": [5.0, 5.0, 0.0, 0.0],
                "wind_direction_set_1": [350.0, 10.0, 0.0, 0.0],
            }
        )

    def test_vector_mean_direction(self):
        h = synoptic.hourly_mean(self.frame())
        assert h.wind_direction_set_1.iloc[0] % 360 == pytest.approx(0.0, abs=1e-9)
        assert h.wind_speed_set_1.iloc[0] == pytest.approx(5.0)  # scalar mean kept
        assert h.n_obs.iloc[0] == 2

    def test_calm_hour_has_no_direction(self):
        h = synoptic.hourly_mean(self.frame())
        assert np.isnan(h.wind_direction_set_1.iloc[1])
        assert len(h) == 2

    def test_per_station(self):
        a = self.frame()
        b = self.frame().assign(stid="B", wind_direction_set_1=[90.0, 90.0, 90.0, 90.0])
        h = synoptic.hourly_mean(pd.concat([a, b]))
        assert list(h.columns[:2]) == ["stid", "Time"]
        assert h[h.stid == "B"].wind_direction_set_1.iloc[0] == pytest.approx(90.0)
