import json
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import GlucosePixelMatrix as gpm
from GlucosePixelMatrix import GlucoseMatrixDisplay
from util import utc_to_local, entry_local_time


def local(year, month, day, hour, minute=0, second=0, micro=0):
    """The naive local time of a UTC instant, as the app should produce it."""
    return datetime(year, month, day, hour, minute, second, micro, tzinfo=timezone.utc).astimezone().replace(tzinfo=None)


class TestUtcToLocal:
    @pytest.mark.parametrize("value", [
        "2024-05-01T12:00:00.000Z",
        "2024-05-01T12:00:00Z",
        "2024-05-01T12:00:00",
        "2024-05-01T09:00:00-03:00",
        "2024-05-01T12:00:00.000+00:00",
        1714564800000,
    ])
    def test_converts_utc_to_machine_local_time(self, value):
        assert utc_to_local(value) == local(2024, 5, 1, 12)

    def test_accepts_any_number_of_fraction_digits(self):
        assert utc_to_local("2024-05-01T12:00:00.1234567Z") == local(2024, 5, 1, 12, micro=123456)
        assert utc_to_local("2024-05-01T12:00:00.5Z") == local(2024, 5, 1, 12, micro=500000)

    def test_entry_prefers_epoch_date(self):
        item = {"date": 1714564800000, "dateString": "2000-01-01T00:00:00.000Z"}
        assert entry_local_time(item) == local(2024, 5, 1, 12)

    def test_entry_without_any_date(self):
        assert entry_local_time({"sgv": 100}) is None


@pytest.fixture
def make_display(tmp_path):
    def make(**config):
        config.setdefault("token", "s3cret-token")
        config.setdefault("url", "https://ns.example.com/api/v2/")
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        return GlucoseMatrixDisplay(str(path))
    return make


class TestConfigValues:
    @pytest.mark.parametrize("value,expected", [
        (0.1, 10), (0.5, 50), (10, 10), (30, 30), (100, 100),
        (150, 100), (0, 5), (2, 5), (None, 10), ("bad", 10),
    ])
    def test_night_brightness_is_a_valid_percent(self, value, expected):
        assert GlucoseMatrixDisplay.normalize_brightness(value) == expected

    def test_stale_threshold_defaults_to_15_minutes(self, make_display):
        assert make_display().max_time == 15 * 60 * 1000

    def test_stale_threshold_from_config(self, make_display):
        assert make_display(stale_minutes=30).max_time == 30 * 60 * 1000

    def test_legacy_bondary_keys_are_read(self, make_display):
        display = make_display(**{"low bondary glucose": 80, "high bondary glucose": 170})
        assert (display.GLUCOSE_LOW, display.GLUCOSE_HIGH) == (80, 170)

    def test_boundaries_default_when_missing(self, make_display):
        display = make_display()
        assert (display.GLUCOSE_LOW, display.GLUCOSE_HIGH) == (70, 160)

    def test_trailing_slash_in_url_is_ignored(self, make_display):
        assert make_display().url_entries.startswith("https://ns.example.com/api/v2/entries.json?")


class TestTreatmentTimes:
    def test_created_at_is_converted_for_every_uploader(self, make_display):
        display = make_display()
        display.json_treatments_data = [
            {"_id": "a", "eventType": "Bolus", "insulin": 2, "created_at": "2024-05-01T12:00:00.000Z",
             "enteredBy": "xDrip4iOS"},
            {"_id": "b", "eventType": "Carbs", "carbs": 20, "created_at": "2024-05-01T12:00:00Z",
             "utcOffset": -180},
            {"_id": "c", "eventType": "Carbs", "carbs": 20},  # no created_at: skipped
        ]
        display.generate_list_from_treatments_json()
        assert [t.date for t in display.formatted_treatments] == [local(2024, 5, 1, 12)] * 2


class TestTokenRedaction:
    def test_token_is_removed_from_log_lines(self, make_display):
        make_display(token="abc123secret")
        record = logging.LogRecord("x", logging.ERROR, __file__, 1,
                                   "Connection error: %s", ("https://ns/entries.json?token=abc123secret",), None)
        gpm.secret_filter.filter(record)
        assert "abc123secret" not in record.getMessage()

    def test_connection_error_message_has_no_token(self, make_display, monkeypatch):
        import requests
        display = make_display(token="abc123secret")
        display.http = MagicMock()
        display.http.get.side_effect = requests.exceptions.ConnectionError("down")
        monkeypatch.setattr(gpm.time, "sleep", lambda _: None)
        with pytest.raises(gpm.NightscoutConnectionError) as error:
            display.fetch_json_data(display.url_entries, retries=1)
        assert "abc123secret" not in str(error.value)


class _StopLoop(BaseException):
    pass


def test_empty_ping_shows_stale_data_instead_of_crashing(make_display, monkeypatch):
    display = make_display()
    display.fetch_json_data = MagicMock(return_value=[])
    display.update_glucose_command = MagicMock()
    display.run_command = MagicMock()
    display.sync_google_calendar = MagicMock()
    display.run_set_brightness_command = MagicMock()

    def stop(_):
        raise _StopLoop
    monkeypatch.setattr(gpm.time, "sleep", stop)

    with pytest.raises(_StopLoop):
        display.run_command_in_loop()
    display.update_glucose_command.assert_called_once_with(stale_data=True)


class TestConfigurator:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from configurator import configurator, config_loader
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"token": "saved-token", "url": "https://ns.example.com"}))
        monkeypatch.setattr(configurator, "CONFIG_PATH", str(config_path))
        monkeypatch.setattr(config_loader, "CONFIG_PATH", str(config_path))
        monkeypatch.setattr(configurator, "LOG_PATH", str(tmp_path / "app.log"))
        return configurator.app.test_client(), config_path

    def test_get_masks_token(self, client):
        test_client, _ = client
        config = test_client.get("/config").get_json()["config"]
        assert config["token"] == "********"

    def test_save_keeps_stored_token_and_accepts_calendar_keys(self, client):
        test_client, config_path = client
        response = test_client.post("/save", json={
            "token": "********",
            "url": "https://ns.example.com/api/v2",
            "night_brightness": 10,
            "google_calendar_enabled": True,
            "google_calendar_id": "cgm",
        })
        assert response.status_code == 200, response.get_json()
        saved = json.loads(config_path.read_text())
        assert saved["token"] == "saved-token"
        assert saved["google_calendar_id"] == "cgm"

    def test_numeric_token_stays_a_string(self, client):
        test_client, config_path = client
        response = test_client.post("/save", json={"token": "12345", "url": "https://ns.example.com"})
        assert response.status_code == 200, response.get_json()
        assert json.loads(config_path.read_text())["token"] == "12345"

    def test_invalid_config_returns_errors(self, client):
        test_client, _ = client
        response = test_client.post("/save", json={"token": "t", "url": "not-a-url"})
        assert response.status_code == 400
        assert response.get_json()["errors"][0]["path"] == "url"

    def test_logs_hide_saved_token(self, client, tmp_path):
        test_client, _ = client
        (tmp_path / "app.log").write_text("GET https://ns/entries.json?token=saved-token\n")
        assert "saved-token" not in test_client.get("/logs").get_json()["logs"]


def test_fade_color_above_one_is_clamped_to_255():
    from PixelMatrix import PixelMatrix
    from util import ColorType
    matrix = PixelMatrix(32, 60, 180, 70, 160, 10, 5)
    assert matrix.fade_color(ColorType(255, 200, 10), 1.4) == ColorType(255, 255, 14)


class TestConfiguratorPage:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from configurator import configurator
        monkeypatch.setattr(configurator, "PREVIEW_PATH", str(tmp_path / "output_image.png"))
        monkeypatch.setattr(configurator, "CONFIG_PATH", str(tmp_path / "config.json"))
        return configurator.app.test_client(), tmp_path

    def test_schema_keeps_field_order_and_documents_every_visible_field(self, client):
        test_client, _ = client
        schema = test_client.get("/schema").get_json()
        keys = list(schema["properties"])
        assert keys.index("low boundary glucose") < keys.index("high boundary glucose")
        group_ids = {g["id"] for g in schema["x-groups"]}
        for key, spec in schema["properties"].items():
            if spec.get("x-hidden"):
                continue
            assert spec.get("x-group") in group_ids, key
            assert spec.get("title") and spec.get("description"), key

    def test_schema_matches_config_model(self):
        from configurator.config_model import ConfigModel
        schema = json.load(open("configurator/config.schema.json", encoding="utf-8"))
        model_keys = {f.alias or name for name, f in ConfigModel.model_fields.items()}
        assert set(schema["properties"]) == model_keys

    def test_preview_missing_then_served(self, client):
        test_client, tmp_path = client
        assert test_client.get("/preview").status_code == 404
        (tmp_path / "output_image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        response = test_client.get("/preview")
        assert response.status_code == 200
        assert response.mimetype == "image/png"
        assert response.headers["Cache-Control"] == "no-store"

    def test_status_reports_times(self, client):
        test_client, tmp_path = client
        (tmp_path / "config.json").write_text("{}")
        status = test_client.get("/status").get_json()
        assert status["service"]
        assert status["config_updated"] and status["preview_updated"] is None

    def test_page_renders_without_external_scripts(self, client):
        test_client, _ = client
        html = test_client.get("/").get_data(as_text=True)
        assert "<script src=" not in html


class TestNightBrightness:
    def _loop_once(self, display, monkeypatch):
        display.fetch_json_data = MagicMock(return_value=[])
        display.update_glucose_command = MagicMock()
        display.sync_google_calendar = MagicMock()

        def stop(_):
            raise _StopLoop
        monkeypatch.setattr(gpm.time, "sleep", stop)
        with pytest.raises(_StopLoop):
            display.run_command_in_loop()

    def test_failed_brightness_is_retried(self, make_display, monkeypatch):
        display = make_display()
        monkeypatch.setattr(gpm, "get_nightmode", lambda: True)
        display.run_command = MagicMock(return_value=False)
        self._loop_once(display, monkeypatch)
        assert display.last_night_state is None  # not recorded, so sent again

        display.run_command = MagicMock(return_value=True)
        self._loop_once(display, monkeypatch)
        assert display.last_night_state is True

    def test_reset_reapplies_night_brightness(self, make_display, monkeypatch):
        display = make_display()
        monkeypatch.setattr(gpm, "get_nightmode", lambda: True)
        display.last_night_state = True
        display.run_command_count = display.RUN_COMMAND_MAX_COUNT + 1
        sent = []

        def record():
            sent.append(display.display_action)
            return True
        display.run_command = MagicMock(side_effect=record)
        self._loop_once(display, monkeypatch)
        assert sent[0] == ("reset", None)
        assert sent[1] == ("brightness", display.night_brightness)
