import pytest
import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from GlucosePixelMatrix import GlucoseMatrixDisplay, get_nightmode
from util import GlucoseItem, TreatmentItem, IobItem, EntrieEnum, TreatmentEnum


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_data = {
        "ip": "192.168.1.100",
        "token": "test_token",
        "url": "https://test.nightscout.com",
        "glucose_low": 70,
        "glucose_high": 180,
        "os": "linux",
        "image_output_path": "test_output.png",
        "output_type": "image",
        "night_brightness": 30,
        "plot_glucose_intervals": True
    }

    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config_data))
    return str(config_file)


class TestGlucoseMatrixDisplayInit:
    def test_initialization_with_config(self, temp_config_file):
        display = GlucoseMatrixDisplay(temp_config_file, matrix_size=16)
        assert display.matrix_size == 16
        assert display.min_glucose == 60
        assert display.max_glucose == 180
        assert display.ip == "192.168.1.100"

    def test_initialization_missing_config(self):
        with pytest.raises(Exception, match="Error loading configuration file"):
            GlucoseMatrixDisplay("nonexistent.json")

    def test_load_config_invalid_json(self, tmp_path):
        bad_config = tmp_path / "bad_config.json"
        bad_config.write_text("{invalid json")

        with pytest.raises(Exception, match="Error loading configuration file"):
            GlucoseMatrixDisplay(str(bad_config))

    def test_initialization_default_parameters(self, temp_config_file):
        display = GlucoseMatrixDisplay(temp_config_file)
        assert display.matrix_size == 32  # default value
        assert display.min_glucose == 60  # default value
        assert display.max_glucose == 180  # default value

    def test_initialization_custom_parameters(self, temp_config_file):
        display = GlucoseMatrixDisplay(temp_config_file, matrix_size=64, min_glucose=50, max_glucose=200)
        assert display.matrix_size == 64
        assert display.min_glucose == 50
        assert display.max_glucose == 200


class TestConfigurationHandling:
    def test_load_config_values(self, temp_config_file):
        display = GlucoseMatrixDisplay(temp_config_file)

        assert "test_token" in display.url_entries
        assert "test_token" in display.url_treatments
        assert display.os == "linux"

    def test_setup_paths(self, temp_config_file):
        display = GlucoseMatrixDisplay(temp_config_file)

        assert display.NO_DATA_IMAGE_PATH.endswith("nocgmdata.png")
        assert display.OUTPUT_IMAGE_PATH.endswith("output_image.png")

    def test_load_config_missing_keys(self, tmp_path):
        incomplete_config = {"ip": "192.168.1.100"}
        config_file = tmp_path / "incomplete_config.json"
        config_file.write_text(json.dumps(incomplete_config))

        display = GlucoseMatrixDisplay(str(config_file))
        assert display.ip == "192.168.1.100"


class TestDataParsing:
    @pytest.fixture
    def display(self, temp_config_file):
        return GlucoseMatrixDisplay(temp_config_file, matrix_size=16)

    @pytest.fixture
    def sample_entries_json(self):
        now = datetime.now().replace(microsecond=123456) + timedelta(hours=3)
        return [
            {
                "type": "sgv",
                "sgv": 120,
                "dateString": now.isoformat(timespec="microseconds") + "Z",
                "direction": "Flat"
            },
            {
                "type": "sgv",
                "sgv": 115,
                "dateString": (now - timedelta(minutes=5)).isoformat(timespec="microseconds") + "Z",
                "direction": "Flat"
            }
        ]

    @pytest.fixture
    def sample_treatments_json(self):
        now = datetime.now().replace(microsecond=123456) + timedelta(hours=3)
        return [
            {
                "_id": "test1",
                "eventType": "Bolus",
                "insulin": 5.0,
                "created_at": now.isoformat(timespec="microseconds") + "Z",
                "enteredBy": "test",
                "utcOffset": 0
            },
            {
                "_id": "test2",
                "eventType": "Carbs",
                "carbs": 30,
                "created_at": (now - timedelta(minutes=15)).isoformat(timespec="microseconds") + "Z",
                "enteredBy": "test",
                "utcOffset": 0
            }
        ]

    def test_generate_list_from_entries_json(self, display, sample_entries_json):
        display.json_entries_data = sample_entries_json
        display.generate_list_from_entries_json()

        assert len(display.formatted_entries) == 2
        assert display.formatted_entries[0].glucose == 120

    def test_generate_list_from_treatments_json(self, display, sample_treatments_json):
        display.json_treatments_data = sample_treatments_json
        display.generate_list_from_treatments_json()

        assert len(display.formatted_treatments) == 2

    def test_extract_first_and_second_value(self, display):
        now = datetime.now()
        display.formatted_entries = [
            GlucoseItem(EntrieEnum.SGV, 120, now),
            GlucoseItem(EntrieEnum.SGV, 115, now - timedelta(minutes=5)),
            GlucoseItem(EntrieEnum.MBG, 110, now - timedelta(minutes=10))
        ]

        first, second = display.extract_first_and_second_value()
        assert first.glucose == 120
        assert second.glucose == 115


class TestIOBHandling:
    @pytest.fixture
    def display(self, temp_config_file):
        return GlucoseMatrixDisplay(temp_config_file, matrix_size=32)

    def test_insert_iob_item_from_json(self, display):
        display.json_iob = {"iob": {"iob": 2.5}}
        display.insert_iob_item_from_json()
        assert len(display.iob_list) == 1

    def test_delete_outdated_iob_items(self, display):
        now = datetime.now()
        display.pixels_time = [now - timedelta(minutes=i * 5) for i in range(32)]
        display.iob_list = [
            IobItem(now, 2.5),
            IobItem(now - timedelta(minutes=10), 2.0),
            IobItem(now - timedelta(minutes=180), 1.5),
        ]
        display.delete_outdated_iob_items()
        assert all(i.date >= now - timedelta(minutes=40) for i in display.iob_list)


class TestUtilityFunctions:
    def test_get_nightmode(self):
        """Just ensure it runs with current system time."""
        result = get_nightmode()
        assert isinstance(result, bool)
