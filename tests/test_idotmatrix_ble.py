"""Tests for the persistent BLE client, using a fake Bleak backend."""
import os
import struct
import sys
import types
import zlib

import pytest

# Provide a stub "bleak" when it is not installed so the tests run anywhere.
if "bleak" not in sys.modules:
    try:
        import bleak  # noqa: F401
    except ImportError:
        stub = types.ModuleType("bleak")
        stub.BleakClient = object
        stub.BleakScanner = object
        exc = types.ModuleType("bleak.exc")

        class BleakError(Exception):
            pass

        exc.BleakError = BleakError
        stub.exc = exc
        sys.modules["bleak"] = stub
        sys.modules["bleak.exc"] = exc

from bleak.exc import BleakError  # noqa: E402

from idotmatrix_ble import protocol  # noqa: E402
from idotmatrix_ble import client as client_mod  # noqa: E402

PNG = os.path.join(os.path.dirname(__file__), "..", "images", "nocgmdata.png")
IMAGE_MODE = bytes(protocol.image_mode(1))


# --------------------------------------------------------------- protocol
def test_image_payload_header():
    data = bytes(range(256)) * 20  # 5120 bytes -> 2 chunks
    payload = protocol.image_payload(data)
    header_size = 9
    assert len(payload) == len(data) + 2 * header_size
    assert payload[0:2] == struct.pack("h", len(data) + 2)
    assert payload[4] == 0
    assert payload[5:9] == struct.pack("i", len(data))
    second = header_size + protocol.PAYLOAD_CHUNK_SIZE
    assert payload[second + 4] == 2


def test_gif_payload_header():
    data = os.urandom(5000)
    payloads = protocol.gif_payloads(data)
    assert len(payloads) == 2
    first, second = payloads
    assert int.from_bytes(first[0:2], "little") == 16 + 4096
    assert first[4] == 0 and second[4] == 2
    assert int.from_bytes(first[5:9], "little") == len(data)
    assert int.from_bytes(first[9:13], "little") == zlib.crc32(data)


def test_brightness_range():
    assert protocol.brightness(50) == bytearray([5, 0, 4, 128, 50])
    with pytest.raises(ValueError):
        protocol.brightness(4)


# ----------------------------------------------------------------- client
class FakeChar:
    max_write_without_response_size = 100


class FakeServices:
    def get_characteristic(self, uuid):
        return FakeChar()


class FakeBle:
    """Shared state for the fake Bleak backend."""

    def __init__(self):
        self.connects = 0
        self.writes = []
        self.visible = True
        self.fail_next_write = False
        self.client = None


@pytest.fixture
def ble(monkeypatch):
    state = FakeBle()

    class FakeClient:
        def __init__(self, device, disconnected_callback=None, timeout=None):
            self.is_connected = False
            self.services = FakeServices()
            self.mtu_size = 103
            state.client = self

        async def connect(self):
            state.connects += 1
            self.is_connected = True

        async def disconnect(self):
            self.is_connected = False

        async def write_gatt_char(self, char, data, response=False):
            assert response is True
            if state.fail_next_write:
                state.fail_next_write = False
                self.is_connected = False
                raise BleakError("link lost")
            state.writes.append(bytes(data))

    class FakeScanner:
        @staticmethod
        async def find_device_by_address(address, timeout=0):
            return address if state.visible else None

    async def no_bluez(self):
        return None

    monkeypatch.setattr(client_mod, "BleakClient", FakeClient)
    monkeypatch.setattr(client_mod, "BleakScanner", FakeScanner)
    monkeypatch.setattr(client_mod.MatrixClient, "_bluez_disconnect", no_bluez)
    monkeypatch.setattr(client_mod.asyncio, "sleep", _instant_sleep)
    return state


async def _instant_sleep(_seconds):
    return None


@pytest.fixture
def matrix(ble):
    client = client_mod.MatrixClient("AA:BB:CC:DD:EE:FF", command_timeout=5)
    yield client
    client.close()


def test_reuses_one_connection(ble, matrix):
    for _ in range(5):
        matrix.set_image(PNG)
    matrix.set_brightness(40)
    assert ble.connects == 1


def test_image_mode_sent_once_per_connection(ble, matrix):
    for _ in range(3):
        matrix.set_image(PNG)
    assert ble.writes.count(IMAGE_MODE) == 1


def test_gif_and_reset_rearm_image_mode(ble, matrix, tmp_path):
    gif = tmp_path / "x.gif"
    gif.write_bytes(os.urandom(300))
    matrix.set_image(PNG)
    matrix.set_gif(str(gif))
    matrix.set_image(PNG)
    matrix.reset()
    matrix.set_image(PNG)
    assert ble.writes.count(IMAGE_MODE) == 3


def test_reconnects_and_restores_brightness_after_drop(ble, matrix):
    matrix.set_brightness(30)
    matrix.set_image(PNG)
    ble.client.is_connected = False  # link dropped
    ble.writes.clear()
    matrix.set_image(PNG)
    assert ble.connects == 2
    assert ble.writes[0] == bytes(protocol.brightness(30))
    assert ble.writes[1] == IMAGE_MODE


def test_retries_after_write_failure(ble, matrix):
    matrix.set_image(PNG)
    ble.fail_next_write = True
    matrix.set_image(PNG)
    assert ble.connects == 2


def test_device_not_found_fails_fast(ble, matrix):
    ble.visible = False
    with pytest.raises(client_mod.DeviceNotFoundError):
        matrix.set_image(PNG)
    assert ble.connects == 0


def test_writes_are_split_to_characteristic_size(ble, matrix):
    matrix.set_image(PNG)
    assert max(len(w) for w in ble.writes) <= FakeChar.max_write_without_response_size
