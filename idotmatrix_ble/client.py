"""Persistent BLE connection to an iDotMatrix display.

The upstream library connects inside a short-lived CLI process, so every
update opens a new Bluetooth connection and the display flashes its Bluetooth
icon before drawing. This client keeps one BleakClient open on a background
asyncio loop and reuses it for every command, reconnecting only when the link
actually drops.
"""
import asyncio
import concurrent.futures
import logging
import sys
import threading
from typing import Iterable, Optional

from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

from . import protocol

log = logging.getLogger(__name__)


class DeviceNotFoundError(ConnectionError):
    """The display is not advertising (powered off or out of range)."""


class MatrixClient:
    """Thread-safe synchronous facade over a persistent BLE connection."""

    def __init__(
        self,
        address: str,
        connect_timeout: float = 20.0,
        scan_timeout: float = 10.0,
        command_timeout: float = 120.0,
        retries: int = 3,
    ) -> None:
        self.address = address
        self.connect_timeout = connect_timeout
        self.scan_timeout = scan_timeout
        self.command_timeout = command_timeout
        self.retries = retries
        self._client: Optional[BleakClient] = None
        self._lock = asyncio.Lock()
        self._brightness: Optional[int] = None
        # The "enter image mode" command clears the screen (black flash), so it is
        # only sent once per connection instead of before every image.
        self._image_mode_active = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="idotmatrix-ble", daemon=True)
        self._thread.start()

    def _call(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(self.command_timeout)
        except concurrent.futures.TimeoutError:
            # Cancel the coroutine too, otherwise it keeps holding the send lock.
            future.cancel()
            raise TimeoutError(f"iDotMatrix command timed out after {self.command_timeout}s") from None

    # ---------------------------------------------------------- connection
    @property
    def is_connected(self) -> bool:
        return bool(self._client and self._client.is_connected)

    def _on_disconnect(self, client: BleakClient) -> None:
        if client is self._client:  # not an intentional disconnect
            log.warning("iDotMatrix %s disconnected", self.address)

    async def _find_device(self):
        device = await BleakScanner.find_device_by_address(self.address, timeout=self.scan_timeout)
        if device is None and sys.platform.startswith("linux"):
            # A connected device stops advertising. If a previous process died while
            # connected, BlueZ still holds that link, so release it and scan again.
            log.info("Device not advertising; releasing any stale BlueZ connection")
            await self._bluez_disconnect()
            device = await BleakScanner.find_device_by_address(self.address, timeout=self.scan_timeout)
        if device is None:
            raise DeviceNotFoundError(f"iDotMatrix {self.address} was not found (off or out of range?)")
        return device

    async def _ensure_connected(self) -> BleakClient:
        if self._client is not None and self._client.is_connected:
            return self._client
        await self._drop_client()
        log.info("Connecting to iDotMatrix %s", self.address)
        client = BleakClient(
            await self._find_device(),
            disconnected_callback=self._on_disconnect,
            timeout=self.connect_timeout,
        )
        await client.connect()
        self._client = client
        self._image_mode_active = False
        await self._acquire_mtu(client)
        log.info("Connected to iDotMatrix %s (MTU %s)", self.address, client.mtu_size)
        if self._brightness is not None:
            # The display may have power-cycled while we were away.
            await self._write(client, protocol.brightness(self._brightness))
        return client

    @staticmethod
    async def _acquire_mtu(client: BleakClient) -> None:
        """On BlueZ, Bleak assumes the minimum MTU (20-byte writes) until asked for
        the negotiated one, which makes every upload many times slower."""
        acquire = getattr(getattr(client, "_backend", None), "_acquire_mtu", None)
        if acquire is None:
            return
        try:
            await acquire()
        except Exception as error:
            log.debug("Could not read negotiated MTU, using default: %s", error)

    async def _bluez_disconnect(self) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "bluetoothctl", "disconnect", self.address,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), 10)
            await asyncio.sleep(2)
        except (OSError, asyncio.TimeoutError) as error:
            log.warning("bluetoothctl disconnect failed: %s", error)

    async def _drop_client(self) -> None:
        client, self._client = self._client, None
        self._image_mode_active = False
        if client is not None:
            try:
                await client.disconnect()
            except Exception as error:  # already gone
                log.debug("Ignoring disconnect error: %s", error)

    @staticmethod
    async def _write(client: BleakClient, data: bytes) -> None:
        """Write with response so a dead link raises instead of silently dropping data."""
        characteristic = client.services.get_characteristic(protocol.UUID_WRITE_DATA)
        if characteristic is None:
            raise BleakError("iDotMatrix write characteristic not found")
        size = characteristic.max_write_without_response_size
        for i in range(0, len(data), size):
            await client.write_gatt_char(characteristic, data[i:i + size], response=True)

    async def _send(self, packets: Iterable[bytes], image_mode: Optional[bool] = None) -> None:
        """Send packets, reconnecting and retrying on link errors.

        image_mode: True to make sure DIY image mode is on first, False to mark it
        as left (GIF/reset), None to leave it untouched.
        """
        packets = list(packets)
        async with self._lock:
            last_error: Optional[Exception] = None
            for attempt in range(1, self.retries + 1):
                try:
                    client = await self._ensure_connected()
                    if image_mode and not self._image_mode_active:
                        await self._write(client, protocol.image_mode(1))
                        self._image_mode_active = True
                    elif image_mode is False:
                        self._image_mode_active = False
                    for packet in packets:
                        await self._write(client, packet)
                    return
                except DeviceNotFoundError:
                    raise  # retrying right away will not make it appear
                except (BleakError, OSError, asyncio.TimeoutError) as error:
                    last_error = error
                    log.warning("BLE send failed (attempt %d/%d): %s", attempt, self.retries, error)
                    await self._drop_client()
                    await asyncio.sleep(2)
            raise ConnectionError(f"Could not send to iDotMatrix {self.address}") from last_error

    async def _set_brightness(self, percent: int) -> None:
        packet = protocol.brightness(percent)
        self._brightness = percent  # also re-applied after reconnects
        await self._send([packet])

    async def _close(self) -> None:
        async with self._lock:
            await self._drop_client()

    # ---------------------------------------------------------- public API
    def connect(self) -> None:
        async def _connect():
            async with self._lock:
                await self._ensure_connected()
        self._call(_connect())

    def set_image(self, file_path: str, pixel_size: Optional[int] = None) -> None:
        payload = protocol.image_payload(protocol.png_bytes(file_path, pixel_size))
        self._call(self._send([payload], image_mode=True))

    def set_gif(self, file_path: str) -> None:
        with open(file_path, "rb") as file:
            payloads = protocol.gif_payloads(file.read())
        self._call(self._send(payloads, image_mode=False))

    def set_brightness(self, percent: int) -> None:
        self._call(self._set_brightness(percent))

    def reset(self) -> None:
        self._call(self._send(protocol.reset(), image_mode=False))

    def close(self) -> None:
        try:
            self._call(self._close())
        except Exception as error:
            log.debug("Ignoring close error: %s", error)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
