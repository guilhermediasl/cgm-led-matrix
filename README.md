
# CGM LED Matrix

## Overview

The CGM LED Matrix project is designed to display Continuous Glucose Monitoring (CGM) data on iDotMatrix display. It integrates with Nightscout to fetch glucose data and provides a visual representation of glucose trends, treatments, and other relevant information. The project supports both 32x32 pixel displays and offers various customization options.
![5091691091438186308](https://github.com/user-attachments/assets/f9257e66-7c6f-40aa-b780-1d6ab60271ba)

![demo](https://github.com/user-attachments/assets/1c2f7137-2a3d-4bb9-9802-d1d117d3810d)


## Features

- **Real-time Glucose Data**: Fetches and displays glucose data from Nightscout.
- **Visual Indicators**: Displays glucose trends, treatments (e.g., bolus, carbs), and exercise data.
- **Night Mode**: Automatically adjusts brightness based on the time of day.
- **Bluetooth Integration**: Communicates with LED matrix display via Bluetooth.
- **Reading Age Counter**: The top-left column lights one pink pixel per minute since the last reading (all five after 5 minutes), so a delayed CGM is visible at a glance.
- **Always-current Clock**: The image is redrawn whenever the clock minute or the reading age changes, even without new data.
- **Web Configurator**: A small Flask page (`configurator/configurator.py`) to edit the config, read logs and restart the service.

## Built With

- **Python 3**: Core programming language.
- **[iDotMatrix Library (maintained fork)](https://github.com/guilhermediasl/python3-idotmatrix-library)**: Bluetooth communication with the display over a single persistent connection.
- **Flask + Pydantic**: Web configurator that validates and saves `config.json`.

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip (Python package manager) and Git (the display library is installed from GitHub)
- Nightscout instance with API token
- The machine's time zone set to your local time zone (Nightscout timestamps are UTC and are converted to local time; on a Pi use `sudo raspi-config` or `timedatectl set-timezone`)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/guilhermediasl/cgm-led-matrix.git
   cd cgm-led-matrix
   ```
2. Create a virtual environment named `.venv` and activate it:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   On a headless Raspberry Pi, install `requirements-rpi.txt` instead. It has only what the display and the web configurator need. Test dependencies are in `tests/test_requirements.txt`.
4. Configure the project:

   - Copy `configurator/config.example.json` to `configurator/config.json` and fill in your Nightscout URL, token, and other settings (the web configurator creates it for you on first run).
5. Run the application:

   ```bash
   python GlucosePixelMatrix.py
   ```

   For automatic startup on Raspberry Pi, run `run_glucose.sh` from a systemd service. It uses the project `.venv` explicitly, so all dependencies are available after reboot. Example `/etc/systemd/system/glucose_matrix.service`:

   ```ini
   [Unit]
   Description=Start Glucose Matrix Display
   After=network-online.target bluetooth.target
   Wants=network-online.target

   [Service]
   ExecStart=/home/pi/cgm-led-matrix/run_glucose.sh
   WorkingDirectory=/home/pi/cgm-led-matrix
   User=pi
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```

   Enable it with `sudo systemctl daemon-reload && sudo systemctl enable --now glucose_matrix.service`. On stop, the app releases the Bluetooth link cleanly.

## Usage

### Continuous Glucose Monitoring

The `GlucosePixelMatrix.py` script fetches glucose data from Nightscout and updates the LED matrix display in real-time. Ensure your Nightscout instance is accessible and configured correctly in `config.json`.

## Configuration

All settings live in `configurator/config.json`. The display app reads it **once at startup**, so restart the service after every change. You can edit the file by hand or use the web configurator.

### Web configurator

A small web page to edit the settings, see what the display is showing and follow the logs, from any phone or computer on your network.

```bash
python configurator/configurator.py
```

Then open `http://<pi-address>:5000` (for example `http://raspberrypi.local:5000`). The page offers:

- **Settings grouped by topic** (Nightscout, Display, Glucose range, Clock, Night mode & alerts, Google Calendar), with an explanation under every field, inline validation and a bar that tracks unsaved changes (<kbd>Ctrl</kbd>+<kbd>S</kbd> saves).
- **Live preview** of the last frame sent to the matrix (`temp/output_image.png`), refreshed every 20 seconds, with a legend explaining each element.
- **Display service controls**: restart the service to apply new settings, pull the latest code, or update and restart in one step.
- **Logs** of the display app, with a warnings & errors filter. The Nightscout token is never sent to the page or shown in the logs.

Restarting uses `sudo systemctl restart glucose_matrix.service`, so the user running the configurator needs passwordless sudo for that command (the default `pi` user has it). On Windows the restart button is disabled; restart `GlucosePixelMatrix.py` yourself.

> **Security:** the configurator has no login and listens on all interfaces. Run it only on a trusted home network and do not expose port 5000 to the internet.

To start it on boot, add a second unit, for example `/etc/systemd/system/glucose_configurator.service`:

```ini
[Unit]
Description=CGM LED Matrix configurator
After=network-online.target

[Service]
ExecStart=/home/pi/cgm-led-matrix/.venv/bin/python /home/pi/cgm-led-matrix/configurator/configurator.py
WorkingDirectory=/home/pi/cgm-led-matrix
User=pi
Restart=always

[Install]
WantedBy=multi-user.target
```

To check a config file without the page, run `python configurator/validate_config.py [path/to/config.json]`.

### Settings reference

`configurator/config.schema.json` is the source of truth: it holds the type, limits and description of every key, and the web configurator is built from it.

| Key | Default | What it does |
| --- | --- | --- |
| `url` | (required) | Nightscout site followed by `/api/v2`, e.g. `https://my-site.herokuapp.com/api/v2`. |
| `token` | (required) | Nightscout access token with read access (Admin Tools → Subjects), e.g. `readable-1a2b3c4d5e6f`. |
| `image out` | `"led matrix"` | `"led matrix"` sends frames over Bluetooth; `"pc"` only writes `temp/output_image.png`. |
| `ip` | `""` | Bluetooth MAC address of the panel, e.g. `E9:BA:C7:6F:AB:41`. See below for how to find it. |
| `os` | `"linux"` | `"linux"` makes the app run `rfkill unblock bluetooth` at startup. |
| `plot glucose intervals` | `true` | Draw faint lines between consecutive readings. |
| `low boundary glucose` | `70` | Low limit in mg/dL: yellow at or below it, red below it minus 10. |
| `high boundary glucose` | `160` | High limit in mg/dL: yellow at or above it, red above it plus 10. |
| `show_time` | `false` | Draw the clock. It is always drawn while the data is stale. |
| `time_format` | `"24h"` | `24h`, `12h`, `compact` (1405), `HH:MM`, `HH:MM:SS` or `MM:SS`. |
| `time_position` | `"top-right"` | `top-left`, `top-right`, `bottom-left`, `bottom-right` or `bottom`. |
| `night_brightness` | `10` | Brightness in percent (5–100) from 22:00 to 06:59; daytime is 100%. A fraction such as `0.1` is read as 10%. |
| `stale_minutes` | `15` | Minutes without a new reading before the stale mark appears. |
| `google_calendar_enabled` | `false` | Keep one ten-minute event with the current reading in Google Calendar. |
| `google_calendar_id` | `"primary"` | Calendar name (e.g. `cgm`) or `primary`. |
| `google_calendar_credentials` | `"credentials.json"` | OAuth Desktop-app client JSON. Do not commit it. |
| `google_calendar_token` | `"calendar_token.json"` | Where the authorization is stored. Created on first login. Do not commit it. |

`PIXEL_INTERVAL`, `time_color_fade` and `output type` are accepted for older configs but not used.

### Example configuration

```json
{
    "url": "https://my-site.herokuapp.com/api/v2",
    "token": "readable-1a2b3c4d5e6f",
    "image out": "led matrix",
    "ip": "E9:BA:C7:6F:AB:41",
    "os": "linux",
    "low boundary glucose": 70,
    "high boundary glucose": 160,
    "show_time": true,
    "time_format": "24h",
    "time_position": "bottom-left",
    "night_brightness": 10,
    "stale_minutes": 15
}
```

### Finding the display address

Scan for displays (their names start with `IDM-`) and copy the MAC address into `ip`:

```bash
python -c "import asyncio; from idotmatrix import ConnectionManager; print(asyncio.run(ConnectionManager.scan()))"
```

### Google Calendar setup

The app keeps a single ten-minute event such as `🟡 62 ↗ -1`, deleting the previous one before creating the next.

1. In Google Cloud Console, create or select a project.
2. Enable **Google Calendar API**.
3. Configure the OAuth consent screen. For personal use, add your Google account as a test user if Google requests it.
4. Create OAuth credentials with application type **Desktop app**.
5. Create a dedicated calendar named `cgm` in Google Calendar (its events are replaced on every reading).
6. Download the JSON file and save it in the project root as `credentials.json` (or the name set in `google_calendar_credentials`).
7. Turn on **Sync the reading to Google Calendar** in the configurator (or set `google_calendar_enabled` to `true`) and set the calendar to `cgm`.

Run `python GlucosePixelMatrix.py` once with synchronization enabled. A browser window opens for authorization and the token file is saved. On a headless Pi, authorize once on a computer and copy the token file to the Pi.

## Hardware Setup

This project is designed to run on a **Raspberry Pi Zero** and communicates with a **32x32 LED Matrix Display**. The specific LED matrix used can be found on [AliExpress](https://pt.aliexpress.com/item/1005006130862334.html?spm=a2g0o.order_list.order_list_main.203.193fcaa42y5ODu&gatewayAdapt=glo2bra).

### Requirements

- **Raspberry Pi Zero**: Used as the main controller.
- **32x32 LED Matrix Display**: The display communicates via Bluetooth.
- **[python3-idotmatrix-library (fork)](https://github.com/guilhermediasl/python3-idotmatrix-library)**: installed by `requirements*.txt`, pinned to a release tag.

### Communication Setup

The app keeps one Bluetooth Low Energy connection open to the display and reuses it for every update, so the display does not flash its Bluetooth icon or a black frame between images. It reconnects automatically if the link drops or the display is power-cycled. Ensure that:

1. Bluetooth is enabled on the Raspberry Pi (the app runs `rfkill unblock bluetooth` on Linux at startup).
2. The display is in range and not connected to the phone app. Pairing is not required.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](./LICENSE) file for details.

## Acknowledgements

- [Nightscout](https://nightscout.info/) for providing the CGM data platform.
- [python3-idotmatrix-library](https://github.com/derkalle4/python3-idotmatrix-library) by Kalle Minkner, Jon-Mailes Graeffe and contributors, for reverse-engineering the iDotMatrix protocol. This project uses a [maintained fork](https://github.com/guilhermediasl/python3-idotmatrix-library) with a persistent connection.
