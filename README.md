
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
- **Weather Integration**: Displays weather data using WeatherAPI.

## Built With

- **Python 3**: Core programming language.
- **[iDotMatrix Library (maintained fork)](https://github.com/guilhermediasl/python3-idotmatrix-library)**: Bluetooth communication with the display over a single persistent connection.
- **PyQt5**: GUI for configuring and controlling the displays.

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip (Python package manager) and Git (the display library is installed from GitHub)
- Nightscout instance with API token

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

   On a headless Raspberry Pi, install `requirements-rpi.txt` instead. It excludes the PyQt GUI and other desktop-only packages.
4. Configure the project:

   - Copy the `config.json` template from `led_matrix_configurator` and update it with your Nightscout URL, token, and other settings.
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

The `config.json` file in the `led_matrix_configurator` directory contains the following settings:

- **Nightscout URL and Token**: For fetching glucose data.
- **Matrix Size**: Supported sizes are 16x16 and 32x32.
- **Brightness**: Adjusts display brightness for day and night modes.
- **Google Calendar**: Configure the four keys below. The app deletes its previous event before creating a new ten-minute event such as `🟡 62 ↗ -1`.

### Google Calendar Setup

1. In Google Cloud Console, create or select a project.
2. Enable **Google Calendar API**.
3. Configure the OAuth consent screen. For personal use, add your Google account as a test user if Google requests it.
4. Create OAuth credentials with application type **Desktop app**.
5. Create a calendar named `cgm` in Google Calendar.
6. Download the JSON file and save it in the project root with the name configured by `google_calendar_credentials` (normally `credentials.json`).
7. Fill the Calendar settings as follows:

```json
{
   "google_calendar_enabled": true,
   "google_calendar_id": "cgm",
   "google_calendar_credentials": "credentials.json",
   "google_calendar_token": "calendar_token.json"
}
```

- `google_calendar_enabled`: use `true` to synchronize events or `false` to disable synchronization.
- `google_calendar_id`: use the calendar name, such as `cgm`. The app finds the calendar by name. `primary` can still be used for the main calendar.
- `google_calendar_credentials`: path to the OAuth Desktop app JSON downloaded from Google Cloud. Do not commit this file.
- `google_calendar_token`: path where the app stores the authorization token. It is created automatically after the first login. Do not commit this file.

Run `python GlucosePixelMatrix.py` once with synchronization enabled. A browser window will open for authorization. The app then creates one event lasting ten minutes and replaces it on the next update.

# CGM LED Matrix

## Configuration File

The `config.json` file is used to configure the behavior of the LED matrix. Below is a description of each field:

- **high bondary glucose**: The upper glucose level threshold (e.g., 160).
- **low bondary glucose**: The lower glucose level threshold (e.g., 70).
- **image out**: The output device for the image. Use `"pc"` for testing on a computer or `"led_matrix"` for the LED matrix.
- **ip**: The Bluetooth MAC address of the LED matrix (e.g., `"E9:BA:C7:6F:AB:41"`).
- **night_brightness**: The brightness level for nighttime (e.g., `0.1` for 10% brightness).
- **os**: The operating system. Use `"windows"` for PC or `"linux"` for Raspberry Pi.
- **token**: The authentication token for the nightscout  API (e.g., `"api-password"`).
- **url**: The API endpoint for glucose data (e.g., `"https://glucose.fly.dev/api/v2"`).

### Example Configuration

```json
{
    "high bondary glucose": 160,
    "low bondary glucose": 70,
    "image out": "pc",
    "ip": "E9:BA:C7:6F:AB:41",
    "night_brightness": 0.1,
    "os": "windows",
    "token": "api-password",
    "url": "https://glucose.fly.dev/api/v2"
}
```

Ensure the `ip` field matches the Bluetooth MAC address of your LED matrix. To find it, scan for displays (their names start with `IDM-`):

```bash
python -c "import asyncio; from idotmatrix import ConnectionManager; print(asyncio.run(ConnectionManager.scan()))"
```

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
