
# CGM LED Matrix

## Overview

The CGM LED Matrix project is designed to display Continuous Glucose Monitoring (CGM) data on iDotMatrix display. It integrates with Nightscout to fetch glucose data and provides a visual representation of glucose trends, treatments, and other relevant information. The project supports both 32x32 pixel displays and offers various customization options.

## Features

- **Real-time Glucose Data**: Fetches and displays glucose data from Nightscout.
- **Visual Indicators**: Displays glucose trends, treatments (e.g., bolus, carbs), and exercise data.
- **Night Mode**: Automatically adjusts brightness based on the time of day.
- **Bluetooth Integration**: Communicates with LED matrix display via Bluetooth.
- **Image and GIF Support**: Displays static images or animated GIFs on the matrix.
- **Weather Integration**: Displays weather data using WeatherAPI.

## Built With

- **Python 3**: Core programming language.
- **iDotMatrix Library**: For Bluetooth communication with LED displays.
- **PyQt5**: GUI for configuring and controlling the displays.

## Installation

### Prerequisites

- Python 3.7 or higher
- Pip (Python package manager)
- Nightscout instance with API token

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/cgm-led-matrix.git
   cd cgm-led-matrix
   ```
2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Configure the project:

   - Copy the `config.json` template from `led_matrix_configurator` and update it with your Nightscout URL, token, and other settings.
5. Run the application:

   ```bash
   python GlucosePixelMatrix.py
   ```

## Usage

### Continuous Glucose Monitoring

The `GlucosePixelMatrix.py` script fetches glucose data from Nightscout and updates the LED matrix display in real-time. Ensure your Nightscout instance is accessible and configured correctly in `config.json`.

## Configuration

The `config.json` file in the `led_matrix_configurator` directory contains the following settings:

- **Nightscout URL and Token**: For fetching glucose data.
- **Matrix Size**: Supported sizes are 16x16 and 32x32.
- **Brightness**: Adjusts display brightness for day and night modes.
- **Output Type**: Choose between image or GIF output.

# CGM LED Matrix

## Configuration File

The `config.json` file is used to configure the behavior of the LED matrix. Below is a description of each field:

- **high bondary glucose**: The upper glucose level threshold (e.g., 160).
- **low bondary glucose**: The lower glucose level threshold (e.g., 70).
- **image out**: The output device for the image. Use `"pc"` for testing on a computer or `"led_matrix"` for the LED matrix.
- **ip**: The Bluetooth MAC address of the LED matrix (e.g., `"E9:BA:C7:6F:AB:41"`).
- **night_brightness**: The brightness level for nighttime (e.g., `0.1` for 10% brightness).
- **os**: The operating system. Use `"windows"` for PC or `"linux"` for Raspberry Pi.
- **output type**: The type of output image. Use `"gif"` for animated images.
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
    "output type": "gif",
    "token": "api-password",
    "url": "https://glucose.fly.dev/api/v2"
}
```

Ensure the `ip` field matches the Bluetooth MAC address of your LED matrix, follow the documentation for the python3-idotmatrix-client documentation.

## Hardware Setup

This project is designed to run on a **Raspberry Pi Zero** and communicates with a **32x32 LED Matrix Display**. The specific LED matrix used can be found on [AliExpress](https://pt.aliexpress.com/item/1005006130862334.html?spm=a2g0o.order_list.order_list_main.203.193fcaa42y5ODu&gatewayAdapt=glo2bra).

### Requirements

- **Raspberry Pi Zero**: Used as the main controller.
- **32x32 LED Matrix Display**: The display communicates via Bluetooth.
- **Python3-idotmatrix-client**: A Python library for communicating with the LED matrix. The library's documentation can be found [here](https://github.com/derkalle4/python3-idotmatrix-client).

### Communication Setup

To set up communication with the LED matrix, follow the steps outlined in the [python3-idotmatrix-client documentation](https://github.com/derkalle4/python3-idotmatrix-client). Ensure that:

1. The Raspberry Pi Zero is configured with Bluetooth enabled.
2. The LED matrix is paired with the Raspberry Pi.

Refer to the library's documentation for detailed instructions on installation and usage.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](./idotmatrix/LICENSE) file for details.

## Acknowledgements

- [Nightscout](https://nightscout.info/) for providing the CGM data platform.
- [iDotMatrix Library](https://github.com/derkalle4/python3-idotmatrix-library) for LED matrix communication.
