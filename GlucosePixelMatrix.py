import math
import os
import subprocess
import requests
import time
import json
import datetime
import logging
from logging.handlers import RotatingFileHandler
from typing import List
from http.client import RemoteDisconnected
from util import GlucoseItem, IobItem, TreatmentItem, ExerciseItem, TreatmentEnum, EntrieEnum
from PixelMatrix import PixelMatrix
import bisect

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = 'app.log'
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class GlucoseMatrixDisplay:
    """Main application class for continuous glucose monitoring on LED matrix."""
    def __init__(self, config_path=os.path.join('led_matrix_configurator', 'config.json'), matrix_size=32, min_glucose=60, max_glucose=180):
        """Initialize the glucose matrix display system.
        
        Args:
            config_path: Path to configuration JSON file
            matrix_size: Size of LED matrix (default 32x32)
            min_glucose: Minimum glucose value for scaling (mg/dL)
            max_glucose: Maximum glucose value for scaling (mg/dL)
        """
        self.matrix_size = matrix_size
        self.min_glucose = min_glucose
        self.max_glucose = max_glucose
        self.PIXEL_INTERVAL = 5
        self.RUN_COMMAND_MAX_COUNT = 800
        self.max_time = self.PIXEL_INTERVAL * 60 * 1000 * self.matrix_size #milliseconds
        self.pixels_time = [datetime.datetime.now() - datetime.timedelta(minutes=i * self.PIXEL_INTERVAL) for i in range(self.matrix_size)]
        self.config = self.load_config(config_path)
        self.arrow = ''
        self.first_glucose_entry = GlucoseItem(EntrieEnum.SGV, 0, datetime.datetime.now())
        self.second_glucose_entry = GlucoseItem(EntrieEnum.SGV, 0, datetime.datetime.now())
        self.formmated_entries: List[GlucoseItem] = []
        self.formmated_treatments: List[TreatmentItem | ExerciseItem] = []
        self.iob_list: List[IobItem] = []
        self.newer_id = None
        self.command = ''
        self.last_nightstate = None
        self.run_command_count = 0
        self._load_config_values()
        self._setup_paths()
        if self.image_out == "led matrix" and self.os != "windows": self.unblock_bluetooth()

    def load_config(self, config_path) -> dict:
        """Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            dict: Configuration dictionary
            
        Raises:
            Exception: If file cannot be loaded or parsed
        """
        try:
            logging.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as file:
                config = json.load(file)
                logging.info(f"Configuration loaded successfully: {config}")
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error loading configuration file: {e}")
            raise Exception(f"Error loading configuration file: {e}")
        
    def _load_config_values(self):
        """Extract and set configuration values from loaded config."""
        self.ip = self.config.get('ip')
        token = self.config.get('token')
        self.url_entries = f"{self.config.get('url')}/entries.json?token={token}&count=150"
        self.url_treatments = f"{self.config.get('url')}/treatments.json?token={token}&count=10"
        self.url_ping_entries = f"{self.config.get('url')}/entries.json?token={token}&count=1"
        self.url_iob = f"{self.config.get('url')}/properties/iob?token={token}"
        self.GLUCOSE_LOW = self.config.get('low bondary glucose')
        self.GLUCOSE_HIGHT = self.config.get('high bondary glucose')
        self.os = self.config.get('os', 'linux').lower()
        self.image_out = self.config.get('image out', 'led matrix')
        self.output_type = self.config.get("output type")
        self.night_brightness = self.config.get('night_brightness')
        self.PLOT_GLUCOSE_INTERVALS: bool = self.config.get('plot glucose intervals', True)

    def _setup_paths(self):
        """Initialize file paths for images and outputs."""
        self.NO_DATA_IMAGE_PATH = os.path.join('images', 'nocgmdata.png')
        self.NO_WIFI_IMAGE_PATH = os.path.join('images', 'no_wifi.png')
        self.OUTPUT_IMAGE_PATH = os.path.join("temp", "output_image.png")
        self.OUTPUT_GIF_PATH = os.path.join("temp", "output_gif.gif")
               
    def update_glucose_command(self, image_path=None):
        """Update the LED matrix with latest glucose data.
        
        Args:
            image_path: Optional path to specific image file to display
        """
        logging.info("Updating glucose command.")
        self.json_entries_data = self.fetch_json_data(self.url_entries)
        self.json_treatments_data = self.fetch_json_data(self.url_treatments)
        self.json_iob = self.fetch_json_data(self.url_iob)

        if self.json_entries_data:
            self.parse_matrix_values()
            self._set_pixels_time()
            self.pixelMatrix = self.build_pixel_matrix()

            if image_path:
                output_path = image_path
            elif self.output_type == "image":
                output_path = self.OUTPUT_IMAGE_PATH
                self.pixelMatrix.generate_image(output_path)
            else:
                self.pixelMatrix.generate_timer_gif()
                output_path = self.OUTPUT_GIF_PATH

            type_command = "--image true --set-image" if image_path or self.output_type == "image" else "--set-gif"

            self.reset_formmated_jsons()
            logging.info(f"Output generated and saved as {output_path}")

            if self.os == 'windows':
                self.command = f"idotmatrix/run_in_venv.bat --address {self.ip} {type_command} {output_path}"
            else:
                self.command = f"./idotmatrix/run_in_venv.sh --address {self.ip} {type_command} {output_path}"
        logging.info(f"Command updated: {self.command}")



    def run_command(self):
        """Execute the prepared command to update LED matrix display."""
        logging.info(f"Running command: {self.command}")
        if self.image_out != "led matrix":            
            logging.info("Image output is not 'led matrix', skipping command execution.")
            return

        for _ in range(1,5):
            try:
                result = subprocess.run(self.command, shell=True, check=True)
                if result.returncode != 0:
                    logging.error("Command failed.")
                else:
                    logging.info(f"Command executed successfully, with last glucose: {self.first_glucose_entry}")
                    self.increase_command_run_count()
                    break
            except subprocess.CalledProcessError as e:
                logging.error(f"Command failed with error: {e}")
                time.sleep(2)

    def set_command(self, command: str):
        """Set the command string for LED matrix communication.
        
        Args:
            command: Shell command to execute
        """
        self.command = command

    def run_command_in_loop(self):
        """Main application loop - continuously fetch and display glucose data."""
        logging.info("Starting command loop.")
        last_communication = datetime.datetime.now()
        
        while True:
            try:
                ping_json = self.fetch_json_data(self.url_ping_entries)[0]
                time_since_last_communication = (datetime.datetime.now() - last_communication).total_seconds()
                logging.info(f"Time since last communication: {time_since_last_communication:.2f} seconds")
                
                if self.is_run_command_count_exceeded():
                    logging.info("Run command count exceeded, resetting command.")
                    self.run_reset_command()
                    self.reset_run_command_count()

                if self.last_nightstate is None or self.has_dayshift_change(self.last_nightstate):
                    self.run_set_brightness_command()
                    self.last_nightstate = self.get_nightmode()

                if not ping_json or self.is_old_data(ping_json, self.max_time, logging_enabled=True):
                    if self.NO_DATA_IMAGE_PATH in self.command:
                        continue
                    logging.info("Old or missing data detected, updating to no data image.")
                    self.update_glucose_command(self.NO_DATA_IMAGE_PATH)
                    self.run_command()

                elif ping_json.get("_id") != self.newer_id or time_since_last_communication > 330:
                    logging.info("New data detected." if ping_json.get("_id") != self.newer_id else "No new data, but time since last communication exceeded threshold.")
                    self.json_entries_data = self.fetch_json_data(self.url_entries)
                    self.update_glucose_command()
                    self.run_command()
                    self.newer_id = ping_json.get("_id")
                    last_communication = datetime.datetime.now()
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error in the loop: {e}")
                time.sleep(60)

    def increase_command_run_count(self) -> None:
        """Increment the count of executed commands."""
        self.run_command_count += 1
    
    def is_run_command_count_exceeded(self) -> bool:
        """Check if the command run count exceeds the maximum allowed.
        
        Returns:
            bool: True if run command count exceeds the limit
        """
        count = self.RUN_COMMAND_MAX_COUNT
        if self.run_command_count > count:
            logging.info(f"Command run count ({self.run_command_count}) is higher than {count}.")
            return True
        else:
            return False

    def reset_run_command_count(self) -> None:
        """Reset the command run count to zero."""
        logging.info("Resetting run command count.")
        self.run_command_count = 0

    def run_reset_command(self):
        """Send reset command to LED matrix to clear due to memory limitation of the idot matrix."""
        logging.info("Running reset command.")
        if self.os == 'windows':
            command = f"idotmatrix/run_in_venv.bat --address {self.ip} --reset"
        else:
            command = f"./idotmatrix/run_in_venv.sh --address {self.ip} --reset"
        self.set_command(command)
        self.run_command()

    def run_set_brightness_command(self):
        """Send Adjust LED matrix brightness based on day/night mode."""
        brightness = self.night_brightness if self.get_nightmode() else 100
        if self.os == 'windows':
            command = f"idotmatrix/run_in_venv.bat --address {self.ip} --set-brightness {brightness}"
        else:
            command = f"./idotmatrix/run_in_venv.sh --address {self.ip} --set-brightness {brightness}"
        self.set_command(command)
        self.run_command()

    def reset_formmated_jsons(self):
        """Clear the formatted data arrays for next update cycle."""
        self.formmated_entries = []
        self.formmated_treatments = []

    def fetch_json_data(self, url, retries=5, delay=10, fallback_delay=300):
        """Fetch JSON data from Nightscout server with retry logic.
        
        Args:
            url: API endpoint URL
            retries: Number of retry attempts
            delay: Delay between retries (seconds)
            fallback_delay: Extended delay after max retries (seconds)
            
        Returns:
            dict: JSON response data
        """
        attempt = 0
        while True:
            try:
                logging.info(f"Fetching glucose data from {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                logging.info("Glucose data fetched successfully.")
                return response.json()

            except RemoteDisconnected as e:
                logging.error(f"Remote end closed connection on attempt {attempt + 1}: {e}")
                self.update_glucose_command(self.NO_WIFI_IMAGE_PATH)
                self.run_command()

            except requests.exceptions.ConnectionError as e:
                logging.error(f"Connection error on attempt {attempt + 1}: {e}")

            except requests.exceptions.Timeout as e:
                logging.error(f"Request timed out on attempt {attempt + 1}: {e}")

            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching data on attempt {attempt + 1}: {e}")

            # Handle retries and delays
            attempt += 1
            if attempt < retries:
                logging.info(f"Retrying in {delay} seconds... (Attempt {attempt} of {retries})")
                time.sleep(delay)
            else:
                logging.error(f"Max retries ({retries}) reached. Retrying in {fallback_delay} seconds.")
                attempt = 0  # Reset attempts after max retries
                time.sleep(fallback_delay)  # Wait longer before retrying again

    def set_arrow(self):
        """Extract glucose trend arrow from latest SGV entry."""
        for item in self.formmated_entries:
            if item.type == EntrieEnum.SGV:
                self.arrow = item.direction
                break

    def parse_matrix_values(self):
        """Process all fetched data into display-ready format."""
        self.generate_list_from_entries_json()
        self.generate_list_from_treatments_json()
        self.extract_first_and_second_value()
        self.set_arrow()
        self.inser_iob_item_from_json()
        
    def has_dayshift_change(self, previus_nightmode: bool):
        """Check if day/night mode has changed since last update.
        
        Args:
            previus_nightmode: Previous night mode state
            
        Returns:
            bool: True if mode has changed
        """
        nightmode = self.get_nightmode()
        return nightmode != previus_nightmode
        
    def get_nightmode(self) -> bool:
        """Determine if current time is in night mode.
        
        Returns:
            bool: True if current time is night mode (21:00-06:00)
        """
        DAY_START = 6
        DAY_END = 21
        current_time = datetime.datetime.now()
        return current_time.hour < DAY_START or current_time.hour > DAY_END

    def build_pixel_matrix(self) -> PixelMatrix:
        """Construct the complete pixel matrix with all data visualizations.
        
        Returns:
            PixelMatrix: Configured matrix ready for display
        """
        bolus_with_x_values,carbs_with_x_values,exercises_with_x_values = self.get_treatments_x_values()

        exercise_indexes = self.get_exercises_index()
        interpolated_iob_items = self.get_interpolated_iob_series()

        pixelMatrix = PixelMatrix(self.matrix_size,self.min_glucose,self.max_glucose, self.GLUCOSE_LOW, self.GLUCOSE_HIGHT, self.night_brightness, self.PIXEL_INTERVAL)
        pixelMatrix.set_formmated_entries(self.formmated_entries)
        pixelMatrix.set_formmated_treatments(self.formmated_treatments)
        pixelMatrix.set_arrow(self.arrow)
        pixelMatrix.set_glucose_difference(self.calc_glucose_difference())

        if self.PLOT_GLUCOSE_INTERVALS: pixelMatrix.draw_glucose_intervals()
        
        pixelMatrix.draw_hour_indicators()
        pixelMatrix.draw_glucose_boundaries()
        
        pixelMatrix.draw_iob(interpolated_iob_items)
        pixelMatrix.draw_carbs(carbs_with_x_values)
        pixelMatrix.draw_bolus(bolus_with_x_values)
        pixelMatrix.draw_exercise(exercise_indexes)
        
        
        pixelMatrix.display_entries()
        pixelMatrix.display_glucose_on_matrix(self.first_glucose_entry.glucose)

        return pixelMatrix

    def extract_first_and_second_value(self):
        """Extract the two most recent glucose values."""
        first_value_saved_flag = False
        for item in self.formmated_entries:
            if item.type == EntrieEnum.SGV and not first_value_saved_flag:
                self.first_glucose_entry = item
                first_value_saved_flag = True
                continue
            if item.type == EntrieEnum.SGV:
                self.second_glucose_entry = item
                break
        return self.first_glucose_entry, self.second_glucose_entry

    def get_exercises_index(self) -> set[int]:
        """Calculate X coordinates for exercise period indicators.
        
        Returns:
            set[int]: Set of X coordinates where exercise periods overlap with entries
        """
        exercise_indexes = set()
        for treatment in self.formmated_treatments:
            if treatment.type != TreatmentEnum.EXERCISE:
                continue

            exercise_start = treatment.date
            exercise_end = exercise_start + datetime.timedelta(minutes=treatment.amount)

            for index, entry in enumerate(self.formmated_entries):
                if exercise_start <= entry.date <= exercise_end:
                    exercise_indexes.add(self.matrix_size - 1 - index)

        return exercise_indexes

    def generate_list_from_entries_json(self):
        """Convert JSON entries data to GlucoseItem objects.
        
        Args:
            entries_margin: Extra entries to fetch beyond matrix size
        """
        for item in self.json_entries_data:
            treatment_date = datetime.datetime.strptime(item.get("dateString"), "%Y-%m-%dT%H:%M:%S.%fZ")
            treatment_date += datetime.timedelta(minutes= -180)
            if treatment_date < datetime.datetime.now() - datetime.timedelta(minutes= self.matrix_size * self.PIXEL_INTERVAL + self.PIXEL_INTERVAL / 2):
                break
            if item.get("type") == EntrieEnum.SGV:
                self.formmated_entries.append(GlucoseItem(EntrieEnum.SGV,
                                                  int(item.get(EntrieEnum.SGV)),
                                                  treatment_date,
                                                  item.get("direction")))
            elif item.get("type") == EntrieEnum.MBG:
                self.formmated_entries.append(GlucoseItem(EntrieEnum.MBG,
                                                  int(item.get(EntrieEnum.MBG)),
                                                  treatment_date))

    def generate_list_from_treatments_json(self):
        """Convert JSON treatments data to TreatmentItem and ExerciseItem objects."""
        for item in self.json_treatments_data:
            time = datetime.datetime.strptime(item.get("created_at"), "%Y-%m-%dT%H:%M:%S.%fZ") + datetime.timedelta(minutes=item.get('utcOffset', 0))
            if 'xDrip4iOS' in item.get("enteredBy"): 
                time += datetime.timedelta(minutes= -180)
            if item.get("eventType") == TreatmentEnum.CARBS.value:
                if not item.get("carbs"):
                    continue
                self.formmated_treatments.append(TreatmentItem(item.get("_id"),
                                                                    TreatmentEnum.CARBS,
                                                                    time,
                                                                    item.get("carbs")))
            elif item.get("eventType") == TreatmentEnum.BOLUS.value:
                if not item.get("insulin"):
                    continue
                self.formmated_treatments.append(TreatmentItem(item.get("_id"),
                                                                    TreatmentEnum.BOLUS,
                                                                    time,
                                                                    item.get("insulin")))
            elif item.get("eventType") == TreatmentEnum.EXERCISE.value:
                if not item.get("duration"):
                    continue
                self.formmated_treatments.append(ExerciseItem(TreatmentEnum.EXERCISE,
                                                                    time,
                                                                    int(item.get("duration"))))
    
    def calc_glucose_difference(self) -> int:
        """
        Calculate glucose difference between the most recent glucose value
        and the interpolated value 5 minutes before.
        Handles irregular intervals by linear interpolation.
        """
        if not self.formmated_entries or len(self.formmated_entries) < 2:
            return 0

        # most recent entry
        first = self.extract_first_and_second_value()[0]
        target_time = first.date - datetime.timedelta(minutes=5)

        # find two entries surrounding target_time
        before = None
        after = None
        for entry in self.formmated_entries[1:]:
            if entry.type != EntrieEnum.SGV:
                continue
            
            if entry.date <= target_time:
                before = entry
                break
            
            after = entry

        if before and after:
            # interpolate linearly
            total_delta = (after.date - before.date).total_seconds()
            if total_delta == 0:
                past_glucose = before.glucose
            else:
                ratio = (target_time - before.date).total_seconds() / total_delta
                past_glucose = before.glucose + ratio * (after.glucose - before.glucose)
        elif before:  
            # no after entry, use nearest before
            past_glucose = before.glucose
        elif after:   
            # no before entry, use nearest after
            past_glucose = after.glucose
        else:
            return 0
        
        return round(first.glucose - past_glucose)

    def is_old_data(self, json, max_time, logging_enabled=False):
        """Check if the glucose data is older than acceptable threshold.
        
        Args:
            json: JSON data with sysTime field
            max_time: Maximum acceptable age in milliseconds
            logging_enabled: Whether to log data age
            
        Returns:
            bool: True if data is too old
        """
        created_at_str = json.get('sysTime')

        if created_at_str is None:
            raise ValueError("No 'sysTime' timestamp found in the JSON data.")

        created_at = datetime.datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))

        current_time = datetime.datetime.now(datetime.timezone.utc)

        time_difference_ms = (current_time - created_at).total_seconds() * 1000

        time_difference_sec = time_difference_ms / 1000
        minutes = int(time_difference_sec // 60)
        seconds = int(time_difference_sec % 60)

        if logging_enabled:
            logging.info(f"The data is {minutes:02d}:{seconds:02d} old.")

        return time_difference_ms > max_time


    def unblock_bluetooth(self):
        """Unblock Bluetooth on Linux systems for LED matrix communication."""
        try:
            logging.info("Attempting to unblock Bluetooth...")
            subprocess.run(['sudo', 'rfkill', 'unblock', 'bluetooth'], check=True, text=True, capture_output=True)
            logging.info(f"Bluetooth unblocked successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to unblock Bluetooth: {e.stderr}")

    def calculate_time_difference(self):
        """Calculate time difference between now and latest glucose reading.
        
        Returns:
            int: Time difference in minutes
        """
        current_time = datetime.datetime.now()
        time_difference = current_time - self.formmated_entries[0].date
        minutes_difference = time_difference.total_seconds() // 60
        return int(minutes_difference)

    def get_treatments_x_values(self):
        """Calculate X coordinates for treatment markers on the timeline.
        
        Returns:
            tuple: Three lists (bolus_values, carbs_values, exercise_values)
        """
        if not self.formmated_entries:
            logging.warning("No glucose entries available.")
            return [], [], []

        newer_entry_time = self.formmated_entries[0].date
        older_entry_time = self.formmated_entries[-1].date

        entry_dates = {entry: idx for idx, entry in enumerate(self.pixels_time)}

        bolus_values = []
        carbs_values = []
        exercise_values = []

        for treatment in self.formmated_treatments:
            # Skip treatments outside the time window
            treatment_end_time = (treatment.date + datetime.timedelta(minutes=treatment.amount) 
                                if treatment.type == TreatmentEnum.EXERCISE else treatment.date)
            
            if ((treatment.type == TreatmentEnum.EXERCISE and 
                (treatment_end_time < older_entry_time or treatment.date > newer_entry_time)) or
                (treatment.type != TreatmentEnum.EXERCISE and 
                (treatment.date < older_entry_time or treatment.date > newer_entry_time))):
                continue
                
            closest_date = self._find_closest_date(treatment.date, self.pixels_time[::-1])
            
            if closest_date is None:
                continue
                
            x_value = entry_dates[closest_date]
            matrix_x = self.matrix_size - x_value - 1
            
            # Process by treatment type
            if treatment.type == TreatmentEnum.EXERCISE:
                time_elapsed = max(0, (older_entry_time - treatment.date).total_seconds() / 60)
                remaining_time = math.ceil(treatment.amount - time_elapsed) if time_elapsed > 0 else treatment.amount
                
                exercise_values.append((matrix_x, remaining_time, treatment.type))
                
            elif treatment.type == TreatmentEnum.BOLUS:
                bolus_values.append((matrix_x, treatment.amount, treatment.type))
                
            elif treatment.type == TreatmentEnum.CARBS:
                carbs_values.append((matrix_x, treatment.amount, treatment.type))
        
        return bolus_values, carbs_values, exercise_values

    def _find_closest_date(self, target_date, date_list):
        """Find the closest date in a sorted list using binary search.
        
        Args:
            target_date: Date to find closest match for
            date_list: Sorted list of datetime objects
            
        Returns:
            datetime: Closest date from the list, or None if empty
        """
        if not date_list:
            return None
            
        # Handle edge cases
        if target_date <= date_list[0]:
            return date_list[0]
        if target_date >= date_list[-1]:
            return date_list[-1]
        
        # Binary search
        left, right = 0, len(date_list) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if date_list[mid] == target_date:
                return date_list[mid]
                
            if date_list[mid] < target_date:
                left = mid + 1
            else:
                right = mid - 1
        
        left = min(max(left, 0), len(date_list) - 1)
        
        if left == 0:
            return date_list[0]
        if left == len(date_list) - 1:
            return date_list[-1]
        
        before = date_list[left-1]
        after = date_list[left]
        
        if abs((target_date - before).total_seconds()) <= abs((after - target_date).total_seconds()):
            return before
        else:
            return after

    def inser_iob_item_from_json(self):
        """Insert latest IOB sample (with timestamp) into self.iob_list (newest first)."""
        iob_value = self.json_iob.get("iob", {}).get("iob", None)

        current_time = datetime.datetime.now()
        amount = float(iob_value) if iob_value is not None else 0.0
        # Prevent duplicate timestamp inserts (same minute)
        if not self.iob_list or abs((current_time - self.iob_list[0].date).total_seconds()) >= 30:
            self.iob_list.insert(0, IobItem(current_time, round(amount, 3)))
        # Keep only needed history (slightly more than matrix for interpolation safety)
        max_keep = self.matrix_size * 2
        if len(self.iob_list) > max_keep:
            self.iob_list = self.iob_list[:max_keep]

    def get_interpolated_iob_series(self) -> List[IobItem]:
        """Return matrix_size IobItems (newest first) at PIXEL_INTERVAL spacing.
        Only interpolate between collected samples in self.iob_list.
        Do NOT extrapolate further into the past than the oldest stored sample.
        Older pixels beyond memory are filled with 0.0.
        """
        if not self.iob_list:
            now = datetime.datetime.now()
            return [IobItem(now - datetime.timedelta(minutes=i * self.PIXEL_INTERVAL), 0.0)
                    for i in range(self.matrix_size)]

        # Sort samples oldest -> newest
        samples = sorted(self.iob_list, key=lambda x: x.date)
        oldest_time = samples[0].date
        newest_time = samples[-1].date
        newest_value = float(samples[-1].amount)

        # Pre-extract for faster search
        sample_times = [s.date for s in samples]
        sample_vals = [float(s.amount) for s in samples]

        def interpolate(target: datetime.datetime) -> float:
            # Clamp to newest (should only occur for offset 0)
            if target >= newest_time:
                return newest_value
            # Outside (older) than memory: caller guards; return 0.0 if called
            if target < oldest_time:
                return 0.0
            # Binary search position
            idx = bisect.bisect_left(sample_times, target)
            if idx == 0:
                return sample_vals[0]
            if idx >= len(sample_times):
                return sample_vals[-1]
            left_t, right_t = sample_times[idx - 1], sample_times[idx]
            left_v, right_v = sample_vals[idx - 1], sample_vals[idx]
            span = (right_t - left_t).total_seconds()
            if span <= 0:
                return left_v
            ratio = (target - left_t).total_seconds() / span
            return left_v + ratio * (right_v - left_v)

        series: List[IobItem] = []
        for offset in range(self.matrix_size):
            t = newest_time - datetime.timedelta(minutes=offset * self.PIXEL_INTERVAL)
            if t < oldest_time:
                val = 0.0
            else:
                val = interpolate(t)
            series.append(IobItem(t, round(val, 3)))

        return series  # index 0 newest

    def _set_pixels_time(self):
        """Set the pixel time array based on current time and matrix size."""
        self.pixels_time = [datetime.datetime.now() - datetime.timedelta(minutes=i * self.PIXEL_INTERVAL) for i in range(self.matrix_size)]

if __name__ == "__main__":
    GlucoseMatrixDisplay().run_command_in_loop()
