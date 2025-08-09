import math
import os
import subprocess
from PIL import Image, ImageEnhance
import requests
import time
import json
import datetime
import logging
from logging.handlers import RotatingFileHandler
from typing import List
from http.client import RemoteDisconnected
from util import GlucoseItem, TreatmentItem, ExerciseItem, TreatmentEnum, EntrieEnum
from PixelMatrix import PixelMatrix

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = 'app.log'
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class GlucoseMatrixDisplay:
    def __init__(self, config_path=os.path.join('led_matrix_configurator', 'config.json'), matrix_size=32, min_glucose=60, max_glucose=180):
        self.matrix_size = matrix_size
        self.min_glucose = min_glucose
        self.max_glucose = max_glucose
        self.max_time = 5 * 60 * 1000 * self.matrix_size #milliseconds
        self.config = self.load_config(config_path)
        self.arrow = ''
        self.glucose_difference = 0
        self.first_value = 0
        self.second_value = 0
        self.formmated_entries: List[GlucoseItem] = []
        self.formmated_treatments: List[TreatmentItem | ExerciseItem] = []
        self.iob_list: List[float] = []
        self.newer_id = None
        self.command = ''
        self.last_nightstate = None
        self._load_config_values()
        self._setup_paths()
        if self.image_out == "led matrix" and self.os != "windows": self.unblock_bluetooth()

    def load_config(self, config_path) -> dict:
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
        self.ip = self.config.get('ip')
        token = self.config.get('token')
        self.url_entries = f"{self.config.get('url')}/entries.json?token={token}&count=40"
        self.url_treatments = f"{self.config.get('url')}/treatments.json?token={token}&count=10"
        self.url_ping_entries = f"{self.config.get('url')}/entries.json?token={token}&count=1"
        self.url_iob = f"{self.config.get('url')}/properties/iob?token={token}"
        self.GLUCOSE_LOW = self.config.get('low bondary glucose')
        self.GLUCOSE_HIGHT = self.config.get('high bondary glucose')
        self.os = self.config.get('os', 'linux').lower()
        self.image_out = self.config.get('image out', 'led matrix')
        self.output_type = self.config.get("output type")
        self.night_brightness = self.config.get('night_brightness', 0.3)  

    def _setup_paths(self):
        self.NO_DATA_IMAGE_PATH = os.path.join('images', 'nocgmdata.png')
        self.NO_WIFI_IMAGE_PATH = os.path.join('images', 'no_wifi.png')
        self.OUTPUT_IMAGE_PATH = os.path.join("temp", "output_image.png")
        self.OUTPUT_GIF_PATH = os.path.join("temp", "output_gif.gif")
               
    def update_glucose_command(self, image_path=None):
        logging.info("Updating glucose command.")
        self.json_entries_data = self.fetch_json_data(self.url_entries)
        self.json_treatments_data = self.fetch_json_data(self.url_treatments)
        self.json_iob = self.fetch_json_data(self.url_iob)

        if self.json_entries_data:
            self.parse_matrix_values()
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
                    logging.info(f"Command executed successfully, with last glucose: {self.first_value}")
                    break
            except subprocess.CalledProcessError as e:
                logging.error(f"Command failed with error: {e}")
                time.sleep(2)

    def set_command(self, command: str):
        self.command = command

    def run_command_in_loop(self):
        logging.info("Starting command loop.")
        last_comunication = datetime.datetime.now()
        
        while True:
            try:
                ping_json = self.fetch_json_data(self.url_ping_entries)[0]
                time_since_last_comunication = (datetime.datetime.now() - last_comunication).total_seconds()
                logging.info(f"Time since last communication: {time_since_last_comunication:.2f} seconds")

                if self.last_nightstate == None or self.has_dayshift_change(self.last_nightstate):
                    self.run_reset_command()
                    self.run_set_brightness_command()
                    self.last_nightstate = self.get_nightmode()

                if not ping_json or self.is_old_data(ping_json, self.max_time, logging_enabled=True):
                    if self.NO_DATA_IMAGE_PATH in self.command:
                        continue
                    logging.info("Old or missing data detected, updating to no data image.")
                    self.update_glucose_command(self.NO_DATA_IMAGE_PATH)
                    self.run_command()

                elif ping_json.get("_id") != self.newer_id or time_since_last_comunication > 330:
                    logging.info("New glucose data detected, updating display.")
                    self.json_entries_data = self.fetch_json_data(self.url_entries)
                    self.update_glucose_command()
                    self.run_command()
                    self.newer_id = ping_json.get("_id")
                    last_comunication = datetime.datetime.now()
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error in the loop: {e}")
                time.sleep(60)

    def run_reset_command(self):
        logging.info("Running reset command.")
        if self.os == 'windows':
            command = f"idotmatrix/run_in_venv.bat --address {self.ip} --reset"
        else:
            command = f"./idotmatrix/run_in_venv.sh --address {self.ip} --reset"
        self.set_command(command)
        self.run_command()

    def run_set_brightness_command(self):
        brightness = self.night_brightness if self.get_nightmode() else 100
        if self.os == 'windows':
            command = f"idotmatrix/run_in_venv.bat --address {self.ip} --set-brightness {brightness}"
        else:
            command = f"./idotmatrix/run_in_venv.sh --address {self.ip} --set-brightness {brightness}"
        self.set_command(command)
        self.run_command()

    def reset_formmated_jsons(self):
        self.formmated_entries = []
        self.formmated_treatments = []

    def fetch_json_data(self, url, retries=5, delay=10, fallback_delay=300):
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
        for item in self.formmated_entries:
            if item.type == EntrieEnum.SGV:
                self.arrow = item.direction
                break

    def parse_matrix_values(self):
        self.generate_list_from_entries_json()
        self.generate_list_from_treatments_json()
        self.extract_first_and_second_value()
        self.set_glucose_difference()
        self.set_arrow()
        self.iob_list = self.get_iob()
        
    def has_dayshift_change(self, previus_nightmode: bool):
        nightmode = self.get_nightmode()
        return nightmode != previus_nightmode
        
    def get_nightmode(self) -> bool:
        current_time = datetime.datetime.now()
        return current_time.hour < 6 or current_time.hour > 21

    def build_pixel_matrix(self):
        bolus_with_x_values,carbs_with_x_values,exercises_with_x_values = self.get_treatments_x_values()

        exercise_indexes = self.get_exercises_index()

        pixelMatrix = PixelMatrix(self.matrix_size,self.min_glucose,self.max_glucose, self.GLUCOSE_LOW, self.GLUCOSE_HIGHT, self.night_brightness)
        pixelMatrix.set_formmated_entries(self.formmated_entries)
        pixelMatrix.set_formmated_treatments(self.formmated_treatments)
        pixelMatrix.set_arrow(self.arrow)
        pixelMatrix.set_glucose_difference(self.glucose_difference)
 
        pixelMatrix.display_glucose_on_matrix(self.first_value)

        pixelMatrix.draw_axis()

        pixelMatrix.draw_iob(self.iob_list)
        pixelMatrix.draw_carbs(carbs_with_x_values)
        pixelMatrix.draw_bolus(bolus_with_x_values)
        pixelMatrix.draw_exercise(exercise_indexes)
        pixelMatrix.display_entries(self.formmated_entries)

        return pixelMatrix

    def extract_first_and_second_value(self):
        first_value_saved_flag = False
        for item in self.formmated_entries:
            if item.type == EntrieEnum.SGV and not first_value_saved_flag:
                self.first_value = item.glucose
                first_value_saved_flag = True
                continue
            if item.type == EntrieEnum.SGV:
                self.second_value = item.glucose
                break

    def get_exercises_index(self) -> set[int]:
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

    def generate_list_from_entries_json(self, entries_margin = 3):
        for item in self.json_entries_data:
            treatment_date = datetime.datetime.strptime(item.get("dateString"), "%Y-%m-%dT%H:%M:%S.%fZ")
            treatment_date += datetime.timedelta(minutes= -180)
            if item.get("type") == EntrieEnum.SGV:
                self.formmated_entries.append(GlucoseItem(EntrieEnum.SGV,
                                                  item.get(EntrieEnum.SGV),
                                                  treatment_date,
                                                  item.get("direction")))
            elif item.get("type") == EntrieEnum.MBG:
                self.formmated_entries.append(GlucoseItem(EntrieEnum.MBG,
                                                  item.get(EntrieEnum.MBG),
                                                  treatment_date))

            if len(self.formmated_entries) == self.matrix_size + entries_margin:
                break

    def generate_list_from_treatments_json(self):
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

    def set_glucose_difference(self):
        self.glucose_difference = int(self.first_value) - int(self.second_value)

    def get_glucose_difference_signal(self):
        return '-' if self.glucose_difference < 0 else '+'

    def is_old_data(self, json, max_time, logging_enabled=False):
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
        try:
            logging.info("Attempting to unblock Bluetooth...")
            subprocess.run(['sudo', 'rfkill', 'unblock', 'bluetooth'], check=True, text=True, capture_output=True)
            logging.info(f"Bluetooth unblocked successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to unblock Bluetooth: {e.stderr}")

    def calculate_time_difference(self):
        current_time = datetime.datetime.now()
        time_difference = current_time - self.formmated_entries[0].date
        minutes_difference = time_difference.total_seconds() // 60
        return int(minutes_difference)

    def get_treatments_x_values(self):
        if not self.formmated_entries:
            logging.warning("No glucose entries available.")
            return [], [], []

        newer_entry_time = self.formmated_entries[0].date
        older_entry_time = self.formmated_entries[-1].date

        # Lookup dictionary for entry dates to quickly find the index
        entry_dates = {entry.date: idx for idx, entry in enumerate(self.formmated_entries)}
        sorted_dates = sorted(entry_dates.keys())

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
                
            # Find closest entry date using binary search
            closest_date = self._find_closest_date(treatment.date, sorted_dates)
            
            # Skip if no close date was found
            if closest_date is None:
                continue
                
            x_value = entry_dates[closest_date]
            matrix_x = self.matrix_size - x_value - 1  # Adjust for matrix coordinates
            
            # Process by treatment type
            if treatment.type == TreatmentEnum.EXERCISE:
                # Calculate remaining exercise time
                time_elapsed = max(0, (older_entry_time - treatment.date).total_seconds() / 60)
                remaining_time = math.ceil(treatment.amount - time_elapsed) if time_elapsed > 0 else treatment.amount
                
                exercise_values.append((matrix_x, remaining_time, treatment.type))
                
            elif treatment.type == TreatmentEnum.BOLUS:
                bolus_values.append((matrix_x, treatment.amount, treatment.type))
                
            elif treatment.type == TreatmentEnum.CARBS:
                carbs_values.append((matrix_x, treatment.amount, treatment.type))
        
        return bolus_values, carbs_values, exercise_values

    def _find_closest_date(self, target_date, date_list):
        """
        Find the closest date to target_date in date_list using binary search.
        Returns None only if date_list is empty.
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
        
        # After binary search, left is the insertion point
        # We now know the date must be between dates[left-1] and dates[left]
        # unless left is at the boundaries
        
        # Ensure left is within bounds
        left = min(max(left, 0), len(date_list) - 1)
        
        # If at boundaries, return the boundary value
        if left == 0:
            return date_list[0]
        if left == len(date_list) - 1:
            return date_list[-1]
        
        # Compare distances to determine closest
        before = date_list[left-1]
        after = date_list[left]
        
        if abs((target_date - before).total_seconds()) <= abs((after - target_date).total_seconds()):
            return before
        else:
            return after

    def get_iob(self):
        iob_value = self.json_iob.get("iob", {}).get("iob", None)
        if iob_value == None:
            self.iob_list.insert(0,0)
        else:
            self.iob_list.insert(0,iob_value)
        return self.iob_list[:self.matrix_size]


if __name__ == "__main__":
    GlucoseMatrixDisplay().run_command_in_loop()
