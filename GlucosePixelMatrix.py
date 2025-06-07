import math
import os
import subprocess
from PIL import Image, ImageEnhance
import numpy as np
import requests
import time
import json
import datetime
import logging
from typing import List
from http.client import RemoteDisconnected
from util import GlucoseItem, TreatmentItem, ExerciseItem, TreatmentEnum, EntrieEnum
from PixelMatrix import PixelMatrix

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GlucoseMatrixDisplay:
    def __init__(self, config_path=os.path.join('led_matrix_configurator', 'config.json'), matrix_size=32, min_glucose=60, max_glucose=180):
        self.matrix_size = matrix_size
        self.min_glucose = min_glucose
        self.max_glucose = max_glucose
        self.max_time = 5 * 60 * 1000 * self.matrix_size #milliseconds
        self.config = self.load_config(config_path)
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
        self.night_brightness = float(self.config.get('night_brightness', 0.3))
        self.arrow = ''
        self.glucose_difference = 0
        self.first_value = 0
        self.second_value = 0
        self.formmated_entries: List[GlucoseItem] = []
        self.formmated_treatments: List[TreatmentItem | ExerciseItem] = []
        self.iob_list: List[float] = []
        self.newer_id = None
        self.command = ''
        if self.image_out == "led matrix" and self.os != "windows": self.unblock_bluetooth()

    def load_config(self, config_path):
        try:
            logging.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as file:
                config = json.load(file)
                logging.info(f"Configuration loaded successfully: {config}")
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error loading configuration file: {e}")
            raise Exception(f"Error loading configuration file: {e}")

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
                type_comand = "--image true --set-image"
            elif self.output_type == "image":
                output_path = os.path.join("temp", "output_image.png")
                self.pixelMatrix.generate_image(output_path)
                type_comand = "--image true --set-image"
            else:
                self.pixelMatrix.generate_timer_gif()
                output_path = os.path.join("temp", "output_gif.gif")
                type_comand = "--image true --set-gif"
            self.reset_formmated_jsons()

            if self.os == 'windows':
                self.command = f"idotmatrix/run_in_venv.bat --address {self.ip} {type_comand} {output_path}"
            else:
                self.command = f"./idotmatrix/run_in_venv.sh --address {self.ip} {type_comand} {output_path}"
        logging.info(f"Command updated: {self.command}")

    def run_command(self):
        logging.info(f"Running command: {self.command}")
        if self.image_out != "led matrix":
            img_path = os.path.join("temp", "output_image.png")
            img = Image.open(img_path)

            # Brighten the image
            enhancer = ImageEnhance.Brightness(img)
            bright_img = enhancer.enhance(1.5)  # Increase brightness by 50%

            # Concatenate images horizontally
            side_by_side = Image.new('RGB', (img.width * 2, img.height))
            side_by_side.paste(img, (0, 0))
            side_by_side.paste(bright_img, (img.width, 0))

            # Display the concatenated image
            side_by_side.show()

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

    def run_command_in_loop(self):
        logging.info("Starting command loop.")
        while True:
            try:
                ping_json = self.fetch_json_data(self.url_ping_entries)[0]
                if not ping_json or self.is_old_data(ping_json):
                    if "nocgmdata.png" in self.command:
                        continue
                    logging.info("Old or missing data detected, updating to no data image.")
                    self.update_glucose_command(os.path.join('images', 'nocgmdata.png'))
                    self.run_command()
                elif ping_json.get("_id") != self.newer_id:
                    logging.info("New glucose data detected, updating display.")
                    self.json_entries_data = self.fetch_json_data(self.url_entries)
                    self.update_glucose_command()
                    self.run_command()
                    self.newer_id = ping_json.get("_id")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error in the loop: {e}")
                time.sleep(60)

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
                self.update_glucose_command("./images/no_wifi.png")
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

    def is_old_data(self, json):
        created_at_str = json.get('sysTime')

        if created_at_str is None:
            raise ValueError("No 'sysTime' timestamp found in the JSON data.")

        created_at = datetime.datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))

        current_time = datetime.datetime.now(datetime.timezone.utc)

        time_difference_ms = (current_time - created_at).total_seconds() * 1000

        time_difference_sec = time_difference_ms / 1000
        minutes = int(time_difference_sec // 60)
        seconds = int(time_difference_sec % 60)

        logging.info(f"The data is {minutes:02d}:{seconds:02d} old.")

        return time_difference_ms > self.max_time

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
        bolus_with_x_values = []
        carbs_with_x_values = []
        exercises_with_x_values = []

        if not self.formmated_entries:
            logging.warning("No glucose entries available.")
            return bolus_with_x_values,carbs_with_x_values,exercises_with_x_values

        newer_entry_time = self.formmated_entries[0].date
        older_entry_time = self.formmated_entries[-1].date

        for treatment in self.formmated_treatments:
            if treatment.type == TreatmentEnum.EXERCISE:
                if treatment.date + datetime.timedelta(minutes=treatment.amount) < older_entry_time  or treatment.date > newer_entry_time:
                    continue
            else:
                if treatment.date < older_entry_time  or treatment.date > newer_entry_time:
                    continue

            # Calculate the x position based on the closest glucose entry
            closest_entry = min(self.formmated_entries, key=lambda entry: abs(treatment.date - entry.date))
            x_value = self.formmated_entries.index(closest_entry)

            if treatment.type == TreatmentEnum.EXERCISE:
                # Calculate time elapsed in minutes since the treatment started
                time_elapsed = (older_entry_time - treatment.date).total_seconds() / 60  # in minutes

                if time_elapsed > 0:
                    # If treatment started before the first entry, calculate remaining time
                    discount_time = treatment.amount - time_elapsed
                else:
                    # Calculate how much treatment time is remaining from the current position
                    discount_time = 0

                exercises_with_x_values.append((self.matrix_size - x_value - 1,
                                                math.ceil(treatment.amount - discount_time),
                                                treatment.type))

            elif treatment.type in (TreatmentEnum.BOLUS, TreatmentEnum.CARBS):
                # Ensure the treatment falls within the time range covered by glucose entries
                if older_entry_time <= treatment.date <= newer_entry_time:
                    if treatment.type == TreatmentEnum.BOLUS:
                        bolus_with_x_values.append((self.matrix_size - x_value - 1,
                                                    treatment.amount,
                                                    treatment.type))
                    else:
                        carbs_with_x_values.append((self.matrix_size - x_value - 1,
                                                    treatment.amount,
                                                    treatment.type))

        return bolus_with_x_values, carbs_with_x_values, exercises_with_x_values

    def get_iob(self):
        iob_value = self.json_iob.get("iob", {}).get("iob", None)
        if iob_value == None:
            self.iob_list.insert(0,0)
        else:
            self.iob_list.insert(0,iob_value)
        return self.iob_list[:self.matrix_size]


if __name__ == "__main__":
    GlucoseMatrixDisplay().run_command_in_loop()
