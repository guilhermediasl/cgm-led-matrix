from datetime import datetime
import math
from typing import List
import numpy as np
import png
import pytz
from PIL import Image
import os
from patterns import digit_patterns, arrow_patterns, signal_patterns
from util import Color, EntrieEnum, GlucoseItem, TreatmentItem, ColorType, ExerciseItem

class PixelMatrix:
    def __init__(self, matrix_size: int, min_glucose: int, max_glucose: int, GLUCOSE_LOW, GLUCOSE_HIGH, night_brightness):
        self.min_glucose = min_glucose
        self.matrix_size = matrix_size
        self.max_glucose = max_glucose
        self.GLUCOSE_LOW = GLUCOSE_LOW
        self.GLUCOSE_HIGH = GLUCOSE_HIGH
        self.night_brightness = night_brightness
        self.pixels = np.zeros((matrix_size, matrix_size, 3), dtype=np.uint8)

    def set_formmated_entries(self, formmated_entries: List[GlucoseItem]):
        self.formmated_entries = formmated_entries

    def set_formmated_treatments(self, formmated_treatments: List[TreatmentItem | ExerciseItem]):
        self.formmated_treatments = formmated_treatments

    def set_arrow(self, arrow: str):
        self.arrow = arrow

    def set_glucose_difference(self, glucose_difference: int):
        self.glucose_difference = glucose_difference

    def set_pixel(self, x: int, y: int, r: int, g: int, b: int):
        if 0 <= x < self.matrix_size and 0 <= y < self.matrix_size:
            self.pixels[y][x] = ColorType(r, g, b)

    def get_pixel(self, x: int, y: int) -> ColorType:
        return self.pixels[y][x]

    def paint_background(self, color):
        for y in range(self.matrix_size):
            for x in range(self.matrix_size):
                self.pixels[y][x] = color

    def set_interpoleted_pixel(self, x: int, y: int, glucose_start:int, color: ColorType, percentil: float):
        start_y = self.glucose_to_y_coordinate(glucose_start) + 2
        y = start_y + y
        if 0 <= x < self.matrix_size and 0 <= y < self.matrix_size:
            interpolated_color = self.interpolate_color(Color.black.rgb, color, percentil, 0, 1)
            self.pixels[y][x] = interpolated_color

    def draw_pattern(self, pattern: np.ndarray, x: int, y: int, color: ColorType):
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                if pattern[i, j]:
                    self.set_pixel(x + j, y + i, *color)

    def draw_vertical_line(self, x: int, color: ColorType, glucose: int, height: int, enable_five=False, blink=False):
        start_y = self.glucose_to_y_coordinate(glucose) + 2
        if start_y + height < self.matrix_size:
            y_max = start_y + height
        else:
            y_max = self.matrix_size

        for y in range(start_y, y_max):
            temp_color = color
            if blink:
                if y % 2 == 0:
                    temp_color = self.fade_color(color, 0.3)
            if enable_five:
                if not self.is_five_apart(start_y, y):
                    temp_color = self.fade_color(color, 0.5)

            self.set_pixel(x, y, *temp_color)

    def draw_horizontal_line(self, glucose: int, color: ColorType, start_x: int, finish_x: int):
        y = self.glucose_to_y_coordinate(glucose) + 1
        finish_x = min(start_x + finish_x, self.matrix_size)
        start_x = max(start_x, 0)
        for x in range(start_x, finish_x):
            self.set_pixel(x, y, *color)

    def draw_axis(self) -> None:
        # Draw hour indicators lines
        for idx in (12, 24):
            self.draw_vertical_line(self.matrix_size - 1 - idx, self.fade_color(Color.white.rgb, 0.02), self.GLUCOSE_HIGH, 18, blink=True)

        # Draw glucose boundaries lines
        for glucose in (self.GLUCOSE_LOW, self.GLUCOSE_HIGH):
            self.draw_horizontal_line(glucose, self.fade_color(Color.white.rgb, 0.1), 0, self.matrix_size)

    def draw_iob(self, iob_list: List[float]) -> None:
        for id,iob in enumerate(iob_list):
            fractional_iob, integer_iob = math.modf(iob)
            integer_iob = int(integer_iob)

            self.draw_vertical_line(self.matrix_size - id - 1,
                                            self.fade_color(Color.blue.rgb, 0.05),
                                            self.GLUCOSE_HIGH,
                                            integer_iob)

            if fractional_iob <= 0.1: continue

            self.set_interpoleted_pixel(self.matrix_size - id - 1,
                                                integer_iob,
                                                self.GLUCOSE_HIGH,
                                                self.fade_color(Color.blue.rgb, 0.1),
                                                fractional_iob)

    def draw_carbs(self, carbs_with_x_values: List) -> None:
        for treatment in carbs_with_x_values:
            self.draw_vertical_line(treatment[0],
                                    self.fade_color(Color.orange.rgb, 0.2),
                                    self.GLUCOSE_HIGH,
                                    treatment[1],
                                    True)

    def draw_bolus(self, bolus_with_x_values: List) -> None:
        for treatment in bolus_with_x_values:
            self.draw_vertical_line(treatment[0],
                                    self.fade_color(Color.blue.rgb, 0.3),
                                    self.GLUCOSE_HIGH,
                                    treatment[1],
                                    True)

    def draw_exercise(self, exercise_indexes: set[int]) -> None:
        for exercise_index in exercise_indexes:
            self.set_pixel(exercise_index, self.glucose_to_y_coordinate(self.GLUCOSE_HIGH) + 1, *self.fade_color(Color.purple.rgb, 0.5))
            self.set_pixel(exercise_index, self.glucose_to_y_coordinate(self.GLUCOSE_LOW) + 1, *self.fade_color(Color.purple.rgb, 0.5))


    def get_out_of_range_glucose_str(self, glucose: int) -> str:
        if glucose <= 39:
            return "LOW"
        elif glucose >= 400:
            return "HIGH"
        else:
            return str(glucose)

    def is_glucose_out_of_range(self, glucose: int) -> bool:
        return glucose <= 39 or glucose >= 400

    def get_digits_width(self, glucose_str: str) -> int:
        width = 0
        for digit in glucose_str:
            width += len(digit_patterns()[digit][0])
        return width

    def display_glucose_on_matrix(self, glucose_value: int) -> None:
        digit_width, digit_height, spacing = 3, 5, 1

        if self.is_glucose_out_of_range(glucose_value):
            glucose_str = self.get_out_of_range_glucose_str(glucose_value)
            color = Color.red.rgb
        else:
            glucose_str = str(glucose_value)
            color = Color.white.rgb


        digits_width = len(glucose_str) * spacing + self.get_digits_width(glucose_str)

        arrow_pattern = arrow_patterns().get(self.arrow, np.zeros((5, 5)))
        arrow_width = arrow_pattern.shape[1] + spacing
        signal_width = 3 + spacing

        glucose_diff_str = str(abs(self.glucose_difference))
        glucose_diff_width = len(glucose_diff_str) * (digit_width + spacing)
        total_width = digits_width + arrow_width + signal_width + glucose_diff_width

        x_position = (self.matrix_size - total_width) // 2 + 1
        y_position = (self.matrix_size - digit_height) // 2 - 13

        for digit in glucose_str:
            digit_pattern = digit_patterns()[digit]
            self.draw_pattern(digit_pattern, x_position, y_position, color)
            x_position += self.get_digit_width(digit) + spacing

        self.draw_pattern(arrow_pattern, x_position, y_position, color)
        x_position += arrow_width

        signal_pattern = signal_patterns()[self.get_glucose_difference_signal()]
        self.draw_pattern(signal_pattern, x_position, y_position, color)
        x_position += signal_width

        for digit in glucose_diff_str:
            digit_pattern = digit_patterns()[digit]
            self.draw_pattern(digit_pattern, x_position, y_position, color)
            x_position += digit_width + spacing

    def get_digit_width(self, digit: str) -> int:
        return len(digit_patterns()[digit][0])

    def display_entries(self, formmated_entries: List[GlucoseItem]):
        glucose_plot = [[] for _ in range(self.matrix_size)]
        now = datetime.now()

        for entry in formmated_entries:
            time_diff_minutes = (now - entry.date).total_seconds() / 60
            idx = int(time_diff_minutes // 5)
            
            if idx > self.matrix_size - 1:
                break
            
            if 0 <= idx < self.matrix_size:
                glucose_plot[idx].append(entry.glucose)
            
        trail_plotted = False
        for idx, glucose_values in enumerate(glucose_plot):
            if not glucose_values:
                continue
            
            median_glucose = int(np.average(glucose_values))
            x = self.matrix_size - idx - 1
            y = self.glucose_to_y_coordinate(median_glucose)
            r, g, b = self.determine_color(median_glucose)
            self.set_pixel(x, y, r, g, b)
            
            # Old glucose trail
            if not trail_plotted:
                past_idx = x
                fade_factor = 0.8
                r, g, b = Color.white.rgb
                
                while past_idx <= self.matrix_size - 1:
                    past_idx += 1
                    
                    r, g, b = self.fade_color(ColorType(r, g, b), fade_factor)
                    if r > 0 or g > 0 or b > 0:
                        self.set_pixel(past_idx, y, r, g, b)
                trail_plotted = True

    def get_low_brightness_pixels(self) -> List[List[ColorType]]:
        brightness = self.get_brightness_on_hour()
        low_brightness_pixels = [
            [self.fade_color(self.get_pixel(x, y), brightness) for x in range(self.matrix_size)]
            for y in range(self.matrix_size)
        ]

        return low_brightness_pixels

    def generate_image(self, output_file="output_image.png"):

        pixel_matrix = self.pixels

        png_matrix = []
        for row in pixel_matrix:
            png_matrix.append([val for pixel in row for val in pixel])

        with open(output_file, "wb") as f:
            writer = png.Writer(self.matrix_size, self.matrix_size, greyscale=False) # type: ignore
            writer.write(f, png_matrix)

    # def generate_timer_gif(self, output_file=os.path.join("temp", "output_gif.gif")):
    #     frame_files = []
    #     first_frame_path = os.path.join("temp", "frame-0.png")
    #     frame_files.append(first_frame_path)

    #     for index in range(1,6):
    #         self.set_pixel(0, index - 1, *self.fade_color(Color.white.rgb, 0.1))

    #     self.generate_image(first_frame_path)

    #     for index in range(1,6):
    #         self.set_pixel(0, index - 1, *self.fade_color(Color.pink.rgb, 1))
    #         frame_filename = os.path.join("temp", f"frame-{index}.png")
    #         self.generate_image(frame_filename)
    #         frame_files.append(frame_filename)

    #     frames = [Image.open(frame) for frame in frame_files]
    #     frames[0].save(
    #         output_file,
    #         save_all=True,
    #         append_images=frames[1:],
    #         duration=(30000,60000,60000,60000,60000,60000),  # 30 seconds for the first frame, 1 minute for the others
    #         loop=None
    #     )

    def generate_timer_gif(self, output_file=os.path.join("temp", "output_gif.gif")):
        frame_files = []
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        # Generate the first frame (base frame)
        first_frame_path = os.path.join(temp_dir, "frame-0.png")
        self.generate_image(first_frame_path)
        frame_files.append(first_frame_path)

        # Generate transparent frames with one visible pixel
        for index in range(1, 6):
            transparent_frame = np.zeros((self.matrix_size, self.matrix_size, 4), dtype=np.uint8)  # RGBA fully transparent
            x, y = 0, index - 1
            r, g, b = self.fade_color(Color.pink.rgb, 1)
            transparent_frame[y][x] = [r, g, b, 255]  # Set one pixel to full opacity

            frame_filename = os.path.join(temp_dir, f"frame-{index}.png")
            Image.fromarray(transparent_frame, mode="RGBA").save(frame_filename)
            frame_files.append(frame_filename)

        # Open images and convert to palette with transparency
        frames = [Image.open(f).convert("RGBA") for f in frame_files]
        palette_frames = []

        for frame in frames:
            alpha = frame.getchannel("A")
            rgb_image = frame.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=255) # type: ignore

            # Create transparency mask: transparent pixels will get index 255
            transparent_index = 255
            mask = Image.eval(alpha, lambda a: 255 if a <= 0 else 0)
            rgb_image.paste(transparent_index, mask)

            rgb_image.info["transparency"] = transparent_index
            rgb_image.info["disposal"] = 1
            palette_frames.append(rgb_image)

        # Save the GIF
        palette_frames[0].save(
            output_file,
            save_all=True,
            append_images=palette_frames[1:],
            duration=(30000, 60000, 60000, 60000, 60000, 60000),
            loop=None,
            disposal=1
        )

    def glucose_to_y_coordinate(self, glucose: int) -> int:
        glucose = max(self.min_glucose, min(glucose, self.max_glucose))
        available_y_range = self.matrix_size - 6
        normalized = (glucose - self.min_glucose) / (self.max_glucose - self.min_glucose)
        return int((1 - normalized) * available_y_range) + 5

    def get_brightness_on_hour(self, timezone_str="America/Recife") -> float:
        local_tz = pytz.timezone(timezone_str)
        current_time = datetime.now(local_tz)
        current_hour = current_time.hour

        if 21 <= current_hour or current_hour < 6:
            return self.night_brightness
        else:
            return 1.0

    def determine_color(self, glucose: float, entry_type=EntrieEnum) -> ColorType:
        if entry_type == EntrieEnum.MBG:
            return Color.white.rgb

        if glucose < self.GLUCOSE_LOW - 10:
            return self.interpolate_color(Color.red.rgb, Color.yellow.rgb, glucose, self.get_min_sgv(), self.GLUCOSE_LOW - 10,)
        if glucose > self.GLUCOSE_HIGH + 10:
            return self.interpolate_color(Color.yellow.rgb, Color.red.rgb, glucose, self.GLUCOSE_HIGH + 10, self.get_max_sgv())
        elif glucose <= self.GLUCOSE_LOW or glucose >= self.GLUCOSE_HIGH:
            return Color.yellow.rgb
        else:
            return Color.green.rgb

    def interpolate_color(self, low_color: ColorType, high_color: ColorType, value: float, min_value: int, max_value: int) -> ColorType:
        if value < min_value:
            value = min_value
        elif value > max_value:
            value = max_value

        t = (value - min_value) / (max_value - min_value)

        r = int(low_color[0] + t * (high_color[0] - low_color[0]))
        g = int(low_color[1] + t * (high_color[1] - low_color[1]))
        b = int(low_color[2] + t * (high_color[2] - low_color[2]))

        return ColorType(r, g, b)

    def get_glucose_difference_signal(self) -> str:
        return '-' if self.glucose_difference < 0 else '+'

    def get_max_sgv(self) -> int:
        max_sgv = 0
        for entry in self.formmated_entries:
            max_sgv = max(max_sgv, entry.glucose)

        return max_sgv

    def get_min_sgv(self) -> int:
        min_sgv = self.formmated_entries[0].glucose
        for entry in self.formmated_entries:
            min_sgv = min(min_sgv, entry.glucose)

        return min_sgv

    def is_five_apart(self, init: int, current: int) -> bool:
        return (current - init + 1) % 5 == 0

    def fade_color(self, color: ColorType, percentil: float) -> ColorType:
        corrected_color = []

        # Smooth the boost more aggressively toward low percentils
        BASE = 0.8
        MAX_BOOST = 1.5
        EXPONENT = 8.0

        # Only boost red/green when brightness is low
        red_green_correction = BASE + (MAX_BOOST - BASE) * ((1 - percentil) ** EXPONENT)

        correction_factors = (red_green_correction, red_green_correction, 1.0)

        for idx, item in enumerate(color):
            corrected = round(item * percentil * correction_factors[idx])
            corrected_color.append(min(255, max(0, corrected)))

        return ColorType(*corrected_color)