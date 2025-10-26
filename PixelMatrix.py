from datetime import datetime
import math
from typing import List
import numpy as np
import png
import pytz
from PIL import Image
import os
from patterns import get_digit_patterns, get_arrow_patterns, get_signal_patterns
from util import Color, EntrieEnum, GlucoseItem, IobItem, TreatmentItem, ColorType, ExerciseItem

class PixelMatrix:
    """LED matrix display controller for glucose monitoring visualization."""
    def __init__(self, matrix_size: int, min_glucose: int, max_glucose: int, GLUCOSE_LOW, GLUCOSE_HIGH, night_brightness, PIXEL_INTERVAL):
        """Initialize the pixel matrix with display parameters.
        
        Args:
            matrix_size: Size of the square LED matrix (e.g., 32 for 32x32)
            min_glucose: Minimum glucose value for display scaling
            max_glucose: Maximum glucose value for display scaling
            GLUCOSE_LOW: Lower threshold for normal glucose range
            GLUCOSE_HIGH: Upper threshold for normal glucose range
            night_brightness: Brightness factor for night mode (0.0 - 1.0)
        """
        self.min_glucose = min_glucose
        self.matrix_size = matrix_size
        self.max_glucose = max_glucose
        self.GLUCOSE_LOW = GLUCOSE_LOW
        self.GLUCOSE_HIGH = GLUCOSE_HIGH
        self.night_brightness = night_brightness
        self.pixels = np.zeros((matrix_size, matrix_size, 3), dtype=np.uint8)
        self.PIXEL_INTERVAL = PIXEL_INTERVAL
        self.GLUCOSE_LOW_OUT_OF_RANGE = 39
        self.GLUCOSE_HIGH_OUT_OF_RANGE = 400

    def set_formmated_entries(self, formmated_entries: List[GlucoseItem]):
        """Set the glucose entries data for display.
        
        Args:
            formmated_entries: List[GlucoseItem] of glucose readings
        """
        self.formmated_entries = formmated_entries

    def set_formmated_treatments(self, formmated_treatments: List[TreatmentItem | ExerciseItem]):
        """Set the treatment data for display.
        
        Args:
            formmated_treatments: List[TreatmentItem | ExerciseItem] of treatments (bolus, carbs, exercise) to visualize
        """
        self.formmated_treatments = formmated_treatments

    def set_arrow(self, arrow: str):
        """Set the glucose trend arrow direction.
        
        Args:
            arrow: Direction string indicating glucose trend
        """
        self.arrow = arrow

    def set_glucose_difference(self):
        """Set the glucose difference from the 2 previous reading.
        
        Args:
            glucose_difference: Change in glucose value (mg/dL)
        """
        buckets = getattr(self, "entries_by_time", None)
        if buckets:
            recent_values = [v for v in buckets if v is not None][:2]
            if len(recent_values) == 2:
                diff = recent_values[0] - recent_values[1]
            else:
                diff = 0
        # Fallback to raw entries if needed
        if diff == 0 and (not buckets or sum(v is not None for v in buckets) < 2):
            entries = getattr(self, "formmated_entries", [])
            if len(entries) >= 2:
                # entries[0] should be newest; if not, sort descending by date
                if entries[0].date < entries[1].date:
                    entries = sorted(entries, key=lambda e: e.date, reverse=True)
                g1 = entries[0].glucose
                g2 = entries[1].glucose
                if g1 is not None and g2 is not None:
                    diff = g1 - g2

        self.glucose_difference = diff
        
    def set_pixel(self, x: int, y: int, r: int, g: int, b: int):
        """Set a single pixel color on the matrix.
        
        Args:
            x: X coordinate (0 to matrix_size-1)
            y: Y coordinate (0 to matrix_size-1)
            r: Red component (0 - 255)
            g: Green component (0 - 255)
            b: Blue component (0 - 255)
        """
        if 0 <= x < self.matrix_size and 0 <= y < self.matrix_size:
            self.pixels[y][x] = ColorType(r, g, b)

    def get_pixel(self, x: int, y: int) -> ColorType:
        """Get the color of a pixel at specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            ColorType: RGB color values of the pixel
        """
        return self.pixels[y][x]

    def paint_background(self, color):
        """Fill the entire matrix with a solid color.
        
        Args:
            color: ColorType tuple for the background color
        """
        for y in range(self.matrix_size):
            for x in range(self.matrix_size):
                self.pixels[y][x] = color

    def set_interpolated_pixel(self, x: int, y: int, glucose_start:int, color: ColorType, percentil: float):
        """Set a pixel with interpolated color based on glucose level.
        
        Args:
            x: X coordinate
            y: Y coordinate offset from glucose baseline
            glucose_start: Base glucose level for Y calculation
            color: Target color for interpolation
            percentil: Interpolation factor (0.0 - 1.0)
        """
        start_y = self.glucose_to_y_coordinate(glucose_start) + 2
        y = start_y + y
        if 0 <= x < self.matrix_size and 0 <= y < self.matrix_size:
            interpolated_color = self.interpolate_color(Color.black.rgb, color, percentil, 0, 1)
            self.pixels[y][x] = interpolated_color

    def draw_pattern(self, pattern: np.ndarray, x: int, y: int, color: ColorType):
        """Draw a bitmap pattern on the matrix.
        
        Args:
            pattern: 2D numpy array representing the pattern (1 = pixel on, 0 = pixel off)
            x: Starting X coordinate
            y: Starting Y coordinate
            color: RGB color for the pattern
        """
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                if pattern[i, j]:
                    self.set_pixel(x + j, y + i, *color)

    def draw_vertical_line(self, x: int, color: ColorType, glucose: int, height: int, enable_five=False, blink=False):
        """Draw a vertical line representing data values.
        
        Args:
            x: X coordinate for the line
            color: RGB color of the line
            glucose: Glucose level for Y positioning
            height: Height of the line in pixels
            enable_five: Whether to highlight every 5th pixel
            blink: Whether to apply blinking effect
        """
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
                if self.is_five_apart(start_y, y):
                    fade_amount = 1.4
                else:
                    fade_amount = 0.8
                    
                temp_color = self.fade_color(color, fade_amount)

            self.set_pixel(x, y, *temp_color)

    def draw_horizontal_line(self, glucose: int, color: ColorType, start_x: int, finish_x: int):
        """Draw a horizontal reference line at glucose level.
        
        Args:
            glucose: Glucose level determining Y position
            color: RGB color
            start_x: Starting X coordinate
            finish_x: Ending X coordinate
        """
        y = self.glucose_to_y_coordinate(glucose) + 1
        finish_x = min(start_x + finish_x, self.matrix_size)
        start_x = max(start_x, 0)
        for x in range(start_x, finish_x):
            self.set_pixel(x, y, *color)

    def draw_hour_indicators(self) -> None:
        """Draw vertical hour indicator lines on the matrix."""
        PIXELS_PER_HOUR = 12
        HOUR_COLUMN_HIGH = 18
        intervals = [i for i in range(PIXELS_PER_HOUR, self.matrix_size, PIXELS_PER_HOUR)]
        for idx in intervals:
            self.draw_vertical_line(self.matrix_size - 1 - idx, 
                                    self.fade_color(Color.white.rgb, 0.02), 
                                    self.GLUCOSE_HIGH, HOUR_COLUMN_HIGH, blink=True)

    def draw_glucose_boundaries(self) -> None:
        """Draw horizontal lines indicating target glucose range."""
        for glucose in (self.GLUCOSE_LOW, self.GLUCOSE_HIGH):
            self.draw_horizontal_line(glucose, self.fade_color(Color.white.rgb, 0.1), 0, self.matrix_size)

    def draw_iob(self, iob_list: List[IobItem]) -> None:
        """Draw insulin-on-board (IOB) visualization.
        
        Args:
            iob_list: List of IOB values over time
        """
        print("Drawing IOB values...")
        print(f"IOB list length: {len(iob_list)}")
        print("IOB list:", iob_list)
        for id,iobItem in enumerate(iob_list):
            fractional_iob, integer_iob = math.modf(iobItem.amount)
            integer_iob = int(integer_iob)
            
            # Draw integer IOB as a vertical solid line
            self.draw_vertical_line(self.matrix_size - id - 1,
                                            self.fade_color(Color.blue.rgb, 0.1),
                                            self.GLUCOSE_HIGH,
                                            integer_iob)

            MINIMUM_FRACTIONAL_IOB = 0.1
            if fractional_iob <= MINIMUM_FRACTIONAL_IOB: continue
            # Draw fractional IOB as an interpolated pixel
            self.set_interpolated_pixel(self.matrix_size - id - 1,
                                                integer_iob,
                                                self.GLUCOSE_HIGH,
                                                self.fade_color(Color.blue.rgb, 0.1),
                                                fractional_iob)

    def draw_carbs(self, carbs_with_x_values: List) -> None:
        """Draw carbohydrate intake markers.
        
        Args:
            carbs_with_x_values: List of tuples (x_position, carb_amount, type)
        """
        for treatment in carbs_with_x_values:
            self.draw_vertical_line(treatment[0],
                                    self.fade_color(Color.orange.rgb, 0.2),
                                    self.GLUCOSE_HIGH,
                                    treatment[1],
                                    True)

    def draw_bolus(self, bolus_with_x_values: List) -> None:
        """Draw bolus markers.
        
        Args:
            bolus_with_x_values: List of tuples (x_position, insulin_amount, type)
        """
        for treatment in bolus_with_x_values:
            self.draw_vertical_line(treatment[0],
                                    self.fade_color(Color.blue.rgb, 0.3),
                                    self.GLUCOSE_HIGH,
                                    treatment[1],
                                    True)

    def draw_exercise(self, exercise_indexes: set[int]) -> None:
        """Draw exercise period indicators.
        
        Args:
            exercise_indexes: Set of X coordinates where exercise occurred
        """
        for exercise_index in exercise_indexes:
            self.set_pixel(exercise_index,
                           self.glucose_to_y_coordinate(self.GLUCOSE_HIGH) + 1,
                           *self.fade_color(Color.purple.rgb, 0.5))
            
            self.set_pixel(exercise_index,
                           self.glucose_to_y_coordinate(self.GLUCOSE_LOW) + 1,
                           *self.fade_color(Color.purple.rgb, 0.5))

    def _draw_text(self, text: str, x: int, y: int, color: ColorType) -> int:
        """Draw text string.
        
        Args:
            text: String to draw
            x: Starting X coordinate
            y: Starting Y coordinate
            color: RGB color for the text
            
        Returns:
            int: X coordinate after drawing (for chaining)
        """
        for digit in text:
            pattern = get_digit_patterns()[digit]
            self.draw_pattern(pattern, x, y, color)
            x += self.get_digit_width(digit) + 1
        return x

    def get_out_of_range_glucose_str(self, glucose: int) -> str:
        """Convert out-of-range glucose values to display strings.
        
        Args:
            glucose: Glucose value in mg/dL
            
        Returns:
            str: "LOW", "HIGH", or numeric string
        """
        if glucose <= self.GLUCOSE_LOW_OUT_OF_RANGE:
            return "LOW"
        elif glucose >= self.GLUCOSE_HIGH_OUT_OF_RANGE:
            return "HIGH"
        else:
            return str(glucose)

    def is_glucose_out_of_range(self, glucose: int) -> bool:
        """Check if glucose value is outside normal sensor range.
        
        Args:
            glucose: Glucose value in mg/dL
            
        Returns:
            bool: True if value is out of sensor range
        """
        return glucose <= self.GLUCOSE_LOW_OUT_OF_RANGE or glucose >= self.GLUCOSE_HIGH_OUT_OF_RANGE

    def get_digits_width(self, glucose_str: str) -> int:
        """Calculate total width needed for digit string display.
        
        Args:
            glucose_str: String of digits to measure
            
        Returns:
            int: Total width in pixels
        """
        width = 0
        for digit in glucose_str:
            width += len(get_digit_patterns()[digit][0])
        return width

    def display_glucose_on_matrix(self, glucose_value: int) -> None:
        """Display current glucose value with trend arrow and difference.
        
        Args:
            glucose_value: Current glucose reading in mg/dL
        """
        DIGIT_WIDTH = 3
        DIGIT_HEIGHT = 5
        SPACING = 1
        SIGNAL_WIDTH = 3
        VERTICAL_OFFSET = 13

        if self.is_glucose_out_of_range(glucose_value):
            glucose_str = self.get_out_of_range_glucose_str(glucose_value)
            color = Color.red.rgb
        else:
            glucose_str = str(glucose_value)
            color = Color.white.rgb

        arrow_pattern = get_arrow_patterns().get(self.arrow, np.zeros((DIGIT_HEIGHT, DIGIT_HEIGHT)))
        signal_pattern = get_signal_patterns()[self.get_glucose_difference_signal(self.glucose_difference)]
        glucose_diff_str = str(abs(self.glucose_difference))

        glucose_digits_width = self.get_digits_width(glucose_str) + (len(glucose_str) - 1) * SPACING
        arrow_width = arrow_pattern.shape[1]
        glucose_diff_width = len(glucose_diff_str) * (DIGIT_WIDTH + SPACING) - SPACING
        total_width = (
            glucose_digits_width +
            SPACING + arrow_width +
            SPACING + SIGNAL_WIDTH +
            SPACING + glucose_diff_width
        )

        x = (self.matrix_size - total_width) // 2 + 1
        y = (self.matrix_size - DIGIT_HEIGHT) // 2 - VERTICAL_OFFSET

        x = self._draw_text(glucose_str, x, y, color)

        self.draw_pattern(arrow_pattern, x, y, color)
        x += arrow_width + SPACING

        self.draw_pattern(signal_pattern, x, y, color)
        x += SIGNAL_WIDTH + SPACING

        self._draw_text(glucose_diff_str, x, y, color)

    def get_digit_width(self, digit: str) -> int:
        """Get width of a digit pattern.
        
        Args:
            digit: Character digit
            
        Returns:
            int: Width in pixels
        """
        return len(get_digit_patterns()[digit][0])

    def draw_entries(self):
        """Display glucose readings as a timeline graph.
        
        Args:
            formmated_entries: List of glucose readings to plot
        """
        first_valid_point = True

        for pixel_interval_index, glucose_values in enumerate(self.entries_by_time):
            if not glucose_values:
                continue
            
            median_glucose = int(np.mean(glucose_values))
            x = self._time_index_to_x(pixel_interval_index)
            y = self.glucose_to_y_coordinate(median_glucose)

            self._plot_entry(x, y, median_glucose)

            if first_valid_point:
                first_valid_point = False
                if pixel_interval_index != 0:
                    self._draw_trail(x, y)
                
    def average_entries_by_time(self, entries) -> List[int]:
        """Group glucose entries into buckets indexed by minutes elapsed."""
        glucose_plot = [[] for _ in range(self.matrix_size)]
        now = datetime.now()

        for entry in entries:
            minutes_elapsed = (now - entry.date).total_seconds() / 60
            minutes_index = int(minutes_elapsed // self.PIXEL_INTERVAL)

            if minutes_index >= self.matrix_size:
                break
            if minutes_index >= 0:
                glucose_plot[minutes_index].append(entry.glucose)

        # Calculate average glucose for each time bucket
        average_entries_by_time = []
        for bucket in glucose_plot:
            if bucket:
                average_entries_by_time.append(int(np.mean(bucket)))
            else:
                average_entries_by_time.append(None)
        self.entries_by_time = average_entries_by_time
        return average_entries_by_time

    def _time_index_to_x(self, minutes_index: int) -> int:
        """Convert time bucket index to matrix X coordinate."""
        return self.matrix_size - minutes_index - 1

    def _plot_entry(self, x: int, y: int, glucose: int) -> None:
        """Plot a single glucose entry on the matrix."""
        r, g, b = self.determine_color(glucose)
        self.set_pixel(x, y, r, g, b)

    def _draw_trail(self, start_x: int, y: int) -> None:
        """Draw faded glucose trail to the right of the latest point."""
        MAX_TRAIL_LENGTH = 8
        FADE_FACTOR = 0.8

        r, g, b = Color.white.rgb
        for offset in range(1, MAX_TRAIL_LENGTH + 1):
            x = start_x + offset
            if x >= self.matrix_size:
                break

            r, g, b = self.fade_color(ColorType(r, g, b), FADE_FACTOR)
            if r == g == b == 0:
                break
            self.set_pixel(x, y, r, g, b)
        
    def get_no_data_pixels_amount(self) -> int:
        """Calculate number of pixels without data based on last entry time.
        
        Returns:
            int: Number of pixels without data
        """
        no_data_amount = 0
        last_entrys_time = self.formmated_entries[0].date
        no_data_amount = (datetime.now() - last_entrys_time).total_seconds() // (self.PIXEL_INTERVAL * 60)
        if no_data_amount > 1 and no_data_amount < 1.5:
            no_data_amount = 0
        return round(no_data_amount)

    def generate_image(self, output_file="output_image.png"):
        """Export current matrix as PNG image.
        
        Args:
            output_file: Name for the output PNG file
        """
        pixel_matrix = self.pixels

        png_matrix = []
        for row in pixel_matrix:
            png_matrix.append([val for pixel in row for val in pixel])

        with open(output_file, "wb") as f:
            writer = png.Writer(self.matrix_size, self.matrix_size, greyscale=False) # type: ignore
            writer.write(f, png_matrix)

    def generate_timer_gif(self, output_file=os.path.join("temp", "output_gif.gif"), timer_size=5):
        """Generate animated GIF with timer visualization.
        
        Args:
            output_file: Path for the output GIF file
            timer_size: Number of timer frames to generate
        """
        frame_files = []
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        # Generate base frame
        first_frame_path = os.path.join(temp_dir, "frame-0.png")
        
        for index in range(0, timer_size):
            self.set_pixel(0, index, *self.fade_color(Color.white.rgb, 0.1))
        self.generate_image(first_frame_path)
        frame_files.append(first_frame_path)

        # Generate transparent frames with one visible pixel
        for index in range(0, timer_size):
            transparent_frame = np.zeros((self.matrix_size, self.matrix_size, 4), dtype=np.uint8)  # RGBA fully transparent
            x, y = 0, index
            r, g, b = self.fade_color(Color.pink.rgb, 1)
            transparent_frame[y][x] = [r, g, b, 255]

            frame_filename = os.path.join(temp_dir, f"frame-{index + 1}.png")
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
        """Convert glucose value to Y coordinate on the matrix.
        
        Args:
            glucose: Glucose value in mg/dL
            
        Returns:
            int: Y coordinate (0 = top, matrix_size-1 = bottom)
        """
        DIGIT_HEIGHT = 5
        glucose = max(self.min_glucose, min(glucose, self.max_glucose))
        available_y_range = self.matrix_size - DIGIT_HEIGHT - 1
        normalized = (glucose - self.min_glucose) / (self.max_glucose - self.min_glucose)
        return int((1 - normalized) * available_y_range) + DIGIT_HEIGHT
    
    def y_coordinate_to_glucose(self, y: int) -> int:
        """Convert Y coordinate back to glucose value.
        
        Args:
            y: Y coordinate (0 = top, matrix_size-1 = bottom)
            
        Returns:
            int: Glucose value in mg/dL
        """
        DIGIT_HEIGHT = 5
        available_y_range = self.matrix_size - DIGIT_HEIGHT - 1
        normalized = 1 - (y - DIGIT_HEIGHT) / available_y_range
        glucose = normalized * (self.max_glucose - self.min_glucose) + self.min_glucose
        return int(glucose)

    def get_brightness_on_hour(self, timezone_str="America/Recife") -> float:
        """Determine brightness factor based on current time.
        
        Args:
            timezone_str: Timezone string for local time calculation
            
        Returns:
            float: Brightness factor (0.0-1.0)
        """
        DAY_START_HOUR = 6
        NIGHT_START_HOUR = 21

        local_tz = pytz.timezone(timezone_str)
        current_time = datetime.now(local_tz)
        current_hour = current_time.hour

        if NIGHT_START_HOUR <= current_hour or current_hour < DAY_START_HOUR:
            return self.night_brightness
        else:
            return 1.0

    def determine_color(self, glucose: float, entry_type=EntrieEnum) -> ColorType:
        """Determine color for glucose value based on target ranges.
        
        Args:
            glucose: Glucose value in mg/dL
            entry_type: Type of glucose entry (SGV or MBG)
            
        Returns:
            ColorType: RGB color representing glucose status
        """
        MARGIN = 10
        if entry_type == EntrieEnum.MBG:
            return Color.white.rgb

        if glucose < self.GLUCOSE_LOW - MARGIN:
            return self.interpolate_color(Color.red.rgb, Color.yellow.rgb, glucose, self.get_min_sgv(), self.GLUCOSE_LOW - MARGIN)
        if glucose > self.GLUCOSE_HIGH + MARGIN:
            return self.interpolate_color(Color.yellow.rgb, Color.red.rgb, glucose, self.GLUCOSE_HIGH + MARGIN, self.get_max_sgv())
        elif glucose <= self.GLUCOSE_LOW or glucose >= self.GLUCOSE_HIGH:
            return Color.yellow.rgb
        else:
            return Color.green.rgb

    # def fade_color(self, color: ColorType, percentil: float) -> ColorType:
    #     percentil = max(0.0, min(1.0, percentil))
        
    #     corrected_color = []

    #     # Smooth the boost more aggressively toward low percentils
    #     BASE = 0.8
    #     MAX_BOOST = 1.5
    #     EXPONENT = 8.0
    #     GREEN_GAIN = 0.8
    #     RED_GAIN = 1.0

    #     # Adjust green correction to reduce its intensity at lower values
    #     red_correction = BASE + (MAX_BOOST - BASE) * ((1 - percentil) ** EXPONENT * RED_GAIN)
    #     green_correction = BASE + (MAX_BOOST - BASE) * ((1 - percentil) ** (EXPONENT * GREEN_GAIN))
    #     blue_correction = 1.0

    #     correction_factors = (red_correction, green_correction, blue_correction)

    #     for idx, item in enumerate(color):
    #         corrected = round(item * percentil * correction_factors[idx])
    #         # Ensure the value is within 0-255 range
    #         corrected_color.append(max(0, min(255, corrected)))

    #     return ColorType(*corrected_color)
    
    
    def draw_glucose_intervals(self, fade_strength: float = 0.2) -> None:
        """Draw faded intervals between glucose entries to visualize trends.

        Args:
            fade_strength: Brightness factor for interval lines (0.0 - 1.0)
        """
        if not hasattr(self, "formmated_entries") or len(self.formmated_entries) < 2:
            return
                
        averaged_by_time = self.average_entries_by_time(self.formmated_entries)

        prev_x = None
        prev_y = None
        prev_g = None
        for minutes_index, g in enumerate(averaged_by_time):
            if g is None:
                continue

            x = self._time_index_to_x(minutes_index)
            y = self.glucose_to_y_coordinate(int(g))

            if prev_y is not None and prev_x is not None and prev_g is not None:
                # Draw faded interval between previous and current point using glucose interpolation for color
                self._draw_line(prev_x, prev_y, x, y, int(prev_g), int(g), fade_strength)

            prev_x, prev_y, prev_g = x, y, g

    def draw_box(self, x1: int, y1: int, x2: int, y2: int, color: ColorType) -> None:
        """Draw a rectangular box on the matrix, with infill.
        
        Args:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            color: RGB color of the box
        """
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                self.set_pixel(x, y, *color)
    
    def _draw_line(self, x1: int, y1: int, x2: int, y2: int, g1: int, g2: int, fade_strength: float):
        """Draw a line between two points using Bresenham's algorithm.
        
        Args:
            x1, y1: Start coordinates
            x2, y2: End coordinates
            g1, g2: Glucose values at the start and end points (for color)
        """
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy

        total_steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
        step_index = 0

        while True:
            if total_steps > 1:
                t = step_index / (total_steps - 1)
            else:
                t = 0.0
            g = g1 + (g2 - g1) * t
            color = self.determine_color(g)
            faded_color = self.fade_color(color, fade_strength)
            self.set_pixel(x1, y1, *faded_color)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                if x1 == x2:
                    break
                err += dy
                x1 += sx
            if e2 <= dx:
                if y1 == y2:
                    break
                err += dx
                y1 += sy
            step_index += 1

    def fade_color(self, color: ColorType, percentil: float) -> ColorType:
        """Apply brightness and color correction to a color.
        
        Args:
            color: Original RGB color
            percentil: Brightness factor (0.0 - 1.0)
            
        Returns:
            ColorType: Adjusted RGB color
        """
        percentil = max(0.1, percentil)
        
        r = int(color.r * percentil)
        g = int(color.g * percentil)
        b = int(color.b * percentil)

        return ColorType(r, g, b)
        

    def interpolate_color(self, low_color: ColorType, high_color: ColorType, value: float, min_value: int, max_value: int) -> ColorType:
        """Interpolate between two colors based on a value range.
        
        Args:
            low_color: RGB color for minimum value
            high_color: RGB color for maximum value
            value: Current value to interpolate
            min_value: Minimum value in range
            max_value: Maximum value in range
            
        Returns:
            ColorType: Interpolated RGB color
        """
        if value < min_value:
            value = min_value
        elif value > max_value:
            value = max_value

        interpolation_factor = (value - min_value) / (max_value - min_value)

        r = int(low_color[0] + interpolation_factor * (high_color[0] - low_color[0]))
        g = int(low_color[1] + interpolation_factor * (high_color[1] - low_color[1]))
        b = int(low_color[2] + interpolation_factor * (high_color[2] - low_color[2]))

        return ColorType(r, g, b)

    def get_glucose_difference_signal(self, glucose_difference: int) -> str:
        """Get the sign of glucose difference as string.
        
        Returns:
            str: '+' or '-'
        """
        return '-' if glucose_difference < 0 else '+'

    def get_max_sgv(self) -> int:
        """Find maximum glucose value in current entries.
        
        Returns:
            int: Maximum glucose value in mg/dL
        """
        return max(entry.glucose for entry in self.formmated_entries)

    def get_min_sgv(self) -> int:
        """Find minimum glucose value in current entries.
        
        Returns:
            int: Minimum glucose value in mg/dL
        """
        return min(entry.glucose for entry in self.formmated_entries)

    def is_five_apart(self, init: int, current: int) -> bool:
        """Check if current position is 5 pixels away from initial position.
        
        Args:
            init: Initial Y position
            current: Current Y position
            
        Returns:
            bool: True if positions are 5 apart
        """
        return (current - init + 1) % 5 == 0

    def draw_time_display(self, position="top-right", format_type="HH:MM", fade_strength=0.3):
        """Draw time on the matrix.
        
        Args:
            position: Where to place time
            format_type: Time format
            fade_strength: Brightness level for time display (0.0 to 1.0)
        """
        now = datetime.now()
        
        # Normalize common format tokens
        fmt = (format_type or "HH:MM").lower()

        if fmt in ("12h", "hh:mm a", "hh:mmam", "hh:mm pm"):
            hour = now.strftime("%I")
            minute = now.strftime("%M")
            time_str = f"{hour}:{minute}"
        elif fmt in ("24h", "hh:mm", "hh:mm:ss", "hhmm") or fmt == "hh:mm:ss":
            if fmt == "hh:mm:ss" or format_type == "HH:MM:SS":
                time_str = now.strftime("%H:%M:%S")
            elif fmt == "hhmm" or format_type == "compact":
                time_str = now.strftime("%H%M")
            else:
                time_str = now.strftime("%H:%M")
        elif format_type == "MM:SS" or fmt == "mm:ss":
            time_str = now.strftime("%M:%S")
        else:
            time_str = now.strftime("%H:%M")
        
        text_width = self._get_time_text_width(time_str)
        
        if position == "top-left":
            x = 0
            y = 3 + self.glucose_to_y_coordinate(self.GLUCOSE_HIGH)
        elif position == "top-right":
            x = self.matrix_size - text_width - 1
            y = 3 + self.glucose_to_y_coordinate(self.GLUCOSE_HIGH)
        elif position == "bottom-left":
            x = 0
            y = self.glucose_to_y_coordinate(self.GLUCOSE_LOW) - 4
        elif position == "bottom-right":
            x = self.matrix_size - text_width - 1
            y = self.matrix_size - 7
        elif position == "bottom":
            x = 2 
            y = self.matrix_size - 7
        else:  # default to top-left
            x = 1
            y = 1
        
        x = max(0, min(x, self.matrix_size - text_width))
        y = max(0, min(y, self.matrix_size - 6))
        
        color = self.fade_color(Color.white.rgb, fade_strength)
        self.draw_box(x - 1, y , x + text_width - 3, y + 4, Color.black.rgb)
        self._draw_time_text(time_str, x, y, color)

    def _draw_time_text(self, text, start_x, start_y, color):
        """Draw time text using time-specific patterns.
        
        Args:
            text: Text to draw
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            color: ColorType for the text
            
        Returns:
            int: Final X coordinate after drawing
        """
        time_patterns = get_digit_patterns()
        current_x = start_x
        
        for char in text:
            current_color = color
            spacing = 1  # Default spacing
            
            # Special handling for colon character
            if char == ':':
                # Make colon 50% more faded (reduce brightness by 50%)
                current_color = self.fade_color(color, 0.3)  # 50% more faded
                spacing = 0 
                current_x -= 1
            
            if char in time_patterns:
                pattern = time_patterns[char]
                # Draw the pattern pixel by pixel to ensure it's visible
                for row in range(pattern.shape[0]):
                    for col in range(pattern.shape[1]):
                        if pattern[row, col] == 1:
                            pixel_x = current_x + col
                            pixel_y = start_y + row
                            if (0 <= pixel_x < self.matrix_size and 
                                0 <= pixel_y < self.matrix_size):
                                self.set_pixel(pixel_x, pixel_y, current_color.r, current_color.g, current_color.b)
                current_x += pattern.shape[1] + spacing  # Use variable spacing
            else:
                # Fallback to regular digit patterns
                digit_patterns = get_digit_patterns()
                if char in digit_patterns:
                    pattern = digit_patterns[char]
                    for row in range(pattern.shape[0]):
                        for col in range(pattern.shape[1]):
                            if pattern[row, col] == 1:
                                pixel_x = current_x + col
                                pixel_y = start_y + row
                                if (0 <= pixel_x < self.matrix_size and 
                                    0 <= pixel_y < self.matrix_size):
                                    self.set_pixel(pixel_x, pixel_y, current_color.r, current_color.g, current_color.b)
                    current_x += pattern.shape[1] + spacing  # Use variable spacing

        return current_x

    def draw_corner_time(self, position="top-right", format_type="HH:MM"):
        """Draw time in matrix corner with automatic positioning.
        
        Args:
            position: "top-left", "top-right", "bottom-left", "bottom-right"
            compact: If True, use HHMM format; if False, use HH:MM
        """
        fade_strength = 1
        
        self.draw_time_display(position, format_type, fade_strength)

    def display_time_with_blink(self, x_offset=0, y_offset=0, fade_strength=0.4):
        """Display time with blinking separator (every second).
        
        Args:
            x_offset: X position offset for time display
            y_offset: Y position offset for time display
            fade_strength: Brightness level for time display
        """
        now = datetime.now()
        hours = now.strftime("%HH")
        minutes = now.strftime("%MM")
        
        color = self.fade_color(Color.cyan.rgb, fade_strength)
        time_patterns = get_digit_patterns()
        
        # Draw hours
        current_x = self._draw_time_text(hours, x_offset, y_offset, color)
        
        # Draw blinking colon (blink every second)
        if now.second % 2 == 0:
            if ':' in time_patterns:
                colon_pattern = time_patterns[':']
                self.draw_pattern(colon_pattern, current_x, y_offset, color)
    
        current_x += 2  # Space for colon
    
        # Draw minutes
        self._draw_time_text(minutes, current_x, y_offset, color)

    def _get_time_text_width(self, time_str):
        """Calculate the width needed for time text display.
        
        Args:
            time_str: Time string to measure
            
        Returns:
            int: Width in pixels needed for the text
        """
        try:
            time_patterns = get_digit_patterns()
        except:
            # Fallback to digit patterns if time patterns not available
            time_patterns = get_digit_patterns()
        
        total_width = 0
        
        for char in time_str:
            if char in time_patterns:
                pattern = time_patterns[char]
                total_width += pattern.shape[1] + 1  # Add 1 for spacing
            elif char in get_digit_patterns():
                # Fallback to digit patterns
                pattern = get_digit_patterns()[char]
                total_width += pattern.shape[1] + 1
        
        return max(1, total_width - 1)  # Remove last spacing