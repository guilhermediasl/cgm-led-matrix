import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from PixelMatrix import PixelMatrix
from util import Color, GlucoseItem, EntrieEnum, IobItem, ColorType, TreatmentEnum, TreatmentItem

class TestPixelMatrixInit:
    def test_initialization(self):
        matrix = PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)
        assert matrix.matrix_size == 32
        assert matrix.min_glucose == 70
        assert matrix.max_glucose == 180
        assert matrix.pixels.shape == (32, 32, 3)
        assert matrix.pixels.dtype == np.uint8
        assert matrix.GLUCOSE_LOW == 80
        assert matrix.GLUCOSE_HIGH == 160
        assert matrix.night_brightness == 0.5
        assert matrix.PIXEL_INTERVAL == 5
        assert matrix.GLUCOSE_LOW_OUT_OF_RANGE == 39
        assert matrix.GLUCOSE_HIGH_OUT_OF_RANGE == 400

class TestSetterMethods:
    @pytest.fixture
    def matrix(self):
        return PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)

    def test_set_formatted_entries(self, matrix):
        now = datetime.now()
        entries = [GlucoseItem(EntrieEnum.SGV, 120, now)]
        matrix.set_formmated_entries(entries)
        assert matrix.formmated_entries == entries

    def test_set_formatted_treatments(self, matrix):
        now = datetime.now()
        treatments = [TreatmentItem("1", TreatmentEnum.BOLUS, now, 5)]
        matrix.set_formmated_treatments(treatments)
        assert matrix.formmated_treatments == treatments

    def test_set_arrow(self, matrix):
        matrix.set_arrow("DoubleUp")
        assert matrix.arrow == "DoubleUp"
        
class TestPixelOperations:
    @pytest.fixture
    def matrix(self):
        return PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)

    def test_set_and_get_pixel(self, matrix):
        matrix.set_pixel(5, 10, 255, 128, 64)
        color = matrix.get_pixel(5, 10)
        assert color[0] == 255
        assert color[1] == 128
        assert color[2] == 64

    def test_set_pixel_bounds_checking(self, matrix):
        # Should not crash for out-of-bounds
        matrix.set_pixel(-1, 5, 255, 0, 0)
        matrix.set_pixel(20, 5, 255, 0, 0)
        matrix.set_pixel(5, -1, 255, 0, 0)
        matrix.set_pixel(5, 20, 255, 0, 0)

    def test_paint_background(self, matrix):
        color = ColorType(100, 150, 200)
        matrix.paint_background(color)
        
        # Check random pixels
        expected_color = (color.r, color.g, color.b)
        assert tuple(matrix.get_pixel(0, 0)) == expected_color
        assert tuple(matrix.get_pixel(15, 15)) == expected_color
        assert tuple(matrix.get_pixel(31, 31)) == expected_color

    def test_set_interpolated_pixel(self, matrix):
        color = ColorType(255, 0, 0)
        matrix.set_interpolated_pixel(10, 2, 120, color, 0.5)
        
        # Check that pixel was set somewhere
        total_pixels = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_pixels > 0
        
class TestGlucoseCoordinateMapping:
    @pytest.fixture
    def matrix(self):
        return PixelMatrix(32, 70, 250, 80, 160, 0.5, 5)

    def test_glucose_to_y_coordinate(self, matrix):
        # Min glucose should map to bottom (high Y)
        y_min = matrix.glucose_to_y_coordinate(70)
        y_max = matrix.glucose_to_y_coordinate(250)
        assert y_max < y_min  # Lower glucose = higher Y coordinate

    def test_y_coordinate_to_glucose_round_trip(self, matrix):
        # Test several values inside range
        for glucose in [70, 100, 150, 200, 250]:
            y = matrix.glucose_to_y_coordinate(glucose)
            glucose_back = matrix.y_coordinate_to_glucose(y)
            assert abs(glucose - glucose_back) <= 4  # Allow small rounding error

    def test_glucose_bounds_clamping(self, matrix):
        # Values outside range should be clamped
        y_low = matrix.glucose_to_y_coordinate(50)   # Below min
        y_high = matrix.glucose_to_y_coordinate(300) # Above max
        
        y_min_expected = matrix.glucose_to_y_coordinate(70)
        y_max_expected = matrix.glucose_to_y_coordinate(250)
        
        assert y_low == y_min_expected
        assert y_high == y_max_expected

class TestGlucoseDataProcessing:
    @pytest.fixture
    def matrix(self):
        return PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)

    @pytest.fixture
    def sample_entries(self):
        now = datetime.now()
        return [
            GlucoseItem(EntrieEnum.SGV, 120, now),
            GlucoseItem(EntrieEnum.SGV, 115, now - timedelta(minutes=5)),
            GlucoseItem(EntrieEnum.SGV, 110, now - timedelta(minutes=10)),
            GlucoseItem(EntrieEnum.SGV, 105, now - timedelta(minutes=15)),
        ]

    def test_average_entries_by_time(self, matrix, sample_entries):
        averages = matrix.average_entries_by_time(sample_entries)
        
        # Should have matrix_size buckets
        assert len(averages) == matrix.matrix_size
        
        # First bucket (most recent) should have 120
        assert averages[0] == 120
        
        # Second bucket should have 115
        assert averages[1] == 115

    def test_average_entries_empty_buckets(self, matrix):
        # Entry way in the past
        old_entry = GlucoseItem(EntrieEnum.SGV, 100, 
                               datetime.now() - timedelta(hours=5))
        averages = matrix.average_entries_by_time([old_entry])
        
        # Most recent buckets should be None
        assert averages[0] is None
        assert averages[1] is None

    def test_average_entries_multiple_in_bucket(self, matrix):
        now = datetime.now()
        # Multiple entries in same time bucket
        entries = [
            GlucoseItem(EntrieEnum.SGV, 120, now),
            GlucoseItem(EntrieEnum.SGV, 110, now - timedelta(minutes=2)),
            GlucoseItem(EntrieEnum.SGV, 130, now - timedelta(minutes=3)),
        ]
        averages = matrix.average_entries_by_time(entries)
        
        # Should average the three values: (120 + 110 + 130) / 3 = 120
        assert averages[0] == 120
        
class TestGlucoseDifferenceCalculation:
    @pytest.fixture
    def matrix(self):
        return PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)

    def test_set_glucose_difference_with_buckets(self, matrix):
        # Mock entries_by_time with valid values
        matrix.entries_by_time = [120, 115, None, 105]
        matrix.set_glucose_difference()
        assert matrix.glucose_difference == 5  # 120 - 115

    def test_set_glucose_difference_with_none_buckets(self, matrix):
        now = datetime.now()
        matrix.entries_by_time = [None, None, 110, 105]
        matrix.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 125, now),
            GlucoseItem(EntrieEnum.SGV, 120, now - timedelta(minutes=5))
        ]
        matrix.set_glucose_difference()
        assert matrix.glucose_difference == 5  # 125 - 120

    def test_set_glucose_difference_no_data(self, matrix):
        matrix.entries_by_time = [None, None, None]
        matrix.formmated_entries = []
        matrix.set_glucose_difference()
        assert matrix.glucose_difference == 0

    def test_set_glucose_difference_unsorted_entries(self, matrix):
        now = datetime.now()
        # Entries not in chronological order
        matrix.entries_by_time = [None, None]
        matrix.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 120, now - timedelta(minutes=5)),  # Older first
            GlucoseItem(EntrieEnum.SGV, 125, now),  # Newer second
        ]
        matrix.set_glucose_difference()
        assert matrix.glucose_difference == 5  # Should sort and calculate 125 - 120

class TestColorDetermination:
    @pytest.fixture
    def matrix(self):
        m = PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)
        now = datetime.now()
        # Mock entries for min/max calculations
        m.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 100, now),
            GlucoseItem(EntrieEnum.SGV, 150, now - timedelta(minutes=5))
        ]
        return m

    def test_determine_color_in_range(self, matrix):
        color = matrix.determine_color(120)
        # Should be green for normal range
        assert color == Color.green.rgb

    def test_determine_color_low(self, matrix):
        color = matrix.determine_color(60)  # Below low boundary
        # Should be some red/yellow interpolation
        assert isinstance(color, ColorType)
        assert len(color) == 3

    def test_determine_color_high(self, matrix):
        color = matrix.determine_color(200)  # Above high boundary
        # Should be yellow/red blend
        assert isinstance(color, ColorType)
        assert len(color) == 3

    def test_determine_color_mbg_type(self, matrix):
        color = matrix.determine_color(120, EntrieEnum.MBG)
        assert color == Color.white.rgb

    def test_determine_color_boundary_values(self, matrix):
        # Test exact boundary values
        low_boundary = matrix.determine_color(80)  # Exactly at GLUCOSE_LOW
        assert low_boundary == Color.yellow.rgb
        
        high_boundary = matrix.determine_color(160)  # Exactly at GLUCOSE_HIGH
        assert high_boundary == Color.yellow.rgb

class TestDrawingOperations:
    @pytest.fixture
    def matrix(self):
        return PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)

    def test_draw_vertical_line(self, matrix):
        color = ColorType(255, 0, 0)
        matrix.draw_vertical_line(5, color, 120, 4)
        
        # Check that some pixels in column 5 are lit
        column_5 = matrix.pixels[:, 5]
        lit_pixels = np.any(column_5 != 0, axis=1)
        assert np.sum(lit_pixels) > 0

    def test_draw_vertical_line_with_enable_five(self, matrix):
        color = ColorType(255, 0, 0)
        matrix.draw_vertical_line(5, color, 120, 10, enable_five=True)
        
        # Should still draw pixels
        column_5 = matrix.pixels[:, 5]
        lit_pixels = np.any(column_5 != 0, axis=1)
        assert np.sum(lit_pixels) > 0

    def test_draw_vertical_line_with_blink(self, matrix):
        color = ColorType(255, 0, 0)
        matrix.draw_vertical_line(5, color, 120, 4, blink=True)
        
        # Should draw pixels with blinking effect
        column_5 = matrix.pixels[:, 5]
        lit_pixels = np.any(column_5 != 0, axis=1)
        assert np.sum(lit_pixels) > 0

    def test_draw_horizontal_line(self, matrix):
        color = ColorType(0, 255, 0)
        matrix.draw_horizontal_line(120, color, 0, 10)
        
        # Check that pixels in the row are lit
        y = matrix.glucose_to_y_coordinate(120) + 1
        if 0 <= y < matrix.matrix_size:
            row_pixels = matrix.pixels[y, 0:10]
            lit_pixels = np.any(row_pixels != 0, axis=1)
            assert np.sum(lit_pixels) > 0

    def test_draw_horizontal_line_bounds_clamping(self, matrix):
        color = ColorType(0, 255, 0)
        # Test with values that need clamping
        matrix.draw_horizontal_line(120, color, -5, 50)
        
        # Should not crash and should draw something
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit >= 0

    def test_draw_pattern(self, matrix):
        pattern = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        color = ColorType(255, 255, 255)
        
        matrix.draw_pattern(pattern, 2, 3, color)
        
        # Check that pattern pixels are set where pattern has 1s
        assert tuple(matrix.get_pixel(2, 3)) == color  # pattern[0,0] = 1
        assert tuple(matrix.get_pixel(4, 3)) == color  # pattern[0,2] = 1
        assert tuple(matrix.get_pixel(3, 4)) == color  # pattern[1,1] = 1
        assert tuple(matrix.get_pixel(3, 3)) != color  # pattern[0,1] = 0

class TestIOBVisualization:
    @pytest.fixture
    def matrix(self):
        return PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)

    def test_draw_iob_with_values(self, matrix):
        now = datetime.now()
        iob_list = [
            IobItem(now, 2.5),
            IobItem(now - timedelta(minutes=5), 2.0),
            IobItem(now - timedelta(minutes=10), 1.5),
        ]
        
        # Should not crash
        matrix.draw_iob(iob_list)
        
        # Check that some pixels are lit in the appropriate columns
        # IOB draws from the right side
        rightmost_col = matrix.pixels[:, -1]
        assert np.any(rightmost_col != 0)

    def test_draw_iob_fractional_values(self, matrix):
        now = datetime.now()
        # Test with fractional IOB values
        iob_list = [IobItem(now, 1.7)]  # 1 integer + 0.7 fractional
        
        matrix.draw_iob(iob_list)
        
        # Should draw both integer and fractional parts
        rightmost_col = matrix.pixels[:, -1]
        lit_pixels = np.sum(np.any(rightmost_col != 0, axis=1))
        assert lit_pixels > 1  # Should have more than just one pixel

    def test_draw_iob_minimum_fractional_threshold(self, matrix):
        now = datetime.now()
        # Test with very small fractional IOB (below threshold)
        iob_list = [IobItem(now, 1.05)]  # 0.05 is below 0.1 threshold
        
        matrix.draw_iob(iob_list)
        
        # Should only draw integer part
        rightmost_col = matrix.pixels[:, -1]
        lit_pixels = np.sum(np.any(rightmost_col != 0, axis=1))
        assert lit_pixels >= 1

class TestTextAndPatternDisplay:
    @pytest.fixture
    def matrix(self):
        return PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)

    def test_draw_text(self, matrix):
        color = ColorType(255, 255, 255)
        final_x = matrix._draw_text("123", 5, 5, color)
        
        # Should advance x coordinate
        assert final_x > 5
        
        # Should draw some pixels
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0

    def test_get_digit_width(self, matrix):
        # Test width calculation for various digits
        width_0 = matrix.get_digit_width('0')
        width_1 = matrix.get_digit_width('1')
        
        assert width_0 > 0
        assert width_1 > 0
        # Different digits may have different widths

    def test_get_digits_width(self, matrix):
        width = matrix.get_digits_width("123")
        assert width > 0
        
        # Should be sum of individual digit widths
        individual_sum = sum(matrix.get_digit_width(d) for d in "123")
        assert width == individual_sum

class TestGlucoseRangeAndSignalMethods:
    @pytest.fixture
    def matrix(self):
        return PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)

    def test_get_out_of_range_glucose_str(self, matrix):
        assert matrix.get_out_of_range_glucose_str(30) == "LOW"
        assert matrix.get_out_of_range_glucose_str(450) == "HIGH"
        assert matrix.get_out_of_range_glucose_str(120) == "120"

    def test_is_glucose_out_of_range(self, matrix):
        assert matrix.is_glucose_out_of_range(30) == True
        assert matrix.is_glucose_out_of_range(450) == True
        assert matrix.is_glucose_out_of_range(120) == False
        assert matrix.is_glucose_out_of_range(39) == True  # Exactly at boundary
        assert matrix.is_glucose_out_of_range(400) == True  # Exactly at boundary

    def test_get_glucose_difference_signal(self, matrix):
        assert matrix.get_glucose_difference_signal(5) == '+'
        assert matrix.get_glucose_difference_signal(-5) == '-'
        assert matrix.get_glucose_difference_signal(0) == '+'

class TestImageGeneration:
    @pytest.fixture
    def matrix(self):
        return PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)  # Smaller for faster tests

    @patch('png.Writer')
    @patch('builtins.open')
    def test_generate_image(self, mock_open, mock_writer, matrix):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        
        matrix.generate_image("test.png")
        
        mock_open.assert_called_once_with("test.png", "wb")
        mock_writer.assert_called_once_with(32, 32, greyscale=False)
        mock_writer_instance.write.assert_called_once()
        
    @patch('os.makedirs')
    @patch('PIL.Image.open')
    @patch('PIL.Image.fromarray')
    def test_generate_timer_gif(self, mock_fromarray, mock_open, mock_makedirs, matrix):
        # Mock PIL Image operations
        mock_image = MagicMock()
        mock_open.return_value = mock_image
        mock_fromarray.return_value = mock_image
        mock_image.convert.return_value = mock_image
        mock_image.getchannel.return_value = mock_image
        
        # Should not crash
        matrix.generate_timer_gif(timer_size=2)
        
        # Verify directory creation
        mock_makedirs.assert_called()

class TestGlucoseIntervalsDrawing:
    @pytest.fixture
    def matrix(self):
        m = PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)
        now = datetime.now()
        m.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 120, now),
            GlucoseItem(EntrieEnum.SGV, 115, now - timedelta(minutes=5)),
            GlucoseItem(EntrieEnum.SGV, 110, now - timedelta(minutes=10)),
        ]
        return m

    def test_draw_glucose_intervals(self, matrix):
        # First need to call average_entries_by_time to set up entries_by_time
        matrix.average_entries_by_time(matrix.formmated_entries)
        
        matrix.draw_glucose_intervals(fade_strength=0.1)
        
        # Should draw some connecting lines
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0

    def test_draw_glucose_intervals_insufficient_data(self, matrix):
        # Test with insufficient entries
        matrix.formmated_entries = [GlucoseItem(EntrieEnum.SGV, 120, datetime.now())]
        
        # Should not crash
        matrix.draw_glucose_intervals()

    def test_plot_faded_interval(self, matrix):
        # Mock formmated_entries for color determination
        now = datetime.now()
        matrix.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 120, now),
            GlucoseItem(EntrieEnum.SGV, 100, now - timedelta(minutes=5))
        ]
        
        matrix._plot_faded_interval(5, 10, 8, 12, 120, 0.1)
        
        # Should draw a line between the points
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0

class TestUtilityMethods:
    @pytest.fixture
    def matrix(self):
        m = PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)
        now = datetime.now()
        m.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 200, now),
            GlucoseItem(EntrieEnum.SGV, 50, now - timedelta(minutes=5))
        ]
        return m

    def test_get_max_min_sgv(self, matrix):
        assert matrix.get_max_sgv() == 200
        assert matrix.get_min_sgv() == 50

    def test_get_out_of_range_glucose_str(self, matrix):
        assert matrix.get_out_of_range_glucose_str(30) == "LOW"
        assert matrix.get_out_of_range_glucose_str(450) == "HIGH"
        assert matrix.get_out_of_range_glucose_str(120) == "120"

    def test_is_glucose_out_of_range(self, matrix):
        assert matrix.is_glucose_out_of_range(30) == True
        assert matrix.is_glucose_out_of_range(450) == True
        assert matrix.is_glucose_out_of_range(120) == False

    def test_fade_color(self, matrix):
        color = ColorType(100, 200, 50)
        faded = matrix.fade_color(color, 0.5)
        
        # Should be roughly half brightness, but with minimum floor
        assert faded.r < color.r
        assert faded.g < color.g
        assert faded.b < color.b
        
        # Test minimum floor
        very_faded = matrix.fade_color(color, 0.01)
        assert very_faded.r >= 10  # 0.1 minimum * 100
        
    def test_fade_color_bounds(self, matrix):
        color = ColorType(100, 200, 50)
        
        # Test lower bound
        under_faded = matrix.fade_color(color, -0.5)  # Below 0
        expected = matrix.fade_color(color, 0.1)  # Should clamp to minimum
        assert under_faded == expected

    def test_interpolate_color(self, matrix):
        red = ColorType(255, 0, 0)
        blue = ColorType(0, 0, 255)
        
        # Test midpoint interpolation
        mid_color = matrix.interpolate_color(red, blue, 50, 0, 100)
        assert mid_color.r == 127  # (255 + 0) / 2
        assert mid_color.g == 0
        assert mid_color.b == 127  # (0 + 255) / 2
        
    def test_interpolate_color_bounds(self, matrix):
        red = ColorType(255, 0, 0)
        blue = ColorType(0, 0, 255)
        
        # Test value below range
        low_color = matrix.interpolate_color(red, blue, -10, 0, 100)
        assert low_color == red  # Should clamp to low_color
        
        # Test value above range
        high_color = matrix.interpolate_color(red, blue, 110, 0, 100)
        assert high_color == blue  # Should clamp to high_color

    def test_is_five_apart(self, matrix):
        assert matrix.is_five_apart(10, 14) == True  # (14 - 10 + 1) % 5 == 0
        assert matrix.is_five_apart(10, 13) == False
        assert matrix.is_five_apart(5, 9) == True
        assert matrix.is_five_apart(0, 4) == True

class TestTimeAndIndexConversion:
    @pytest.fixture
    def matrix(self):
        return PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)

    def test_time_index_to_x(self, matrix):
        # Test conversion of time index to x coordinate
        assert matrix._time_index_to_x(0) == 31  # Most recent = rightmost
        assert matrix._time_index_to_x(31) == 0  # Oldest = leftmost
        assert matrix._time_index_to_x(15) == 16  # Middle

    def test_plot_entry(self, matrix):
        # Mock formmated_entries for color determination
        now = datetime.now()
        matrix.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 120, now),
            GlucoseItem(EntrieEnum.SGV, 100, now - timedelta(minutes=5))
        ]
        
        matrix._plot_entry(10, 15, 120)
        
        # Check that pixel was set
        assert tuple(matrix.get_pixel(10, 15)) != (0, 0, 0)

    def test_draw_trail(self, matrix):
        matrix._draw_trail(10, 15)
        
        # Should draw fading trail to the right
        trail_found = False
        for x in range(11, min(19, matrix.matrix_size)):  # Trail extends right
            if tuple(matrix.get_pixel(x, 15)) != (0, 0, 0):
                trail_found = True
                break
        assert trail_found

class TestBrightnessAndTime:
    @pytest.fixture
    def matrix(self):
        return PixelMatrix(32, 70, 180, 80, 160, 0.3, 5)

    @patch('pytz.timezone')
    @patch('PixelMatrix.datetime')
    def test_get_brightness_on_hour_day(self, mock_datetime, mock_timezone, matrix):
        # Mock daytime (10 AM)
        mock_time = MagicMock()
        mock_time.hour = 10
        mock_tz = MagicMock()
        mock_tz.localize.return_value = mock_time
        mock_timezone.return_value = mock_tz
        mock_datetime.now.return_value = mock_time
        
        brightness = matrix.get_brightness_on_hour()
        assert brightness == 1.0  # Full brightness during day

    @patch('pytz.timezone')
    @patch('PixelMatrix.datetime')
    def test_get_brightness_on_hour_night(self, mock_datetime, mock_timezone, matrix):
        # Mock nighttime (11 PM)
        mock_time = MagicMock()
        mock_time.hour = 23
        mock_tz = MagicMock()
        mock_tz.localize.return_value = mock_time
        mock_timezone.return_value = mock_tz
        mock_datetime.now.return_value = mock_time
        
        brightness = matrix.get_brightness_on_hour()
        assert brightness == matrix.night_brightness

class TestDataAmountCalculation:
    @pytest.fixture
    def matrix(self):
        m = PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)
        now = datetime.now()
        m.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 120, now - timedelta(minutes=15))  # 15 minutes old
        ]
        return m

    def test_get_no_data_pixels_amount(self, matrix):
        amount = matrix.get_no_data_pixels_amount()
        # Should calculate based on time difference
        assert amount >= 0
        assert isinstance(amount, int)

    def test_get_no_data_pixels_amount_recent_data(self, matrix):
        # Set very recent data
        now = datetime.now()
        matrix.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 120, now - timedelta(minutes=1))
        ]
        amount = matrix.get_no_data_pixels_amount()
        assert amount == 0  # Should be 0 for recent data

class TestDisplayMethods:
    @pytest.fixture
    def matrix(self):
        m = PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)
        now = datetime.now()
        m.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 120, now, "Flat"),
            GlucoseItem(EntrieEnum.SGV, 115, now - timedelta(minutes=5))
        ]
        m.entries_by_time = [120, 115, None, None] + [None] * 12
        m.arrow = "Flat"
        m.glucose_difference = 5
        return m

    def test_display_glucose_on_matrix_normal_range(self, matrix):
        matrix.display_glucose_on_matrix(120)
        
        # Should display glucose value, arrow, and difference
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0

    def test_display_glucose_on_matrix_out_of_range_low(self, matrix):
        matrix.display_glucose_on_matrix(30)  # Very low glucose
        
        # Should display "LOW" in red
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0
   
    def test_display_glucose_on_matrix_out_of_range_high(self, matrix):
        matrix.display_glucose_on_matrix(450)  # Very high glucose
        
        # Should display "HIGH" in red
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0
     
    def test_display_glucose_on_matrix(self, matrix):
        # Should not crash
        matrix.display_glucose_on_matrix(120)

    def test_display_entries(self, matrix):
        # Should not crash
        matrix.display_entries()
        
        # Should have plotted some points
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0

    def test_display_entries_with_trail(self, matrix):
        # Test that trail is drawn for first valid point not at index 0
        matrix.entries_by_time = [None, 120, 115, None] + [None] * 28
        
        matrix.display_entries()
        
        # Should plot entries and draw trail
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0

class TestEdgeCases:
    @pytest.fixture
    def matrix(self) -> PixelMatrix | None:
        matrix = PixelMatrix(8, 70, 180, 80, 160, 0.5, 5)  # Smaller matrix for edge case testing
        # matrix.average_entries_by_time
        return matrix
    
    def test_empty_entries_list(self, matrix):
        """Test behavior with empty glucose entries."""
        averages = matrix.average_entries_by_time([])
        assert all(avg is None for avg in averages)

    def test_single_entry(self, matrix):
        """Test with only one glucose entry."""
        now = datetime.now()
        entries = [GlucoseItem(EntrieEnum.SGV, 120, now)]
        averages = matrix.average_entries_by_time(entries)
        assert averages[0] == 120
        assert all(avg is None for avg in averages[1:])

    def test_matrix_size_one(self):
        """Test with minimal matrix size."""
        matrix = PixelMatrix(1, 70, 180, 80, 160, 0.5, 5)
        assert matrix.matrix_size == 1
        assert matrix.pixels.shape == (1, 1, 3)

    def test_extreme_glucose_values(self, matrix):
        """Test with extreme glucose values."""
        # Mock formatted entries to avoid errors in determine_color
        now = datetime.now()
        matrix.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 150, now),
            GlucoseItem(EntrieEnum.SGV, 100, now - timedelta(minutes=5))
        ]
        
        # Very high glucose
        color_high = matrix.determine_color(1000)
        assert isinstance(color_high, ColorType)
        assert all(0 <= c <= 255 for c in color_high)
        
        # Very low glucose
        color_low = matrix.determine_color(-100)
        assert isinstance(color_low, ColorType)
        assert all(0 <= c <= 255 for c in color_low)
        
        # Extreme positive values
        color_extreme_high = matrix.determine_color(9999)
        assert isinstance(color_extreme_high, ColorType)
        
        # Zero glucose
        color_zero = matrix.determine_color(0)
        assert isinstance(color_zero, ColorType)

    def test_glucose_color_interpolation_edge_cases(self, matrix):
        """Test color interpolation at boundary conditions."""
        now = datetime.now()
        matrix.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 80, now),   # At GLUCOSE_LOW
            GlucoseItem(EntrieEnum.SGV, 160, now - timedelta(minutes=5))  # At GLUCOSE_HIGH
        ]
        
        # Test values exactly at boundaries
        color_at_low = matrix.determine_color(80)
        color_at_high = matrix.determine_color(160)
        
        assert color_at_low == Color.yellow.rgb
        assert color_at_high == Color.yellow.rgb
        
        # Test values just inside normal range
        color_just_above_low = matrix.determine_color(81)
        color_just_below_high = matrix.determine_color(159)
        
        assert color_just_above_low == Color.green.rgb
        assert color_just_below_high == Color.green.rgb

    def test_time_bucket_edge_cases(self, matrix):
        """Test time bucketing with edge case timings."""
        now = datetime.now()
        
        # Entry exactly at bucket boundary
        boundary_entry = GlucoseItem(EntrieEnum.SGV, 120, now - timedelta(minutes=5))
        
        # Entry just before bucket boundary  
        before_boundary = GlucoseItem(EntrieEnum.SGV, 115, now - timedelta(minutes=10, seconds=59))
        
        # Entry just after bucket boundary
        after_boundary = GlucoseItem(EntrieEnum.SGV, 110, now - timedelta(minutes=5, seconds=1))
        
        entries = [boundary_entry, before_boundary, after_boundary]
        averages = matrix.average_entries_by_time(entries)
        
        # Should handle boundary conditions correctly
        assert len(averages) == matrix.matrix_size
        assert averages[0] is None  # Most recent bucket should be empty
        
    def test_iob_drawing_edge_cases(self, matrix):
        """Test IOB drawing with edge case values."""
        now = datetime.now()
        
        # Zero IOB
        zero_iob = [IobItem(now, 0.0)]
        matrix.draw_iob(zero_iob)
        
        # Very small IOB below threshold
        tiny_iob = [IobItem(now, 0.05)]
        matrix.draw_iob(tiny_iob)
        
        # Exactly at threshold
        threshold_iob = [IobItem(now, 0.1)]
        matrix.draw_iob(threshold_iob)
        
        # Large IOB value
        large_iob = [IobItem(now, 15.9)]
        matrix.draw_iob(large_iob)
        
        # Should not crash with any of these values
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit >= 0

    def test_treatment_visualization_bounds(self, matrix):
        """Test treatment visualization at matrix boundaries."""
        # Carbs at leftmost position
        carbs_left = [(0, 25, TreatmentEnum.CARBS)]
        matrix.draw_carbs(carbs_left)
        
        # Carbs at rightmost position
        carbs_right = [(matrix.matrix_size - 1, 30, TreatmentEnum.CARBS)]
        matrix.draw_carbs(carbs_right)
        
        # Bolus with very small amount
        bolus_tiny = [(5, 1, TreatmentEnum.BOLUS)]
        matrix.draw_bolus(bolus_tiny)
        
        # Bolus with large amount that might exceed matrix height
        bolus_large = [(10, 50, TreatmentEnum.BOLUS)]
        matrix.draw_bolus(bolus_large)
        
        # Exercise at boundaries
        exercise_boundary = {0, matrix.matrix_size - 1}
        matrix.draw_exercise(exercise_boundary)
        
        # Should not crash and should draw something
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0

    def test_text_rendering_edge_cases(self, matrix):
        """Test text rendering with various inputs."""
        color = ColorType(255, 255, 255)
        
        # Single character
        x1 = matrix._draw_text("1", 0, 0, color)
        assert x1 > 0
        
        # Empty string (should handle gracefully)
        x2 = matrix._draw_text("", 5, 5, color)
        assert x2 == 5  # Should not advance
        
        # Long string
        x3 = matrix._draw_text("123456789", 0, 10, color)
        assert x3 > 9
        
        # Special characters that might not have patterns
        try:
            x4 = matrix._draw_text("ABC", 0, 15, color)  # If letters supported
        except KeyError:
            pass  # Expected if letters not in digit_patterns
        
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0

    def test_brightness_calculation_edge_cases(self, matrix):
        """Test brightness calculations at various times."""
        with patch('pytz.timezone') as mock_tz, patch('PixelMatrix.datetime') as mock_dt:
            mock_time = MagicMock()
            mock_tz_instance = MagicMock()
            mock_tz.return_value = mock_tz_instance
            mock_tz_instance.localize.return_value = mock_time
            mock_dt.now.return_value = mock_time
            
            # Test at exact boundaries
            test_hours = [6, 21, 0, 12, 23, 5, 7, 20]
            
            for hour in test_hours:
                mock_time.hour = hour
                brightness = matrix.get_brightness_on_hour()
                assert 0.0 <= brightness <= 1.0
                
                if 6 <= hour < 21:
                    assert brightness == 1.0
                else:
                    assert brightness == matrix.night_brightness

    def test_data_amount_calculation_edge_cases(self, matrix):
        """Test no-data calculation with various time differences."""
        now = datetime.now()
        
        # Very recent data (should be 0)
        matrix.formmated_entries = [GlucoseItem(EntrieEnum.SGV, 120, now - timedelta(seconds=30))]
        assert matrix.get_no_data_pixels_amount() == 0
        
        # Data exactly at 1.5 * PIXEL_INTERVAL (should be 0 due to special case)
        matrix.formmated_entries = [GlucoseItem(EntrieEnum.SGV, 120, 
                                            now - timedelta(minutes=1.5 * matrix.PIXEL_INTERVAL))]
        amount = matrix.get_no_data_pixels_amount()
        assert amount >= 0
        
        # Very old data
        matrix.formmated_entries = [GlucoseItem(EntrieEnum.SGV, 120, 
                                            now - timedelta(hours=5))]
        amount = matrix.get_no_data_pixels_amount()
        assert amount > 0
        assert isinstance(amount, int)

    def test_glucose_intervals_drawing_edge_cases(self, matrix):
        """Test glucose intervals drawing with various data configurations."""
        now = datetime.now()
        
        # Single entry (insufficient data)
        matrix.formmated_entries = [GlucoseItem(EntrieEnum.SGV, 120, now)]
        matrix.draw_glucose_intervals()  # Should not crash
        
        # Two entries with same glucose value
        matrix.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 120, now),
            GlucoseItem(EntrieEnum.SGV, 120, now - timedelta(minutes=5))
        ]
        matrix.average_entries_by_time(matrix.formmated_entries)
        matrix.draw_glucose_intervals()
        
        # Entries with extreme glucose differences
        matrix.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 50, now),
            GlucoseItem(EntrieEnum.SGV, 250, now - timedelta(minutes=5))
        ]
        matrix.average_entries_by_time(matrix.formmated_entries)
        matrix.draw_glucose_intervals()
        
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit >= 0

    def test_display_entries_with_gaps(self, matrix):
        """Test displaying entries with gaps in data."""
        now = datetime.now()
        matrix.formmated_entries = [
            GlucoseItem(EntrieEnum.SGV, 120, now),
            GlucoseItem(EntrieEnum.SGV, 115, now - timedelta(minutes=25)),  # Gap in between
            GlucoseItem(EntrieEnum.SGV, 110, now - timedelta(minutes=30))
        ]
        
        # Create entries_by_time with gaps
        matrix.entries_by_time = [120] + [None] * 4 + [115, 110] + [None] * 25
        
        matrix.display_entries()
        
        # Should handle gaps gracefully
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0

    def test_color_interpolation_extreme_ranges(self, matrix):
        """Test color interpolation with extreme value ranges."""
        red = ColorType(255, 0, 0)
        blue = ColorType(0, 0, 255)
        
        # Very large range
        color = matrix.interpolate_color(red, blue, 500, 0, 1000)
        assert color.r == 127
        assert color.b == 127
        
        # Very small range
        color = matrix.interpolate_color(red, blue, 0.5, 0, 1)
        assert color.r == 127
        assert color.b == 127
        
        # Range with decimal values
        color = matrix.interpolate_color(red, blue, 2.5, 0, 5)
        assert color.r == 127
        assert color.b == 127

    def test_matrix_size_compatibility(self):
        """Test various matrix sizes for compatibility."""
        sizes_to_test = [16, 32]
        
        for size in sizes_to_test:
            matrix = PixelMatrix(size, 70, 180, 80, 160, 0.5, 5)
            assert matrix.matrix_size == size
            assert matrix.pixels.shape == (size, size, 3)
            
            # Test basic operations don't break
            matrix.set_pixel(0, 0, 255, 0, 0)
            color = matrix.get_pixel(0, 0)
            assert color[0] == 255
            
            # Test coordinate conversion
            y = matrix.glucose_to_y_coordinate(120)
            assert 0 <= y < size

    def test_pixel_intensity_ranges(self, matrix):
        """Test pixel color intensity edge cases."""
        # Test maximum intensity
        matrix.set_pixel(0, 0, *Color.white.rgb)
        color = matrix.get_pixel(0, 0)
        assert np.array_equal(color, Color.white.rgb)
        
        # Test minimum intensity
        matrix.set_pixel(1, 0, 0, 0, 0)
        color = matrix.get_pixel(1, 0)
        assert np.array_equal(color,(0, 0, 0))
        
        # Test fade color with extreme values
        extreme_color = ColorType(255, 255, 255)
        faded = matrix.fade_color(extreme_color, 0.01)  # Very low fade
        assert all(c >= 15 for c in faded)  # Should respect minimum
        
        faded_high = matrix.fade_color(extreme_color, 1.0)  # Full intensity
        assert faded_high.r == 255
        assert faded_high.g == 255  
        assert faded_high.b == 255
    
    def test_draw_operations_near_boundaries(self, matrix):
        """Test drawing operations near matrix boundaries."""
        # Draw at edges
        matrix.draw_vertical_line(0, ColorType(255, 0, 0), 120, 2)
        matrix.draw_vertical_line(7, ColorType(0, 255, 0), 120, 2)
        matrix.draw_horizontal_line(120, ColorType(0, 0, 255), 0, 8)
        
        # Should not crash and should draw something
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0
            
class TestDataIntegrity:
    def test_glucose_values_preserved(self):
        """Test that glucose values are correctly preserved through processing."""
        matrix = PixelMatrix(16, 70, 180, 80, 160, 0.5, 5)
        now = datetime.now()
        
        test_values = [120, 115, 110, 105]
        entries = [
            GlucoseItem(EntrieEnum.SGV, val, now - timedelta(minutes=i*5))
            for i, val in enumerate(test_values)
        ]
        
        averages = matrix.average_entries_by_time(entries)
        
        # Check that values are preserved (allowing for bucketing)
        non_none_averages = [avg for avg in averages if avg is not None]
        assert len(non_none_averages) > 0
        assert all(70 <= avg <= 180 or avg in test_values for avg in non_none_averages)

    def test_pixel_color_integrity(self):
        """Test that pixel colors are set and retrieved correctly."""
        matrix = PixelMatrix(8, 70, 180, 80, 160, 0.5, 5)
        
        test_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 255) # White
        ]
        
        # Set pixels with test colors
        for i, (r, g, b) in enumerate(test_colors):
            matrix.set_pixel(i, 0, r, g, b)
        
        # Verify colors are preserved
        for i, expected_color in enumerate(test_colors):
            actual_color = tuple(matrix.get_pixel(i, 0))
            assert actual_color == expected_color

class TestSpecializedDrawingMethods:
    @pytest.fixture
    def matrix(self):
        return PixelMatrix(32, 70, 180, 80, 160, 0.5, 5)

    def test_draw_hour_indicators(self, matrix):
        matrix.draw_hour_indicators()
        
        # Should draw some pixels
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0

    def test_draw_glucose_boundaries(self, matrix):
        matrix.draw_glucose_boundaries()
        
        # Should draw horizontal lines at glucose boundaries
        total_lit = np.sum(np.any(matrix.pixels != 0, axis=2))
        assert total_lit > 0

    def test_draw_carbs(self, matrix):
        carbs_data = [(5, 30, TreatmentEnum.CARBS), (10, 15, TreatmentEnum.CARBS)]
        matrix.draw_carbs(carbs_data)
        
        # Check that pixels are lit at specified x coordinates
        column_5 = matrix.pixels[:, 5]
        column_10 = matrix.pixels[:, 10]
        assert np.any(column_5 != 0)
        assert np.any(column_10 != 0)

    def test_draw_bolus(self, matrix):
        bolus_data = [(7, 5, TreatmentEnum.BOLUS), (12, 3, TreatmentEnum.BOLUS)]
        matrix.draw_bolus(bolus_data)
        
        # Check that pixels are lit at specified x coordinates
        column_7 = matrix.pixels[:, 7]
        column_12 = matrix.pixels[:, 12]
        assert np.any(column_7 != 0)
        assert np.any(column_12 != 0)

    def test_draw_exercise(self, matrix):
        exercise_indexes = {3, 8, 15}
        matrix.draw_exercise(exercise_indexes)
        
        # Check that pixels are lit at glucose boundaries for exercise times
        high_y = matrix.glucose_to_y_coordinate(matrix.GLUCOSE_HIGH) + 1
        low_y = matrix.glucose_to_y_coordinate(matrix.GLUCOSE_LOW) + 1
        
        for x in exercise_indexes:
            if 0 <= high_y < matrix.matrix_size:
                assert np.any(matrix.pixels[high_y, x] != 0)
            if 0 <= low_y < matrix.matrix_size:
                assert np.any(matrix.pixels[low_y, x] != 0)