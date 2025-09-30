import pytest
from datetime import datetime

from util import (
    TreatmentEnum, EntrieEnum, ColorType, Color,
    GlucoseItem, TreatmentItem, ExerciseItem, IobItem
)

class TestEnums:
    def test_treatment_enum_values(self):
        assert TreatmentEnum.BOLUS.value == 'Bolus'
        assert TreatmentEnum.CARBS.value == 'Carbs'
        assert TreatmentEnum.EXERCISE.value == 'Exercise'

    def test_entrie_enum_values(self):
        assert EntrieEnum.SGV == 'sgv'
        assert EntrieEnum.MBG == 'mbg'

class TestColorType:
    def test_color_type_creation(self):
        color = ColorType(255, 128, 64)
        assert color.r == 255
        assert color.g == 128
        assert color.b == 64

class TestColor:
    def test_color_rgb_property(self):
        red = Color.red
        assert red.rgb.r == 255
        assert red.rgb.g == 20
        assert red.rgb.b == 10

    def test_all_colors_have_rgb(self):
        for color in Color:
            rgb = color.rgb
            assert isinstance(rgb, ColorType)
            assert 0 <= rgb.r <= 255
            assert 0 <= rgb.g <= 255
            assert 0 <= rgb.b <= 255

class TestDataClasses:
    def test_glucose_item_creation(self):
        now = datetime.now()
        item = GlucoseItem(EntrieEnum.SGV, 120, now, "Flat")
        assert item.type == EntrieEnum.SGV
        assert item.glucose == 120
        assert item.date == now
        assert item.direction == "Flat"

    def test_treatment_item_creation(self):
        now = datetime.now()
        item = TreatmentItem("test_id", TreatmentEnum.BOLUS, now, 5)
        assert item.id == "test_id"
        assert item.type == TreatmentEnum.BOLUS
        assert item.amount == 5

    def test_exercise_item_creation(self):
        now = datetime.now()
        item = ExerciseItem(TreatmentEnum.EXERCISE, now, 30)
        assert item.type == TreatmentEnum.EXERCISE
        assert item.amount == 30

    def test_iob_item_creation(self):
        now = datetime.now()
        item = IobItem(now, 2.5)
        assert item.date == now
        assert item.amount == 2.5