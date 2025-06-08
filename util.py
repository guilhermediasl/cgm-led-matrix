from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import NamedTuple

class TreatmentEnum(Enum):
    BOLUS = 'Bolus'
    CARBS = 'Carbs'
    EXERCISE = 'Exercise'

class EntrieEnum(str, Enum):
    SGV = 'sgv'
    MBG = 'mbg'
class ColorType(NamedTuple):
    r: int
    g: int
    b: int
class Color(Enum):
    red    = (255, 20, 10)
    green  = (70, 167, 10)
    yellow = (244, 170, 0)
    purple = (250, 0, 105)
    pink   = (250, 100, 120)
    white  = (240, 180, 70)
    blue   = (40, 150, 125)
    cyan   = (150, 220, 100)
    orange = (245, 70, 0)
    black  = (0, 0, 0)

    @property
    def rgb(self) -> ColorType:
        return ColorType(*self.value)

@dataclass
class GlucoseItem:
    type: EntrieEnum
    glucose: int
    date: datetime
    direction: str = ''

    def __str__(self):
        return f"GlucoseItem (type='{self.type}', date='{self.date}', glucose={self.glucose})"

@dataclass
class TreatmentItem:
    id: str
    type: TreatmentEnum
    date: datetime
    amount: int

    def __str__(self):
        return f"TreatmentItem(type='{self.type}', date='{self.date}', amount={self.amount})"

@dataclass
class ExerciseItem:
    type: TreatmentEnum
    date: datetime
    amount: int

    def __str__(self):
        return f"ExerciseItem (type='{self.type}', date='{self.date}', amount={self.amount})"