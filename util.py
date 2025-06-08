from datetime import datetime
from enum import Enum
from typing import Tuple, NamedTuple

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

class GlucoseItem:
    def __init__(self, type: EntrieEnum, glucose: int, date: datetime, direction: str = ''):
        self.type: EntrieEnum = type
        self.glucose: int = glucose
        self.date: datetime = date 
        self.direction: str = direction

    def __str__(self):
        return f"GlucoseItem (type='{self.type}', date='{self.date}', glucose={self.glucose})"

    def __repr__(self):
        return self.__str__()

class TreatmentItem:
    def __init__(self, id: str, type: TreatmentEnum, date: datetime, amount: int):
        self.id: str = id
        self.type: TreatmentEnum = type
        self.date: datetime = date
        self.amount: int = amount

    def __str__(self):
        return f"TreatmentItem(type='{self.type}', date='{self.date}', amount={self.amount})"

    def __repr__(self):
        return self.__str__()

class ExerciseItem:
    def __init__(self, type: TreatmentEnum, date: datetime, amount: int):
        self.type: TreatmentEnum = type
        self.date: datetime = date
        self.amount: int = amount

    def __str__(self):
        return f"ExerciseItem (type='{self.type}', date='{self.date}', amount={self.amount})"

    def __repr__(self):
        return self.__str__()