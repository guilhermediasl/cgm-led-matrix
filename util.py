from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import NamedTuple

class TreatmentEnum(Enum):
    """Enumeration for different types of diabetes treatments."""
    BOLUS = 'Bolus'
    CARBS = 'Carbs'
    EXERCISE = 'Exercise'

class EntrieEnum(str, Enum):
    """Enumeration for different types of glucose entries."""
    SGV = 'sgv'  # Sensor Glucose Value
    MBG = 'mbg'  # Manual Blood Glucose

class ColorType(NamedTuple):
    """RGB color."""
    r: int
    g: int
    b: int

class Color(Enum):
    """Predefined color palette."""
    red    = (255, 20, 10)
    green  = (70, 167, 10)
    yellow = (244, 170, 0)
    purple = (250, 0, 105)
    pink   = (250, 100, 120)
    white  = (220, 175, 100)
    blue   = (40, 150, 125)
    cyan   = (150, 220, 100)
    orange = (245, 70, 0)
    black  = (0, 0, 0)

    @property
    def rgb(self) -> ColorType:
        """Convert enum value to ColorType tuple.
        
        Returns:
            ColorType: RGB values as a named tuple
        """
        return ColorType(*self.value)

@dataclass
class GlucoseItem:
    """Represents a glucose reading from CGM or manual blood glucose."""
    type: EntrieEnum
    glucose: int
    date: datetime
    direction: str = ''

    def __str__(self):
        return f"GlucoseItem (type='{self.type}', date='{self.date}', glucose={self.glucose})"

@dataclass
class TreatmentItem:
    """Represents a diabetes treatment entry (bolus or carbs)."""
    id: str
    type: TreatmentEnum
    date: datetime
    amount: int

    def __str__(self):
        return f"TreatmentItem(type='{self.type}', date='{self.date}', amount={self.amount})"

@dataclass
class ExerciseItem:
    """Represents an exercise activity entry."""
    type: TreatmentEnum
    date: datetime
    amount: int

    def __str__(self):
        return f"ExerciseItem (type='{self.type}', date='{self.date}', amount={self.amount})"