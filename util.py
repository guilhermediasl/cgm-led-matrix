import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import NamedTuple, Optional, Union

_FRACTION = re.compile(r"\.(\d+)")


def utc_to_local(value: Union[str, int, float]) -> datetime:
    """Convert a Nightscout timestamp to a naive datetime in the machine's local time.

    Accepts epoch milliseconds (the `date` field) or an ISO 8601 string such as
    `2024-05-01T12:00:00.000Z`, with or without fractional seconds or an offset.
    Strings without an offset are taken as UTC, which is what Nightscout stores.
    """
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000)
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    # Older Pythons only parse exactly 6 fractional digits.
    text = _FRACTION.sub(lambda m: "." + m.group(1)[:6].ljust(6, "0"), text, count=1)
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().replace(tzinfo=None)


def entry_local_time(item: dict) -> Optional[datetime]:
    """Local time of a Nightscout entry, preferring the unambiguous epoch `date`."""
    if isinstance(item.get("date"), (int, float)):
        return utc_to_local(item["date"])
    if item.get("dateString"):
        return utc_to_local(item["dateString"])
    return None

class TreatmentEnum(Enum):
    """Enumeration for different types of diabetes treatments."""
    BOLUS = 'Bolus'
    CARBS = 'Carbs'
    EXERCISE = 'Exercise'
    BASAL = 'Basal Injection'

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

@dataclass
class IobItem:
    """Represents an insulin on board (IOB) entry."""
    date: datetime
    amount: float

    def __str__(self):
        return f"IobItem (date='{self.date}', amount={self.amount})"