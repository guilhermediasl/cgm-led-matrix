from pydantic import BaseModel, Field, HttpUrl
from typing import Optional


class ConfigModel(BaseModel):
    token: str = Field(..., description="Nightscout API token")
    url: HttpUrl = Field(..., description="Nightscout base URL")
    ip: Optional[str] = Field(None, description="IP or MAC address of the iDot device")
    output_type: str = Field("gif", alias="output type")
    image_out: str = Field("pc", alias="image out")
    os: str = Field("windows")
    show_time: bool = Field(False)
    time_format: str = Field("12h")
    time_position: str = Field("bottom-left")
    time_color_fade: float = Field(0.4)
    plot_glucose_intervals: bool = Field(True, alias="plot glucose intervals")
    low_boundary_glucose: Optional[int] = Field(None, alias="low boundary glucose")
    high_boundary_glucose: Optional[int] = Field(None, alias="high boundary glucose")
    night_brightness: Optional[float] = Field(10.0)
    PIXEL_INTERVAL: int = Field(5)

    class Config:
        allow_population_by_field_name = True
        extra = 'forbid'
