from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='forbid')

    token: str = Field(..., description="Nightscout API token")
    url: str = Field(..., description="Nightscout base URL")
    ip: Optional[str] = Field(None, description="IP or MAC address of the iDot device")
    # Deprecated: GIF output was removed and the display always gets an image.
    # Still accepted so existing config files keep validating; ignored by the app.
    output_type: Optional[str] = Field(None, alias="output type")
    image_out: str = Field("led matrix", alias="image out")
    os: str = Field("linux")
    show_time: bool = Field(False)
    time_format: str = Field("24h")
    time_position: str = Field("top-right")
    time_color_fade: float = Field(0.4)
    plot_glucose_intervals: bool = Field(True, alias="plot glucose intervals")
    low_boundary_glucose: Optional[int] = Field(None, alias="low boundary glucose")
    high_boundary_glucose: Optional[int] = Field(None, alias="high boundary glucose")
    night_brightness: Optional[float] = Field(10.0, ge=0, le=100)
    stale_minutes: int = Field(15, ge=1, description="Minutes without a reading before the data is marked stale")
    PIXEL_INTERVAL: int = Field(5)
    google_calendar_enabled: bool = Field(False)
    google_calendar_id: str = Field("primary")
    google_calendar_credentials: str = Field("credentials.json")
    google_calendar_token: str = Field("calendar_token.json")

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return value.rstrip("/")
