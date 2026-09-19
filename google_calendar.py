import datetime
import logging
import os
from typing import Any


ARROW_SYMBOLS = {
    "DoubleDown": "↓↓",
    "DoubleUp": "↑↑",
    "Flat": "→",
    "FortyFiveDown": "↘",
    "FortyFiveUp": "↗",
    "SingleDown": "↓",
    "SingleUp": "↑",
    "-": "→",
}


def glucose_status_emoji(glucose: int, low_boundary: int, high_boundary: int) -> str:
    margin = 10
    if glucose < low_boundary - margin or glucose > high_boundary + margin:
        return "🔴"
    if glucose <= low_boundary or glucose >= high_boundary:
        return "🟡"
    return "🟢"


def format_glucose_event(
    glucose: int,
    direction: str,
    difference: int,
    low_boundary: int = 70,
    high_boundary: int = 160,
) -> str:
    arrow = ARROW_SYMBOLS.get(direction, direction or "→")
    status = glucose_status_emoji(glucose, low_boundary, high_boundary)
    return f"{status} {glucose} {arrow} {difference:+d}"


class GoogleCalendarSync:
    """Keep one ten-minute event for the current glucose reading."""

    SCOPES = ["https://www.googleapis.com/auth/calendar"]

    def __init__(self, calendar_id="primary", credentials_path="credentials.json", token_path="calendar_token.json", low_boundary=70, high_boundary=160):
        self.calendar_id = calendar_id
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.low_boundary = low_boundary
        self.high_boundary = high_boundary
        self._service: Any = None

    def _get_service(self) -> Any:
        if self._service is not None:
            return self._service

        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise RuntimeError("Google Calendar dependencies are missing. Install requirements.txt.") from exc

        credentials = None
        if os.path.exists(self.token_path):
            credentials = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        elif not credentials or not credentials.valid:
            if not os.path.exists(self.credentials_path):
                raise FileNotFoundError(
                    f"Google OAuth credentials not found: {os.path.abspath(self.credentials_path)}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.SCOPES)
            credentials = flow.run_local_server(port=0)

        with open(self.token_path, "w", encoding="utf-8") as token_file:
            token_file.write(credentials.to_json())
        self._service = build("calendar", "v3", credentials=credentials)
        return self._service

    def _resolve_calendar_id(self, service: Any) -> str:
        if self.calendar_id == "primary":
            return "primary"

        calendars = service.calendarList().list(showHidden=False).execute().get("items", [])
        for calendar in calendars:
            if calendar.get("summary", "").casefold() == self.calendar_id.casefold():
                return calendar["id"]
        raise ValueError(f"Google Calendar not found: {self.calendar_id}")

    def sync(self, glucose: int, direction: str, difference: int) -> str:
        service = self._get_service()
        calendar_id = self._resolve_calendar_id(service)
        previous_events = service.events().list(
            calendarId=calendar_id,
            privateExtendedProperty="source=cgm-led-matrix",
            showDeleted=False,
            singleEvents=True,
        ).execute().get("items", [])
        for event in previous_events:
            service.events().delete(calendarId=calendar_id, eventId=event["id"]).execute()

        start = datetime.datetime.now().astimezone()
        end = start + datetime.timedelta(minutes=10)
        event = {
            "summary": format_glucose_event(
                glucose,
                direction,
                difference,
                self.low_boundary,
                self.high_boundary,
            ),
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
            "extendedProperties": {"private": {"source": "cgm-led-matrix"}},
        }
        return service.events().insert(calendarId=calendar_id, body=event).execute()["id"]