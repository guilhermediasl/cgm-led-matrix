from unittest.mock import MagicMock
from datetime import datetime

from google_calendar import GoogleCalendarSync, format_glucose_event


def test_format_glucose_event():
    assert format_glucose_event(62, "DoubleUp", -1) == "🟡 62 ↑↑ -1"
    assert format_glucose_event(100, "Flat", 0) == "🟢 100 → +0"
    assert format_glucose_event(50, "SingleDown", -20) == "🔴 50 ↓ -20"


def test_sync_deletes_previous_events_before_creating_current_event():
    service = MagicMock()
    service.events().list().execute.return_value = {"items": [{"id": "old-event"}]}
    service.events().insert().execute.return_value = {"id": "new-event"}

    calendar = GoogleCalendarSync()
    calendar._service = service

    assert calendar.sync(62, "DoubleUp", -1) == "new-event"
    assert service.events().delete.call_args.kwargs["eventId"] == "old-event"
    assert service.events().insert.call_args.kwargs["body"]["summary"] == "🟡 62 ↑↑ -1"


def test_sync_resolves_calendar_name_and_creates_ten_minute_event():
    service = MagicMock()
    service.calendarList().list().execute.return_value = {
        "items": [{"id": "cgm-calendar-id", "summary": "cgm"}]
    }
    service.events().list().execute.return_value = {"items": []}
    service.events().insert().execute.return_value = {"id": "new-event"}

    calendar = GoogleCalendarSync(calendar_id="cgm")
    calendar._service = service

    calendar.sync(62, "DoubleUp", -1)

    request = service.events().insert.call_args.kwargs
    start = datetime.fromisoformat(request["body"]["start"]["dateTime"])
    end = datetime.fromisoformat(request["body"]["end"]["dateTime"])
    assert request["calendarId"] == "cgm-calendar-id"
    assert (end - start).total_seconds() == 600