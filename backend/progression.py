"""Gemini-powered next-day deterioration prediction.

Pipeline: daily IMU feature summaries (ml.progression.features) accumulate in
the `daily_features` table, one row per day. Once 14 *consecutive* days are on
record, the most recent window can be sent to Gemini with a structured JSON
schema to predict the next day's deterioration risk. Call `Progression.close_day`
once per day (a background task in main.py does this automatically at UTC
midnight) and `Progression.predict` to get a prediction.

Guardrails:
  * `predict()` refuses to call Gemini until the baseline window is complete
    (see `status()` / `ProgressionError` with status 409).
  * a successful prediction is cached per target day in `progression_predictions`,
    so re-requesting the same day is free until new data changes the window.
  * `RateLimiter` caps outbound Gemini calls per hour regardless of caching.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field

from backend.db import Database
from ml.progression.features import aggregate_daily_features

log = logging.getLogger("progression")


class ProgressionError(Exception):
    """Carries an HTTP status for main.py."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ---- structured Gemini output ----------------------------------------------
class Marker(BaseModel):
    name: str = Field(description="Physiological marker, e.g. 'jaw movement asymmetry'")
    direction: Literal["increased", "decreased", "unchanged"]
    severity: Literal["low", "moderate", "high"]
    evidence: str = Field(description="One sentence citing the specific daily values that triggered this marker")


class DeteriorationPrediction(BaseModel):
    risk_score: Literal["low", "moderate", "elevated", "high"] = Field(
        description="Qualitative deterioration risk for the next day"
    )
    deterioration_index: float = Field(ge=0.0, le=1.0, description="0 = no change expected, 1 = severe deterioration")
    confidence: float = Field(ge=0.0, le=1.0, description="Model's confidence in this prediction")
    markers: list[Marker] = Field(description="Specific features that triggered or support the warning")
    clinical_notes: list[str] = Field(description="Actionable notes for the patient/caretaker, plain language")
    summary: str = Field(description="One paragraph summary of the prediction and its reasoning")


PROMPT_TEMPLATE = """You are assisting a remote monitoring system for a person with ALS (amyotrophic lateral \
sclerosis). Below is a 14-day, day-by-day sequence of statistical features extracted from face-mounted IMU \
sensors tracking jaw movement during silent mouthed speech, plus real-time word-detection and rest-pose data.

Each day's features (null means that feature could not be computed that day, usually from too little data):
- velocity_variance: variance of head-referenced jaw angular velocity (rad/s)^2 -- lower sustained motion can \
indicate reduced range of motion.
- tremor_frequency_hz: dominant frequency of jaw motion in the 3-12 Hz tremor band.
- jaw_asymmetry_index (0-1): imbalance between pitch and yaw rotation range during jaw movement; higher can \
indicate compensatory or uneven muscle weakening.
- mean_clip_duration_s: average duration of a mouthed word.
- speech_cadence_wpm: accepted word detections per minute of active use.
- detection_confidence_mean: mean model confidence on accepted detections.
- rejection_rate (0-1): fraction of attempted words the system could not confidently recognize.
- rest_activity_rms: residual jaw motion while explicitly holding still (fatigue/tremor baseline).
- n_clips / n_detections / n_accepted_detections: daily usage volume (low volume days have noisier features).

Analyze the 14-day trend (not just the last day) and predict deterioration risk for day 15, the day immediately \
after this window. Flag only markers with real evidence in the data -- do not invent trends that are not visible \
in the numbers, and say so plainly when the data is too sparse to be confident.

14-day window (oldest to newest), as JSON:
{window_json}
"""


# ---- client-side guardrails --------------------------------------------------
class RateLimiter:
    """Caps outbound Gemini calls to `max_per_hour`; raises instead of blocking the request."""

    def __init__(self, max_per_hour: int) -> None:
        self.max_per_hour = max_per_hour
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = datetime.now(timezone.utc).timestamp()
        while self._calls and now - self._calls[0] > 3600:
            self._calls.popleft()
        if len(self._calls) >= self.max_per_hour:
            retry_in = 3600 - (now - self._calls[0])
            raise ProgressionError(f"Gemini call rate limit reached; retry in {retry_in:.0f}s", status=429)
        self._calls.append(now)


class BaselineStatus(BaseModel):
    ready: bool
    days_collected: int
    days_required: int
    days_remaining: int
    window_start: date | None = None
    window_end: date | None = None
    message: str


async def check_baseline(db: Database, as_of: date, required_days: int) -> BaselineStatus:
    """Consecutive days of daily_features immediately before `as_of`, newest first until the first gap."""
    end_day = as_of - timedelta(days=1)  # `as_of`'s own data is still being collected, never part of its window
    rows = await db.daily_features_window(end_day, required_days)
    if rows is None:
        return BaselineStatus(
            ready=False, days_collected=0, days_required=required_days, days_remaining=required_days,
            message="database unavailable; cannot evaluate the baseline",
        )
    have = {r["day"] for r in rows}
    streak = 0
    day = end_day
    while day in have:
        streak += 1
        day -= timedelta(days=1)
    if streak >= required_days:
        window = rows[-required_days:]
        return BaselineStatus(
            ready=True, days_collected=streak, days_required=required_days, days_remaining=0,
            window_start=window[0]["day"], window_end=window[-1]["day"],
            message="baseline complete; ready to predict",
        )
    remaining = required_days - streak
    return BaselineStatus(
        ready=False, days_collected=streak, days_required=required_days, days_remaining=remaining,
        message=f"{remaining} more consecutive day(s) of data needed before a prediction can be made",
    )


# ---- Gemini call with retry/backoff -----------------------------------------
class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout_s: float, max_retries: int) -> None:
        self.model, self.timeout_s, self.max_retries = model, timeout_s, max_retries
        self._client = None
        if api_key:
            from google import genai

            self._client = genai.Client(api_key=api_key)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def predict(self, window: list[dict]) -> DeteriorationPrediction:
        if not self.enabled:
            raise ProgressionError("GEMINI_API_KEY is not configured", status=503)
        from google.genai import errors, types

        prompt = PROMPT_TEMPLATE.format(window_json=_window_json(window))
        config = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=DeteriorationPrediction, temperature=0.2,
        )
        backoff = 1.0
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                res = await asyncio.wait_for(
                    self._client.aio.models.generate_content(model=self.model, contents=prompt, config=config),
                    self.timeout_s,
                )
                if isinstance(res.parsed, DeteriorationPrediction):
                    return res.parsed
                return DeteriorationPrediction.model_validate_json(res.text)
            except errors.ClientError as e:
                last_error = e
                if e.code == 429 and attempt < self.max_retries:  # Gemini's own rate limit: back off and retry
                    log.warning("Gemini rate limited (attempt %d/%d); backing off %.1fs", attempt, self.max_retries, backoff)
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise ProgressionError(f"Gemini rejected the request: {e}", status=502) from e
            except (errors.ServerError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < self.max_retries:
                    log.warning("Gemini call failed (attempt %d/%d): %s; retrying in %.1fs", attempt, self.max_retries, e, backoff)
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
            except Exception as e:  # noqa: BLE001  (malformed structured output, etc.)
                raise ProgressionError(f"could not parse Gemini's response: {e}", status=502) from e
        raise ProgressionError(f"Gemini unavailable after {self.max_retries} attempts: {last_error}", status=502)


def _window_json(window: list[dict]) -> str:
    import json

    return json.dumps([{"day": r["day"].isoformat(), **r["features"]} for r in window], indent=2)


# ---- orchestration -----------------------------------------------------------
class Progression:
    def __init__(self, db: Database, api_key: str, model: str, timeout_s: float, max_retries: int,
                 max_calls_per_hour: int, baseline_days: int) -> None:
        self.db = db
        self.baseline_days = baseline_days
        self.client = GeminiClient(api_key, model, timeout_s, max_retries)
        self.limiter = RateLimiter(max_calls_per_hour)

    async def status(self, as_of: date | None = None) -> BaselineStatus:
        return await check_baseline(self.db, as_of or _today(), self.baseline_days)

    async def window(self, as_of: date | None = None) -> list[dict]:
        """Up to the last `baseline_days` of daily feature summaries, for charting (no guardrail: this is
        read-only data transparency, not a Gemini call, so partial/pre-baseline windows are returned too)."""
        as_of = as_of or _today()
        rows = await self.db.daily_features_window(as_of - timedelta(days=1), self.baseline_days)
        if rows is None:
            raise ProgressionError("database unavailable", status=503)
        return [{"day": r["day"].isoformat(), "features": r["features"]} for r in rows]

    async def close_day(self, day: date) -> dict:
        """Compute and store the feature summary for `day` from clips/detections/rest-poses."""
        clips = await self.db.clips_with_samples_for_day(day)
        detections = await self.db.detections_for_day(day)
        rest = await self.db.rest_poses_for_day(day)
        if clips is None or detections is None or rest is None:
            raise ProgressionError("database unavailable; could not close out the day", status=503)
        features = aggregate_daily_features(clips, detections, rest)
        self.db.save_daily_features(day, features["n_clips"], features["n_detections"], features)
        log.info("closed daily features for %s (%d clips, %d detections)", day, features["n_clips"], features["n_detections"])
        return {"day": day.isoformat(), "features": features}

    async def predict(self, as_of: date | None = None, force: bool = False) -> dict:
        as_of = as_of or _today()
        status = await self.status(as_of)
        if not status.ready:
            raise ProgressionError(status.message, status=409)

        if not force:
            cached = await self.db.get_progression_prediction(as_of)
            if cached and cached["window_end"] == status.window_end:
                return {
                    "target_day": cached["target_day"].isoformat(), "window_start": cached["window_start"].isoformat(),
                    "window_end": cached["window_end"].isoformat(), "response": cached["response"], "cached": True,
                }

        self.limiter.acquire()
        window = await self.db.daily_features_window(status.window_end, self.baseline_days)
        if window is None:
            raise ProgressionError("database unavailable; could not load the feature window", status=503)

        prediction = await self.client.predict(window)
        response = prediction.model_dump()
        self.db.save_progression_prediction(as_of, status.window_start, status.window_end, response)
        return {
            "target_day": as_of.isoformat(), "window_start": status.window_start.isoformat(),
            "window_end": status.window_end.isoformat(), "response": response, "cached": False,
        }


def _today() -> date:
    return datetime.now(timezone.utc).date()


async def daily_close_task(progression: Progression) -> None:
    """Sleeps until each UTC midnight, then closes out the day that just ended."""
    while True:
        now = datetime.now(timezone.utc)
        next_midnight = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) + timedelta(days=1)
        await asyncio.sleep((next_midnight - now).total_seconds())
        day = next_midnight.date() - timedelta(days=1)
        try:
            await progression.close_day(day)
        except ProgressionError as e:
            log.warning("daily close for %s skipped: %s", day, e)
        except Exception:  # noqa: BLE001
            log.exception("daily close for %s failed", day)
