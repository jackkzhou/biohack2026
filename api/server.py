import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat, field_validator
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError

logger = logging.getLogger(__name__)
app = FastAPI(title="Mouth Sensor Data Ingestion")

_INSERT_SENSOR_ROWS = text(
    """
    INSERT INTO mouth_sensor_data (time, session_id, label, ax, ay, az)
    VALUES (:time, :session_id, :label, :ax, :ay, :az)
    """
)


class SensorSample(BaseModel):
    time: Annotated[
        datetime,
        Field(description="Timezone-aware ISO 8601 timestamp"),
    ]
    ax: FiniteFloat
    ay: FiniteFloat
    az: FiniteFloat

    @field_validator("time")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("time must include a timezone")
        return value


class StreamBatch(BaseModel):
    session_id: str = Field(min_length=1, max_length=255)
    label: str = Field(min_length=1, max_length=255)
    samples: list[SensorSample] = Field(min_length=1, max_length=5000)

    @field_validator("session_id", "label")
    @classmethod
    def require_non_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


@lru_cache(maxsize=1)
def _get_engine() -> Engine:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    try:
        url = make_url(database_url)
    except ArgumentError as exc:
        raise RuntimeError("DATABASE_URL is invalid") from exc

    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+psycopg2")
    elif url.drivername != "postgresql+psycopg2":
        raise RuntimeError("DATABASE_URL must use PostgreSQL with psycopg2")

    return create_engine(url, pool_pre_ping=True)


@app.post("/stream", status_code=201)
def stream(batch: StreamBatch) -> dict[str, int | str]:
    try:
        engine = _get_engine()
        rows = [
            {
                "time": sample.time,
                "session_id": batch.session_id,
                "label": batch.label,
                "ax": sample.ax,
                "ay": sample.ay,
                "az": sample.az,
            }
            for sample in batch.samples
        ]
        with engine.begin() as connection:
            connection.execute(_INSERT_SENSOR_ROWS, rows)
    except RuntimeError as exc:
        logger.error("Sensor batch could not be stored: %s", exc)
        raise HTTPException(
            status_code=503, detail="Database is not configured correctly"
        ) from exc
    except SQLAlchemyError as exc:
        logger.error("Failed to persist sensor batch (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="Sensor data storage is temporarily unavailable"
        ) from exc

    return {"session_id": batch.session_id, "inserted": len(rows)}
