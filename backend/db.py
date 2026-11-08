"""Tiger Data (TimescaleDB) storage.

Nothing here can block or break live inference:
  * writes go into in-memory queues and are flushed by one background task in
    batches (COPY), never from the sensor thread or the inference loop;
  * the raw-sample queue is bounded and drops its oldest rows if the database is
    slow; clips, deletes and phrases are kept and retried until they succeed;
  * reads have a short timeout and return None when the database is offline,
    so callers fall back to local data.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import numpy as np

log = logging.getLogger("db")

FLUSH_S = 1.0
SAMPLE_BATCH = 5000
SAMPLE_QUEUE_MAX = 60_000  # ~5 min of both sensors at 100 Hz
OPS_MAX = 2000
READ_TIMEOUT_S = 2.0

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS imu_samples (
        time timestamptz NOT NULL, sensor smallint NOT NULL,
        gx real, gy real, gz real, ax real, ay real, az real,
        t_us bigint, qw real, qx real, qy real, qz real)""",
    "SELECT create_hypertable('imu_samples', 'time', if_not_exists => TRUE)",
    """CREATE TABLE IF NOT EXISTS clips (
        id bigint PRIMARY KEY, label text NOT NULL, session integer, source text NOT NULL,
        t_start timestamptz, t_end timestamptz, n_samples integer,
        created_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS clip_samples (
        clip_id bigint NOT NULL, label text NOT NULL, session integer,
        time timestamptz NOT NULL, sensor smallint NOT NULL,
        gx real, gy real, gz real, ax real, ay real, az real,
        t_us bigint, qw real, qx real, qy real, qz real)""",
    "SELECT create_hypertable('clip_samples', 'time', if_not_exists => TRUE)",
    "ALTER TABLE imu_samples ADD COLUMN IF NOT EXISTS qw real, ADD COLUMN IF NOT EXISTS qx real, ADD COLUMN IF NOT EXISTS qy real, ADD COLUMN IF NOT EXISTS qz real",
    "ALTER TABLE clip_samples ADD COLUMN IF NOT EXISTS qw real, ADD COLUMN IF NOT EXISTS qx real, ADD COLUMN IF NOT EXISTS qy real, ADD COLUMN IF NOT EXISTS qz real",
    "CREATE INDEX IF NOT EXISTS clip_samples_clip_idx ON clip_samples (clip_id, time)",
    """CREATE TABLE IF NOT EXISTS phrases (
        id bigint PRIMARY KEY, at timestamptz NOT NULL, words text[] NOT NULL, sentence text NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS detections (
        time timestamptz NOT NULL, word text NOT NULL, confidence real NOT NULL, accepted boolean NOT NULL)""",
    "SELECT create_hypertable('detections', 'time', if_not_exists => TRUE)",
    """CREATE TABLE IF NOT EXISTS tares (
        id bigint PRIMARY KEY, session integer, at timestamptz, ok boolean NOT NULL, error text,
        rotation_angle_deg double precision, rotation jsonb,
        q_jaw double precision[], q_head double precision[],
        gravity_chin double precision[], gravity_head double precision[],
        n_chin integer, n_head integer)""",
    """CREATE TABLE IF NOT EXISTS rest_poses (
        session integer NOT NULL, at timestamptz NOT NULL DEFAULT now(),
        chin double precision[], head double precision[], activity_rms double precision)""",
]

# Same columns as the firmware line: t_us,sensor,gx,gy,gz,ax,ay,az,qw,qx,qy,qz.
# t_us is the ESP32 microsecond counter, unwrapped so it keeps increasing past the 32-bit rollover.
SAMPLE_COLS = ["time", "sensor", "gx", "gy", "gz", "ax", "ay", "az", "t_us", "qw", "qx", "qy", "qz"]
CLIP_COLS = ["clip_id", "label", "session"] + SAMPLE_COLS


def _ts(t: float) -> datetime:
    return datetime.fromtimestamp(t, tz=timezone.utc)


def _records(rows: np.ndarray, clip: dict | None = None) -> list[tuple]:
    """Ring rows [t, t_esp, g(3), a(3), q(4), sensor] -> DB tuples."""
    out = []
    for r in rows:
        base = (_ts(r[0]), int(r[12]), *map(float, r[2:8]), int(round(r[1] * 1e6)), *map(float, r[8:12]))
        out.append(base if clip is None else (clip["id"], clip["label"], clip["session"], *base))
    return out


class Database:
    def __init__(self, url: str, store_raw: bool = True) -> None:
        self.url = url
        self.store_raw = store_raw
        self.online = False
        self.dropped_samples = 0
        self._pool: Any = None
        self._samples: deque[np.ndarray] = deque(maxlen=SAMPLE_QUEUE_MAX)
        self._ops: deque[tuple] = deque()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    # ---- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        if not self.enabled:
            log.info("TIGER_DB_URL not set: running without a database")
            return
        self._task = asyncio.create_task(self._run(), name="db-writer")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                self._task.cancel()
        if self._pool:
            await self._pool.close()

    @property
    def pending(self) -> int:
        """Writes (clips, deletes, phrases...) still waiting to be stored."""
        return len(self._ops)

    async def wait_flushed(self, timeout: float) -> bool:
        """Wait until queued writes are stored. False if the database never caught up."""
        end = time.monotonic() + timeout
        while self._ops and time.monotonic() < end:
            await asyncio.sleep(0.2)
        return not self._ops

    # ---- write API (never blocks, never raises) --------------------------
    def push_sample(self, row: np.ndarray) -> None:
        """Called from the serial thread for every sample."""
        if not (self.enabled and self.store_raw):
            return
        if len(self._samples) == self._samples.maxlen:
            self.dropped_samples += 1
        self._samples.append(row)

    def _op(self, *op: Any) -> None:
        if not self.enabled:
            return
        if len(self._ops) >= OPS_MAX:
            self._ops.popleft()
            log.warning("db op queue full; dropped the oldest pending write")
        self._ops.append(op)

    def save_clip(self, meta: dict, rows: np.ndarray) -> None:
        self._op("clip", meta, rows)

    def delete_clip(self, clip_id: int) -> None:
        self._op("delete", clip_id)

    def save_phrase(self, rec: dict) -> None:
        self._op("phrase", rec)

    def save_detection(self, t: float, word: str, confidence: float, accepted: bool) -> None:
        self._op("detection", t, word, confidence, accepted)

    def save_rest_pose(self, session: int, chin: list[float], head: list[float], rms: float) -> None:
        self._op("rest", session, chin, head, rms)

    def save_tare(self, tare_id: int, rec: dict) -> None:
        """One row per tare attempt (successful or not); `tare_id` is the record's ms timestamp."""
        self._op("tare", tare_id, rec)

    # ---- read API (None = database unavailable) --------------------------
    async def _read(self, fn: Callable[[Any], Awaitable[Any]]) -> Any:
        if not (self.online and self._pool):
            return None
        try:
            async with self._pool.acquire() as con:
                return await asyncio.wait_for(fn(con), READ_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001
            log.warning("db read failed: %s", e)
            return None

    async def get_clip(self, clip_id: int) -> tuple[dict, np.ndarray] | None:
        async def go(con: Any) -> tuple[dict, np.ndarray] | None:
            meta = await con.fetchrow("SELECT id, label, session, source, t_start FROM clips WHERE id=$1", clip_id)
            if not meta:
                return None
            recs = await con.fetch(
                "SELECT time, t_us, gx, gy, gz, ax, ay, az, sensor, qw, qx, qy, qz FROM clip_samples WHERE clip_id=$1 ORDER BY time",
                clip_id,
            )
            rows = np.array(
                [[r["time"].timestamp(), r["t_us"] / 1e6, r["gx"], r["gy"], r["gz"], r["ax"], r["ay"], r["az"],
                  r["qw"] or 0, r["qx"] or 0, r["qy"] or 0, r["qz"] or 0, r["sensor"]] for r in recs],
                dtype=np.float64,
            ).reshape(-1, 13)
            return dict(meta), rows

        return await self._read(go)

    async def counts(self) -> dict[str, int] | None:
        async def go(con: Any) -> dict[str, int]:
            recs = await con.fetch("SELECT label, count(*) AS n FROM clips WHERE source='record' GROUP BY label")
            return {r["label"]: int(r["n"]) for r in recs}

        return await self._read(go)

    async def list_clips(self, limit: int = 2000) -> list[dict] | None:
        """Metadata of stored clips, newest first (no samples)."""
        async def go(con: Any) -> list[dict]:
            recs = await con.fetch(
                "SELECT id, label, session, source, n_samples, created_at FROM clips ORDER BY id DESC LIMIT $1", limit)
            return [
                {"id": int(r["id"]), "label": r["label"], "session": r["session"], "source": r["source"],
                 "n_samples": r["n_samples"], "created_at_ms": int(r["created_at"].timestamp() * 1000)}
                for r in recs
            ]

        return await self._read(go)

    async def history(self, limit: int = 50) -> list[dict] | None:
        async def go(con: Any) -> list[dict]:
            recs = await con.fetch("SELECT id, at, words, sentence FROM phrases ORDER BY at DESC LIMIT $1", limit)
            return [
                {"id": int(r["id"]), "words": list(r["words"]), "sentence": r["sentence"], "at_ms": int(r["at"].timestamp() * 1000)}
                for r in recs
            ]

        return await self._read(go)

    # ---- background writer ----------------------------------------------
    async def _connect(self) -> None:
        import asyncpg

        pool = await asyncio.wait_for(asyncpg.create_pool(self.url, min_size=1, max_size=3, command_timeout=15), 10)
        async with pool.acquire() as con:
            for stmt in SCHEMA:
                try:
                    await con.execute(stmt)
                except Exception as e:  # noqa: BLE001  (e.g. plain Postgres without the timescaledb extension)
                    log.warning("schema statement skipped (%s): %.60s", e, " ".join(stmt.split()))
        self._pool = pool
        self.online = True
        log.info("connected to Tiger Data")

    def _disconnect(self) -> None:
        self.online = False
        pool, self._pool = self._pool, None
        if pool:
            asyncio.create_task(pool.close())

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            if not self.online:
                try:
                    await self._connect()
                    backoff = 1.0
                except Exception as e:  # noqa: BLE001
                    log.warning("database unavailable (%s); retrying in %.0fs. Live inference is unaffected.", e, backoff)
                    try:
                        await asyncio.wait_for(self._stop.wait(), backoff)
                    except asyncio.TimeoutError:
                        pass
                    backoff = min(backoff * 2, 30)
                    continue
            try:
                await self._flush()
            except Exception as e:  # noqa: BLE001
                log.warning("database write failed (%s); will reconnect", e)
                self._disconnect()
                continue
            try:
                await asyncio.wait_for(self._stop.wait(), FLUSH_S)
            except asyncio.TimeoutError:
                pass
        if self.online:  # final flush on shutdown
            try:
                await self._flush()
            except Exception:  # noqa: BLE001
                pass

    async def _flush(self) -> None:
        # ordered, important writes first (an op stays queued until it succeeds)
        while self._ops:
            op = self._ops[0]
            async with self._pool.acquire() as con:
                await self._apply(con, op)
            self._ops.popleft()
        # raw samples: best effort, a failed batch is dropped rather than retried forever
        while self._samples:
            batch = [self._samples.popleft() for _ in range(min(SAMPLE_BATCH, len(self._samples)))]
            try:
                async with self._pool.acquire() as con:
                    await con.copy_records_to_table("imu_samples", records=_records(np.array(batch)), columns=SAMPLE_COLS)
            except Exception:
                self.dropped_samples += len(batch)
                raise

    async def _apply(self, con: Any, op: tuple) -> None:
        kind = op[0]
        if kind == "clip":
            _, m, rows = op
            async with con.transaction():
                status = await con.execute(
                    "INSERT INTO clips (id,label,session,source,t_start,t_end,n_samples) VALUES ($1,$2,$3,$4,$5,$6,$7) "
                    "ON CONFLICT (id) DO NOTHING",
                    m["id"], m["label"], m["session"], m["source"], _ts(m["t_start"]), _ts(m["t_end"]), len(rows),
                )
                if status.endswith(" 0"):  # clip already stored (re-upload): do not duplicate its samples
                    return
                await con.copy_records_to_table("clip_samples", records=_records(rows, m), columns=CLIP_COLS)
        elif kind == "delete":
            async with con.transaction():
                await con.execute("DELETE FROM clip_samples WHERE clip_id=$1", op[1])
                await con.execute("DELETE FROM clips WHERE id=$1", op[1])
        elif kind == "phrase":
            r = op[1]
            await con.execute(
                "INSERT INTO phrases (id, at, words, sentence) VALUES ($1,$2,$3,$4) ON CONFLICT (id) DO NOTHING",
                r["id"], _ts(r["at_ms"] / 1000), r["words"], r["sentence"],
            )
        elif kind == "detection":
            _, t, word, conf, accepted = op
            await con.execute("INSERT INTO detections (time, word, confidence, accepted) VALUES ($1,$2,$3,$4)", _ts(t), word, conf, accepted)
        elif kind == "tare":
            _, tid, r = op
            await con.execute(
                "INSERT INTO tares (id, session, at, ok, error, rotation_angle_deg, rotation, q_jaw, q_head, "
                "gravity_chin, gravity_head, n_chin, n_head) VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10,$11,$12,$13) "
                "ON CONFLICT (id) DO NOTHING",
                tid, r.get("session"), datetime.strptime(r["at"], "%Y-%m-%dT%H:%M:%S%z"), bool(r["ok"]), r.get("error"),
                r.get("rotation_angle_deg"), json.dumps(r["rotation"]) if r.get("rotation") else None,
                r.get("q_jaw"), r.get("q_head"), r.get("gravity_chin"), r.get("gravity_head"), r.get("n_chin"), r.get("n_head"),
            )
        elif kind == "rest":
            _, session, chin, head, rms = op
            await con.execute("INSERT INTO rest_poses (session, chin, head, activity_rms) VALUES ($1,$2,$3,$4)", session, chin, head, rms)
