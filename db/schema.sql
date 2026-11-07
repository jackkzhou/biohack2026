CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS mouth_sensor_data (
    time TIMESTAMPTZ NOT NULL,
    session_id TEXT NOT NULL,
    label TEXT NOT NULL,
    ax DOUBLE PRECISION NOT NULL,
    ay DOUBLE PRECISION NOT NULL,
    az DOUBLE PRECISION NOT NULL
);

SELECT create_hypertable('mouth_sensor_data', 'time');

CREATE INDEX IF NOT EXISTS mouth_sensor_data_session_time_idx
    ON mouth_sensor_data (session_id, time DESC);
