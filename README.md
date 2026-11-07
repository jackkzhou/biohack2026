# biohack2026

## Tiger Data sensor ingestion

Create the TimescaleDB table and hypertable with `db/schema.sql`. Configure
`DATABASE_URL` with the Tiger Data PostgreSQL connection URL, then install
dependencies and start the ingestion API:

```bash
python -m pip install -r requirements.txt
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Send a batch to `POST /stream` with timezone-aware ISO 8601 timestamps:

```json
{
  "session_id": "session-001",
  "label": "hello",
  "samples": [
    {
      "time": "2026-11-07T18:35:00Z",
      "ax": 0.12,
      "ay": -0.03,
      "az": 9.81
    }
  ]
}
```

`ml.db_loader.load_session_data(db_url, session_id, window_size)` returns an
`(X, y)` tuple. `X` has shape `(windows, window_size, 3)` for
`fit_random_forest`, `fit_minirocket_ridge`, and the other time-series models;
windows overlap by one row and do not cross label changes.

## Time-series models

Each model can be imported and tested independently. The modules share
`ml/time_series_utils.py` for input validation and reusable feature helpers:

- `ml/dtw.py`
- `ml/random_forest.py`
- `ml/minirocket.py`
- `ml/cnn.py`

For Colab, upload the chosen model file and `time_series_utils.py` into the same
directory, then import that model (for example, `from random_forest import
fit_random_forest`). Existing imports from
`ml.time_series_classification` remain supported.