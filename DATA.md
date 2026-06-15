# Data

Full experiments use frozen Yahoo Finance adjusted-close snapshots from 2015
through 2023 for SPY, AAPL, MSFT, JPM, XOM, and JNJ. CSV files are stored under
`data/` and each run records SHA-256 checksums in `data_manifest.json`.

If a snapshot is missing, the runner can download it through `yfinance`, but
committed result reports are based on the frozen local files. Tests and smoke
runs use deterministic synthetic returns and do not require network access.
