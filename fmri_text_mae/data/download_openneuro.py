#!/usr/bin/env python3
"""Placeholder for raw OpenNeuro downloads.

The current experiments consume the same preprocessed Huth files already used
elsewhere in this repository (`data_train/train_response`, TextGrids, and
`respdict.json`). Add raw download/conversion steps here only if the local
preprocessed dataset is absent.
"""

if __name__ == "__main__":
    raise SystemExit(
        "No download step is required for the existing repo layout. "
        "Use fmri_text_mae/data/make_windows.py after data_train is present."
    )
