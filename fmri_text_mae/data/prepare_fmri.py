#!/usr/bin/env python3
"""Thin entry point for fMRI preparation.

For this repo's Huth layout, `make_windows.py` loads preprocessed `.hf5`
responses directly and applies train-only z-scoring. Keep dataset-specific
response conversion outside this file unless raw downloads are added later.
"""

from make_windows import main


if __name__ == "__main__":
    main()
