#!/usr/bin/env python3
"""Transcript preparation uses existing Huth TextGrid utilities.

`make_windows.py` calls `decoding/utils_stim.py::get_story_wordseqs`, which
parses TextGrids and produces word/onset sequences aligned to story TR files.
"""

from make_windows import main


if __name__ == "__main__":
    main()
