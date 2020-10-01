"""Path utility functions."""
from os import utime
from pathlib import Path


def rm_recursive(path: Path):
    """Delete directory or file recursive."""
    if path.exists():
        if path.is_file():
            path.unlink()
            return
        if path.is_dir():
            for file in path.glob('*'):
                rm_recursive(file)
            path.rmdir()
