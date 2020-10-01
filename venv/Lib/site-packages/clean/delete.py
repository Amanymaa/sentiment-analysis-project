"""Delete file move setting(s)."""

from .config import Config


def delete_config(id: int):
    """Delete config by id."""
    config = Config()
    return config.delete_glob_path(id)
