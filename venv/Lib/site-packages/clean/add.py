"""Add new file move setting."""

from .config import Config


def add_new_config(glob: str, path: str, is_regexp: bool = False) -> bool:
    """Add new path config.

    Arguments:
        glob {str} -- glob or regular expression text
        path {str} -- the path where the matched files move to

    Keyword Arguments:
        is_regexp {bool} -- if this parameter sets true,
                            the 'glob' parameter works as
                            regular expression (default: {False})

    Returns:
        bool -- is successful

    """
    config = Config()
    return config.add_glob_path(glob, path)
