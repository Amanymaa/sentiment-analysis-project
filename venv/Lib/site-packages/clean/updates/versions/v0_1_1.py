"""0.1.1 Update."""
from .version import Version


def _update_path(path_config: dict) -> dict:
    if 'use_meta_tag' not in path_config:
        path_config['use_meta_tag'] = False
    return path_config


def _downgrade_path(path_config: dict) -> dict:
    if 'use_meta_tag' in path_config:
        del path_config['use_meta_tag']
    return path_config


class V0_1_1(Version):
    """Version 0.1.1 Update."""

    def up(self, config: dict) -> dict:
        """Upgrade config file."""
        config['path'] = [_update_path(x) for x in config['path']]
        return config

    def down(self, config: dict) -> dict:
        """Downgrade config file."""
        config['path'] = [_downgrade_path(x) for x in config['path']]
        return config
