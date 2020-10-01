"""Manage config file updating."""
import inspect
from importlib import import_module
from pathlib import Path


def _version_value(version_name):
    version_value_texts = version_name.lstrip('V').split('_')
    version_values = [int(v) for v in version_value_texts]
    return tuple(version_values)


def _version_tuple_to_value(version_tuple):
    return 'V{}_{}_{}'.format(version_tuple[0], version_tuple[1],
                              version_tuple[2])


def _get_version_classes():
    version_dir = (Path(__file__).parent / 'versions').resolve()
    version_files = [x for x in version_dir.glob('v*_*_*.py')]
    version_modules = [
        import_module('clean.updates.versions.' + x.stem)
        for x in version_files
    ]
    version_classes = [
        inspect.getmembers(x, inspect.isclass)[0] for x in version_modules
    ]
    return sorted([(_version_value(x[0]), x[1]) for x in version_classes])


def need_update(config: dict) -> bool:
    """Check that the config file needs to update."""
    if 'version' not in config:
        return True
    version_classes = _get_version_classes()
    config_version = _version_value(config['version'])
    return config_version < version_classes[-1][0]


def update_config(config: dict):
    """Update the passed config dictionary."""
    version_classes = _get_version_classes()
    if 'version' not in config:
        config_version = (0, 1, 0)
    else:
        config_version = _version_value(config['version'])
    apply_classes = [x[1] for x in version_classes if config_version < x[0]]
    for version in apply_classes:
        config = version().up(config)
    config['version'] = _version_tuple_to_value(version_classes[-1][0])
    return config


if __name__ == '__main__':
    print(_get_version_classes())
