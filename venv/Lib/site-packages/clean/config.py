"""Config file manager."""

import json
import os
from pathlib import Path

import click

from .updates.updator import need_update
from .updates.updator import update_config


class NoConfigFileException(Exception):
    """this exception throws If the config file found."""

    pass


def is_valid_glob_path(glob_and_path):
    """Check config file is valid format."""
    if 'path' not in glob_and_path:
        return False
    if 'glob' not in glob_and_path:
        return False
    return True


def get_config_path() -> Path:
    """Get config file path by environment variable or default path."""
    config_file_name = '.cleanrc'
    env_config_raw_path = os.getenv('CLEANRC_PATH')
    if env_config_raw_path is None:
        default_config_path = Path.home() / config_file_name
    else:
        default_config_path = Path(env_config_raw_path)
        if default_config_path.is_dir():
            default_config_path /= config_file_name
        if not default_config_path.is_file():
            raise NoConfigFileException('{}'.format(str(default_config_path)))
    return default_config_path


class Config(object):
    """Config file manager class.

    Returns:
        Config -- config file instance

    """

    def __init__(self, config_path: Path = None):
        """Initialize config class.

        Keyword Arguments:
            config_path {Path} -- set config file path
                                  (default: {default_config_path})
        """
        if config_path is None:
            config_path = get_config_path()
        self.config_path = config_path
        if not self.config_path.is_file():
            if self.config_path.exists():
                click.echo(
                    'Can\'t create file. Same name something is exist. ' +
                    'Please check your home\'s {}.'.format(str(config_path)))
                exit(1)
            self._create_new_config_file()

        self._load_file()

    def add_glob_path(self, glob: str, path: str,
                      enable_meta_tag: bool = True) -> bool:
        """Add new glob path to config file."""
        if self._is_contain_same_config(glob, path):
            return False
        self.config['path'].append({
            'glob': glob,
            'path': path,
            'use_meta_tag': enable_meta_tag
        })
        self._save_file()
        return True

    def _is_contain_same_config(self, glob: str, path: str) -> bool:
        return any(x['path'] == path and x['glob'] == glob
                   for x in self.config['path'])

    def delete_glob_path(self, id: int) -> dict:
        """Delete registered glob and path by id.

        Arguments:
            id {int} -- the glob and path's id which you want to delete.

        Returns:
            {{'glob': string, 'path': string}|None} -- the setting you destroy.

        """
        # 配列が空でないかどうかチェック
        if not self.config['path']:
            click.echo('There is no path settings. ' +
                       'Please add a path setting by "add" command.')
            return None
        # 配列の添え字が存在するかどうかチェック
        if 0 > id:
            click.echo(
                'Please input 0 or positive id. The max id is {}.'.format(
                    len(self.config['path'])))
        if len(self.config['path']) <= id:
            click.echo('The id is too big. Please input 0 <= id < {}.'.format(
                len(self.config['path'])))
            return None

        deleted_path = self.config['path'].pop(id)
        self._save_file()
        return deleted_path

    def list_glob_path(self) -> list:
        """Return a list of path configs."""
        return [i for i in self.config['path'] if is_valid_glob_path(i)]

    def _save_file(self):
        with self.config_path.open(mode='w', encoding='utf_8') as f:
            f.write(json.dumps(self.config))

    def _create_new_config_file(self):
        with self.config_path.open(mode='w', encoding='utf_8') as f:
            self.config = {'path': []}
            f.write(json.dumps(self.config))

    def get_config(self) -> dict:
        """Get config dictionary."""
        return self.config

    def _back_up_file(self):
        with self.config_path.open(encoding='utf_8') as f:
            with (self.config_path.parent /
                  (self.config_path.name + '.bk')).open(
                      mode='w', encoding='utf_8') as g:
                g.write(f.read())

    def _load_file(self):
        with self.config_path.open(encoding='utf_8') as f:
            config_text = f.read()
        self.config = json.loads(config_text)
        if need_update(self.config):
            self._back_up_file()
            update_config(self.config)
            self._save_file()
