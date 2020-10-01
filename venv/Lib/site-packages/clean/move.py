"""Run file moving."""

from datetime import datetime
import glob
from os import stat
from os import stat_result
from pathlib import Path

import click

from .config import Config


def _replace_meta_tag(path: str, file: Path):
    file_stat = stat(str(file.resolve()))  # Type: stat_result
    file_update_time = datetime.fromtimestamp(
        file_stat.st_mtime)  # Type: datetime
    path = path.replace('<YEAR>', '{0:04d}'.format(file_update_time.year))
    path = path.replace('<MONTH>', '{0:02d}'.format(file_update_time.month))
    path = path.replace('<DAY>', '{0:02d}'.format(file_update_time.day))
    path = path.replace('<HOUR>', '{0:02d}'.format(file_update_time.hour))
    path = path.replace('<MINUTE>', '{0:02d}'.format(file_update_time.minute))
    path = path.replace('<SECOND>', '{0:02d}'.format(file_update_time.second))
    return path


def _resolve_move_into(path: str, file: Path, use_meta_tag: bool = False):
    if use_meta_tag:
        move_into = Path(_replace_meta_tag(path, file))
    else:
        move_into = Path(path)
    if move_into.exists():
        if not move_into.is_dir():
            click.echo(
                '{} already exists. The file move setting will ignore.'.format(
                    str(move_into)))
            return False
    else:
        move_into.mkdir(parents=True)
    return move_into


def move(is_fake: bool = True,
         is_silent: bool = False,
         is_recursive: bool = False):
    """Move files as config setting."""
    config = Config()
    for i in config.list_glob_path():
        file_lists = glob.glob(i['glob'], recursive=is_recursive)
        for file in [Path(x) for x in file_lists]:
            file_name = file.name
            move_into = _resolve_move_into(i['path'], file, i['use_meta_tag'])
            move_to = move_into / file_name
            if move_to.exists():
                click.echo('{} already exists. The file will not move.'.format(
                    str(move_to)))
                continue
            if not is_silent:
                click.echo('{} => {}'.format(str(file), str(move_to)))
            if is_fake:
                continue
            file.rename(move_to)
