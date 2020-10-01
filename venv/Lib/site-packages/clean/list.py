"""Listing config files."""

import click

from .config import Config

_use_meta_tag_text_dict = {True: '[META]', False: ''}  # Type: dict[bool, str]


def list_configs():
    """Show the path config lists to command line."""
    config = Config()
    glob_paths = config.list_glob_path()
    if len(glob_paths) == 0:
        click.echo(
            'No path settings. To add new setting, please use "clean add".')
    for i in enumerate(config.list_glob_path()):
        use_meta_tag_text = _use_meta_tag_text_dict[i[1]['use_meta_tag']]
        click.echo('[{}] {} => {} {}'.format(i[0], i[1]['glob'], i[1]['path'],
                                             use_meta_tag_text))
