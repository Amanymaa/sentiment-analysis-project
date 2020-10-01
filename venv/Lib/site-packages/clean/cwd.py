"""Cwd command."""

from pathlib import Path

import click


def show_cwd():
    """Show current directory."""
    click.echo(str(Path.cwd()))
