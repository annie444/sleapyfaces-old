# SPDX-FileCopyrightText: 2022-present Annie Ehler <annie.ehler.4@gmail.com>
#
# SPDX-License-Identifier: MIT
import click

from ..__about__ import __version__


@click.group(context_settings={'help_option_names': ['-h', '--help']}, invoke_without_command=True)
@click.version_option(version=__version__, prog_name='sleapyfaces')
@click.pass_context
def sleapyfaces(ctx: click.Context):
    click.echo('Hello world!')
