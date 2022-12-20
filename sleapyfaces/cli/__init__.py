# SPDX-FileCopyrightText: 2022-present Annie Ehler <annie.ehler.4@gmail.com>
#
# SPDX-License-Identifier: MIT
import click

from ..__about__ import __version__


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(version=__version__, prog_name="sleapyfaces")
@click.option("--count", default=1, help="Number of greetings.")
@click.option("--name", prompt="Your name", help="The person to greet.")
@click.pass_context
def hello(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo(f"Hello {name}!")
