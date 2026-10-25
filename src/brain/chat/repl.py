"""Terminal loop around ``ChatSession`` (rich, streaming)."""

from __future__ import annotations

from collections.abc import Callable

from rich.console import Console
from rich.markup import escape

from brain.chat.session import ChatSession


def run_repl(
    session: ChatSession,
    console: Console,
    *,
    read: Callable[[], str] | None = None,
) -> None:
    """Read lines until ``/quit`` or EOF; stream answers as they arrive."""
    read = read or (lambda: console.input("[bold cyan]you ›[/] "))
    console.print("[dim]Read-only chat. /mem shows retrieved memory, /quit leaves.[/]")
    while True:
        try:
            text = read()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text.strip():
            continue
        if (result := session.command(text)) is not None:
            console.print(result.output, markup=False, highlight=False)
            if result.quit:
                break
            continue
        console.print("[bold green]brain ›[/] ", end="")
        try:
            for delta in session.turn(text):
                console.print(delta, end="", markup=False, highlight=False)
        except KeyboardInterrupt:
            console.print(" [dim](interrupted)[/]", end="")
        except Exception as err:  # keep the session alive if the model server hiccups
            console.print(f"\n[red]error:[/] {escape(str(err))}")
        console.print()
