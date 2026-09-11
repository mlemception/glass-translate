"""The stdin watcher polls the parent's pipe instead of parking a thread in a read."""
from __future__ import annotations

import io
import os
import sys
import threading
import time

import pytest

from glassrenderer import server


@pytest.fixture
def stdin_pipe(monkeypatch):
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb", buffering=0)
    monkeypatch.setattr(sys, "stdin", stream)
    yield write_fd
    try:
        os.close(write_fd)
    except OSError:
        pass
    stream.close()


def test_open_pipe_with_nothing_written_is_not_eof(stdin_pipe) -> None:
    assert server._stdin_at_eof() is False


def test_bytes_written_by_the_parent_are_drained_not_eof(stdin_pipe) -> None:
    os.write(stdin_pipe, b"ignored\n")
    assert server._stdin_at_eof() is False
    assert server._stdin_at_eof() is False  # drained, still open


def test_closed_write_end_is_eof(stdin_pipe) -> None:
    os.close(stdin_pipe)
    assert server._stdin_at_eof() is True


def test_missing_or_closed_stdin_counts_as_eof(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", None)
    assert server._stdin_at_eof() is True
    closed = io.BytesIO()
    closed.close()
    monkeypatch.setattr(sys, "stdin", closed)
    assert server._stdin_at_eof() is True


def test_watch_stdin_triggers_shutdown_only_after_eof(stdin_pipe, monkeypatch) -> None:
    monkeypatch.setattr(server, "STDIN_POLL_S", 0.01)
    fired = threading.Event()

    class _Httpd:
        def trigger_shutdown(self) -> None:
            fired.set()

    thread = server.watch_stdin(_Httpd())  # type: ignore[arg-type]
    assert thread.daemon
    time.sleep(0.1)
    assert not fired.is_set()
    os.close(stdin_pipe)
    assert fired.wait(2.0)
    thread.join(2.0)
    assert not thread.is_alive()


def test_poll_check_never_blocks(stdin_pipe) -> None:
    started = time.perf_counter()
    for _ in range(50):
        server._stdin_at_eof()
    assert time.perf_counter() - started < 1.0
