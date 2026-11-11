"""Test-suite bootstrap and shared fixtures for ``lightning_grpo``.

The package is not installed into the environment used to run these tests, and the
``open_r1`` rewards live under ``src/``, so both locations are added to ``sys.path``
before collection. The implicit-encoding workaround keeps the suite runnable on Windows
machines whose ANSI code page is not UTF-8 (see :func:`_use_utf8_for_implicit_reads`).
"""

from __future__ import annotations

import io
import locale
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _use_utf8_for_implicit_reads() -> None:
    """Make text reads without an explicit encoding default to UTF-8.

    ``trl`` (imported through ``lightning_grpo.models``) loads its chat-template ``.jinja``
    files via ``Path.read_text()`` without an encoding, which raises ``UnicodeDecodeError``
    on an ANSI code page such as cp936. The files involved here are UTF-8, so widening the
    default keeps the suite runnable without requiring ``python -X utf8``.
    """

    if locale.getpreferredencoding(False).lower() in {"utf-8", "utf8"}:
        return
    original = io.text_encoding

    def utf8_text_encoding(encoding: str | None = None, stacklevel: int = 2) -> str:
        return "utf-8" if encoding is None else original(encoding, stacklevel)

    io.text_encoding = utf8_text_encoding


_use_utf8_for_implicit_reads()


@pytest.fixture
def rollout_batch() -> dict:
    """A minimal batch shaped exactly like the one the rollout coordinator emits."""

    from lightning_grpo.tests.support import make_rollout_batch

    return make_rollout_batch()


@pytest.fixture
def reward_manager():
    from lightning_grpo.tests.support import FakeRewardManager

    return FakeRewardManager()


@pytest.fixture
def sink():
    from lightning_grpo.tests.support import RecordingSink

    return RecordingSink()
