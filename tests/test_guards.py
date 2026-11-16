"""The test guards in conftest.py actually catch what they claim to.

Exercised by feeding events to the audit hook directly, so no real file under
samples/ or data/ is ever opened by these tests either.
"""

from __future__ import annotations

import socket

import conftest
from conftest import PROJECT_ROOT, outbound


def test_phi_guard_matches_the_patient_data_folders():
    seen: list = []
    old = conftest._STATE["phi"]
    conftest._STATE["phi"] = seen
    try:
        conftest._audit("open", (str(PROJECT_ROOT / "samples" / "x.pdf"), "rb", 0))
        conftest._audit("os.scandir", (str(PROJECT_ROOT / "data" / "fixtures"),))
        conftest._audit("open", (str(PROJECT_ROOT / "config" / "extract.toml"), "rb", 0))
        conftest._audit("open", (str(PROJECT_ROOT / "database.txt"), "rb", 0))
    finally:
        conftest._STATE["phi"] = old
    assert [e for e, _ in seen] == ["open", "os.scandir"]


def test_network_guard_sees_sockets_and_flags_outbound(network):
    a, b = socket.socketpair()  # loopback plumbing, as asyncio uses on Windows
    a.close()
    b.close()
    assert outbound(network) == []
    assert outbound([("socket.connect", (("127.0.0.1", conftest.OLLAMA_PORT),))])
    assert outbound([("socket.connect", (("10.0.0.5", 443),))])
    assert outbound([("urllib.Request", ("http://127.0.0.1:11434/api/version", None))])
