import socket

import pytest

from paralic.__main__ import _allowed_hosts, _pick_port, _port_free


def test_busy_port_is_skipped():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert not _port_free("127.0.0.1", busy)
        assert _pick_port("127.0.0.1", busy) != busy


def test_ipv6_loopback_is_supported():
    if not socket.has_ipv6:
        pytest.skip("no IPv6")
    try:
        with socket.socket(socket.AF_INET6) as s:
            s.bind(("::1", 0))
    except OSError:
        pytest.skip("IPv6 loopback unavailable")
    port = _pick_port("::1", 20000)
    assert _port_free("::1", port)


def test_unresolvable_host_exits_with_message():
    with pytest.raises(SystemExit) as exc:
        _port_free("no-such-host.invalid", 8000)
    assert "Cannot use host" in str(exc.value)


def test_allowed_hosts():
    assert _allowed_hosts("0.0.0.0") == "*"
    assert "localhost" in _allowed_hosts("127.0.0.1")
    assert "192.168.1.5" in _allowed_hosts("192.168.1.5")
    assert "::1" in _allowed_hosts("[::1]")
