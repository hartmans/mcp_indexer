import argparse

from mcp_indexer.server import _build_parser, _create_server, _transport_for_args


def test_build_parser_defaults_host_to_localhost_and_stdio():
    args = _build_parser().parse_args(["--config", "config.toml"])

    assert args.config == ["config.toml"]
    assert args.host == "localhost"
    assert args.port is None


def test_build_parser_accepts_host_and_port():
    args = _build_parser().parse_args(
        ["--config", "config.toml", "--host", "0.0.0.0", "--port", "8123"]
    )

    assert args.host == "0.0.0.0"
    assert args.port == 8123


def test_create_server_uses_http_settings_only_when_port_is_specified():
    stdio_server = _create_server()
    http_server = _create_server(host="localhost", port=8123)

    assert stdio_server.settings.host == "127.0.0.1"
    assert stdio_server.settings.port == 8000
    assert http_server.settings.host == "localhost"
    assert http_server.settings.port == 8123


def test_transport_switches_to_streamable_http_when_port_is_specified():
    assert _transport_for_args(argparse.Namespace(port=None)) == "stdio"
    assert _transport_for_args(argparse.Namespace(port=8123)) == "streamable-http"
