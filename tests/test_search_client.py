import pytest

from mcp_indexer.search_client import split_query


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("cat:Physics", ("cat:Physics", None)),
        (
            "cat:Physics AND title:Quantum => Explain quantum physics => simply",
            ("cat:Physics AND title:Quantum", "Explain quantum physics => simply"),
        ),
        ("cat:Physics => ", ("cat:Physics", None)),
    ],
)
def test_split_query(line, expected):
    assert split_query(line) == expected
