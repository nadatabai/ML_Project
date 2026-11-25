import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import calculate_average  # noqa: E402


def test_calculate_average():
    result = calculate_average([10, 12, 14])
    assert result == 12


def test_calculate_average_with_different_values():
    result = calculate_average([10, 12, 18])
    assert abs(result - 13.333333333333334) < 0.000001
