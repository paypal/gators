# License: Apache-2.0
import pytest

from gators.encoders import WOEEncoder


def test_init():
    with pytest.raises(TypeError):
        WOEEncoder(dtype=str)
