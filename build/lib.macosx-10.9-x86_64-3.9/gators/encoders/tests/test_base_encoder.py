# License: Apache-2.0
from gators.encoders import WOEEncoder
import pytest


def test_init():
    with pytest.raises(TypeError):
        WOEEncoder(dtype=str)
