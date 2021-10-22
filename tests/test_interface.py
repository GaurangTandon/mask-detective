import pytest


def test_dummy():
    assert True


@pytest.mark.xfail
def test_failure():
    assert False
