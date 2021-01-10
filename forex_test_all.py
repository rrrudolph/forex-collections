import pytest
import pandas as pd
from get_data import UpdateDB
from ohlc_symbols import te_countries



# Tests for the UpdateDB methods

# Check all Trading Economics links are working
def test_te_request():
    for country in te_countries:
        assert UpdateDB._te_request(country) is not None


def test_finnhub_request():
    assert UpdateDB._finnhub_ohlc_request('VIXM', 15) is not None