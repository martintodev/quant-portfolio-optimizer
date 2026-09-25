from unittest.mock import patch
import pandas as pd
from qpo.data.fetcher import get_price_history


def test_get_price_history():
    fake_df = pd.DataFrame({"Close": [100, 101, 102]})

    with patch("qpo.data.fetcher.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = fake_df

        result = get_price_history(["AAPL"], "2023-01-01", "2023-12-31")

        # your assertions here
        assert "AAPL" in result
        assert result["AAPL"].equals(fake_df)
        mock_ticker.assert_called_with("AAPL")
