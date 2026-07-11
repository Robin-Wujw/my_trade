import pandas as pd
import pytest

from stock_research.api.schema import rename_columns_strict


def test_strict_aliases_accept_harmless_column_variants():
    frame = pd.DataFrame({"Stock_Code": ["000001"], " 股票 名称 ": ["平安银行"]})

    result = rename_columns_strict(
        frame,
        {"code": ("stock code",), "name": ("股票名称",)},
        label="sample",
    )

    assert result[["code", "name"]].to_dict("records") == [
        {"code": "000001", "name": "平安银行"}
    ]


def test_strict_aliases_do_not_guess_by_column_position():
    frame = pd.DataFrame({"unexpected": ["000001"], "other": ["平安银行"]})

    with pytest.raises(KeyError, match="missing required columns"):
        rename_columns_strict(
            frame,
            {"code": ("stock code",), "name": ("股票名称",)},
            label="sample",
        )


def test_strict_aliases_reject_ambiguous_matches():
    frame = pd.DataFrame({"code": ["000001"], "股票代码": ["000002"]})

    with pytest.raises(ValueError, match="ambiguous"):
        rename_columns_strict(frame, {"code": ("股票代码",)}, label="sample")
