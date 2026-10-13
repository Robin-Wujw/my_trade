"""Stable CLI for freezing inputs and independently audited offline replay."""
from stock_research.backtesting.frozen_replay import main
from stock_research.backtesting import frozen_replay as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
