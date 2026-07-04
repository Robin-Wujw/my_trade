# -*- coding: utf-8 -*-
"""Small command-line bridge for unattended pipeline alerts."""
import argparse

from stock_research.api.pushplus import send_pushplus


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args(argv)
    ok = send_pushplus(args.title, f"<p>{args.message}</p>")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
