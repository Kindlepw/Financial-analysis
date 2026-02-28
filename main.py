#!/usr/bin/env python3
"""
S&P 500 Discounted Cash Flow Valuation Tool
============================================

Usage examples
--------------
Run the full S&P 500 analysis and save results to a CSV file:

    python main.py

Analyse a custom list of tickers:

    python main.py --tickers AAPL MSFT GOOGL AMZN

Save results to a specific CSV path:

    python main.py --output results/sp500_valuation.csv

Enable verbose logging:

    python main.py --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

from src.valuation import print_summary, run_valuation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S&P 500 DCF valuation: intrinsic value vs. market price.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help=(
            "Space-separated list of tickers to analyse.  "
            "Defaults to the full S&P 500 list."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default="sp500_dcf_valuation.csv",
        help="Output CSV file path (default: sp500_dcf_valuation.csv).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="Delay in seconds between Yahoo Finance API calls (default: 0.5).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Number of top Buy/Sell companies to highlight in summary (default: 20).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = run_valuation(
        tickers=args.tickers,
        output_csv=str(output_path),
        delay=args.delay,
    )

    if df.empty:
        logging.error("Valuation produced no results.  Check network connectivity.")
        return 1

    print_summary(df, top_n=args.top)
    print(f"\nFull results saved to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
