import argparse
import pandas as pd
import yfinance as yf
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch stock price data.')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol (e.g., AAPL).')
    parser.add_argument('--start_date', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, default=None, help='End date (YYYY-MM-DD).')
    parser.add_argument('--output_path', type=str, default='data/raw/', help='Output directory for data.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Fetch data
    data = yf.download(args.ticker, start=args.start_date, end=args.end_date)

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Save data to CSV
    output_file = os.path.join(args.output_path, f'{args.ticker}.csv')
    data.to_csv(output_file)

    print(f"Data for {args.ticker} saved to {output_file}")


if __name__ == '__main__':
    main()