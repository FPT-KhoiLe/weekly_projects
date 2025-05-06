# src/weekly_projects/cli.py
import argparse, sys
import importlib

def import_week_module(week: str, mod: str):
    """Helper to import e.g. weekly_projects.week1.train"""
    return importlib.import_module(f"weekly_projects.{week}.{mod}")

def week_info(week: str):
    """Helper to import e.g. weekly_projects.week1.info"""
    return importlib.import_module("weekly_projects.weekly_info")

def main():
    parser = argparse.ArgumentParser(
        prog="weekly-projects",
        description="CLI cho kho dự án Weekly AI"
    )

    subp = parser.add_subparsers(dest="week", required=True)

    # ---- Week 1 ----
    w1 = subp.add_parser("week1", help="Tuần 1 – FFN trên MNIST")
    w2 = subp.add_parser("week2", help="Tuần 2 – SimCLR (placeholder)")

    w1_sub = w1.add_subparsers(dest="cmd", required=True, help="Module to run")
    w2_sub = w2.add_subparsers(dest="cmd", required=True, help="Module to run")

    w1_sub.add_parser("info", help="Show info about Week 1")
    tr1 = w1_sub.add_parser("train", help="Train FFN")

    tr1.add_argument("--hidden", nargs="+", type=int, default=[256,256], help="Hidden layers. Ex: 256 256")
    tr1.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    tr1.add_argument("--batch", type=int, default=64, help="Batch size")
    tr1.add_argument("--lr",    type=float, default=1e-3, help="Learning rate")
    tr1.add_argument("--datadir", type=str, help="Path to MNIST data")

    # ---- Week 2 (placeholder) ----

    args = parser.parse_args()

    # Dispatch
    if args.week == "week1":
        if args.cmd == "info":
            week_info("week1").main(week="week1")
        elif args.cmd == "train":
            import_week_module("week1", "train").main(args)
    else:
        parser.print_help()
        sys.exit(1)
