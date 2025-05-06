from pathlib import Path

def main(week: str):
    readme = Path(__file__).parent/ week / "README.md"
    print(f"\n===== {week.upper()} INFO =====\n")
    print(readme.read_text() if readme.exists() else f"Chưa có README {week.upper()}")
