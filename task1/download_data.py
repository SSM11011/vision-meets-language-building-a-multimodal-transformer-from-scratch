"""Download Tiny Shakespeare (Karpathy's char-rnn input) to task1/input.txt.

Usage:
    python task1/download_data.py
"""
import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = os.path.join(os.path.dirname(__file__), "input.txt")


def main() -> None:
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        print(f"Already present: {OUT} ({os.path.getsize(OUT)} bytes)")
        return
    print(f"Downloading Tiny Shakespeare from {URL}")
    try:
        urllib.request.urlretrieve(URL, OUT)
    except Exception as exc:  # noqa: BLE001 - surface any network/file error clearly
        raise SystemExit(
            f"Failed to download dataset: {exc}\n"
            f"Manually download {URL} and save it as {OUT}."
        )
    print(f"Saved {os.path.getsize(OUT)} bytes to {OUT}")


if __name__ == "__main__":
    main()
