"""Launch AimVision desktop GUI."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    subprocess.run([sys.executable, "src/gui/main_app.py"], check=True)


if __name__ == "__main__":
    main()
