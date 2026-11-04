#!/usr/bin/env python3
"""Package source without macOS metadata, then install a versioned VPS release."""
import argparse
import datetime as dt
from pathlib import Path
import subprocess
import tarfile
import tempfile

SOURCE = Path(__file__).resolve().parent
FILES = ["web", "deploy", "requirements-web.txt", "web_common.py", "web_service.py",
         "web_sync.py", "web_tasks.py", "fetcher.py", "today_pick.py",
         "backtest_qvix.py", "qvix_calc.py", "qvix_core.py"]


def exclude_metadata(info):
    if any(p.startswith(".") or p == "__pycache__" for p in Path(info.name).parts):
        return None
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="root@192.243.114.13")
    args = ap.parse_args()
    version = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    remote = "/opt/fund-web/releases/" + version
    with tempfile.TemporaryDirectory(prefix="fund-web-deploy-") as tmp:
        archive = Path(tmp) / "fund-web.tar.gz"
        with tarfile.open(archive, "w:gz") as out:
            for filename in FILES:
                out.add(SOURCE / filename, arcname=filename, filter=exclude_metadata)
        subprocess.run(["scp", str(archive), args.host + ":/tmp/fund-web-" + version + ".tar.gz"], check=True)
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", args.host,
                        f"mkdir -p {remote} && tar -xzf /tmp/fund-web-{version}.tar.gz -C {remote} "
                        f"&& bash {remote}/deploy/install_web.sh {remote}"], check=True)
