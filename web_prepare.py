"""Prepare today's indexed answers before switching a web release live."""
import logging
import time

from web_common import snapshot, today
from web_service import CODE_VERSION
from web_tasks import run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path, _ = snapshot()
    start = time.monotonic()
    try:
        run(dict(snapshot=str(path), code_version=CODE_VERSION, operation="prepare",
                 args={"asof": today().isoformat()}))
    except ValueError as exc:
        from today_pick import IncompleteNavError
        if not isinstance(exc, IncompleteNavError):
            raise
        # Activate the site so requests can show the data error. No cache was written.
        logging.error("今日候选预热中止：%s", exc)
    else:
        print(f"today_cache_ready elapsed={time.monotonic()-start:.2f}s", flush=True)
