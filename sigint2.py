import asyncio, sys, time
sys.path.insert(0, "/Users/vasiliy.makarov/pr/finmath-distill-bench")
import benchmark_model as bm
import signal
signal.signal(signal.SIGINT, bm._sigint_handler)
async def slow_cleanup():
    try:
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        # simulate a lengthy graceful cleanup that a 2nd Ctrl-C must cut short
        print("cancelling... (cleanup would take a while)", flush=True)
        await asyncio.sleep(60)
if __name__ == "__main__":
    try:
        asyncio.run(slow_cleanup())
    except KeyboardInterrupt:
        print("graceful interrupt received", flush=True)
        time.sleep(60)  # more "cleanup"
