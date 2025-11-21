# vastai/serverless/cli.py
import os
import sys
import runpy


def main():
    if len(sys.argv) < 3:
        print("Usage: vast [deploy|serve|run|down] <script> [script-args...]", file=sys.stderr)
        sys.exit(1)

    mode = sys.argv[1]
    script_path = sys.argv[2]

    # Set the env var based on the first arg
    os.environ["VAST_REMOTE_DISPATCH_MODE"] = mode

    # Rebuild sys.argv for the script we're about to run
    # so inside endpoint.py, sys.argv[0] is endpoint.py, etc.
    sys.argv = [script_path] + sys.argv[3:]

    # Execute the target script as __main__
    runpy.run_path(script_path, run_name="__main__")
