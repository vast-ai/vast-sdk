# vastai/serverless/cli.py
import os
import sys
import runpy
import subprocess


def main():
    if len(sys.argv) < 3:
        print("Usage: vast [deploy|serve|run|down] <script> [script-args...]", file=sys.stderr)
        sys.exit(1)

    mode = sys.argv[1]
    script_path = sys.argv[2]
    script_args = sys.argv[3:]

    # For `run`, bypass runpy and just call python3
    if mode == "run":
        cmd = ["python3", script_path] + script_args
        # Replace the current process with python3
        os.execvp("python3", cmd)

    # For all other modes, set env var and run via runpy
    os.environ["VAST_REMOTE_DISPATCH_MODE"] = mode

    # Rebuild sys.argv for the script we're about to run
    sys.argv = [script_path] + script_args

    # Execute the target script as __main__
    runpy.run_path(script_path, run_name="__main__")
