#!/usr/bin/env python3
import argparse

def main():
    parser = argparse.ArgumentParser(description="Serverless Remote Dispatch")

    parser.add_argument("mode", choices=["deploy", "serve", "run"])
    parser.add_argument("--remote-dispatch-file-path", default="./endpoint.py", type=str)
    args = parser.parse_args()

    if args.mode == "deploy":
        pass
    if args.mode == "serve":
        pass
    if args.mode == "run":
        pass

main()
