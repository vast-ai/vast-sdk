#!/usr/bin/python
import os
import json
from vastai.serverless.remote.base import Config
from vastai.serverless.remote.serve import Deployment

if __name__ == "__main__":
    deployment_id = os.environ.get("DEPLOYMENT_ID")

    # TODO: download deployment tarball
    # Allow setting an environment variable to skip download and specify a local tarball to use for testing

    # TODO extract deployment tarball in current working directory, allowing absolute paths
    # We expect the deployment tarball to extract config.json into current working directory, as well as either deployment/ (if deployment is a package) or deployment.py

    def get_config(path: str) -> Config:
        with open("config.json") as f:
            return Config(**json.load(f))

    config: Config = get_config("config.json")
    deployment = Deployment.lookup(config.name)
    if deployment is None:
        raise Exception(f"Failed to lookup registered deployment: {config.name}")

    # (1) Export ENVs that config defines, both for us and our subprocesses; also save to /etc/environment as courtesy for debuggers
    # (2) Run apt gets from config; run pip installs from config
    # (3) run scripts from config; `sh -c` if string or execvp style if list of args

    worker = deployment.into_worker()
    worker.run()
