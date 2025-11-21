#! /usr/bin/python3
import os
import sys
import requests
import subprocess

if __name__ == "__main__":
    vast_upload_url = os.environ["VAST_UPLOAD_URL"]
    vast_upload_auth_token = os.environ["VAST_UPLOAD_AUTH_TOKEN"]
    vast_download_url_base = os.environ["VAST_DOWNLOAD_URL"]
    try:
        deploy_script = sys.argv[1]
    except:
        print("1 positional argument expected: deploy script")
        exit(1)
    with open(deploy_script, 'r') as deploy_script_file:
        blob_id = requests.post(vast_upload_url, headers = {'Authorization': f'Bearer {vast_upload_auth_token}'}, data = deploy_script_file.read()).text
    subprocess.run(['python3', *sys.argv[1:]], env={'VAST_REMOTE_DISPATCH_MODE':'deploy', 'VAST_WORKER_SCRIPT_URL': vast_download_url_base.rstrip('/') + '/' + blob_id})
    
