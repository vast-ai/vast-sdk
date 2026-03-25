import os

if os.environ.get("IS_DEPLOYMENT", False):
    from .deploy import Deployment
else:
    from .serve import Deployment
