import os

from .base import DockerLogin, Image, Autoscaling

if os.environ.get("IS_DEPLOYMENT", False):
    from .deploy import Deployment
else:
    from .serve import Deployment
