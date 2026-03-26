import inspect
import json
import os
from typing import Optional, Any, Callable, Awaitable, ParamSpec, Unpack, BinaryIO
from dataclasses import asdict, dataclass
from vastai.data import query
from vastai import AsyncClient
from vastai._base import _APIKEY_SENTINEL
from vastai.data.deployment import DeploymentConfig
from vastai.serverless.client import ManagedDeployment
from . import serialization
from .base import Deployment_, Config, Image, Autoscaling
from .utils import create_deployment_tarball, compute_deployment_hash
from os.path import getsize
import tempfile
import asyncio

# TODO: implement heartbeat, sync ready.

"""
Design:

Deployment has a root module.

Single file vs package deployments:
single file:
- from vastai.serverless import deployments
- deployment = deployments.Deployment(Image) 
- @deployment.remote...
- deployment.ensure_ready()

multi file: # skip for now
- @deployments.post_configure(name = name, image = image)
  def post_configure(deployment):
    deployment.ensure_ready()
  with post_configure:
    import mod1 
    import mod2

Serve:
- pull tarball
- read env, onstart, installs, submodules, name from config. # non essential
- inserts envs, runs installs, runs onstart. # non essential
- from vastai.serverless import deployments # essential
- import deployment # for package deployments require __init__.py to explicitly import remote function modules 
- my_deployment = deployments.get(name) # name is None if not manually set here. 
- serve(my_deployment) -> # essential
    async loop
    async serve(fname, args) -> serialize(my_deployment.registry[fname](deserialize(args)))
- fname relative to root module


Deploy:
- make endpoint args and get a tarball upload link
- make tarball and post it
- when functions are called:
  - convert fname to be relative to root module. 


Client -> Serve depends:
- POST -> app.endpoint/api/f serialized args -> returns f(args). 
  - serialization: root module agrees (i.e., `deployment` in serve is the root_module of deployment)
Serve depends:
- handle_f -> app.get(f_name)(args,kwargs)
    - app  <- deployments.get_app(name)
      - import deployment -> registers app and app functions
        - deployment module in ./deployment/ or ./deployment.py, untarred in tarball
    - name from config.json
    - environment:
        - On Image defined by from_image
        - with machine requirements set by requires_
        - envs set to envs (from config.json)
        - pip installs installed ""
        - apt gets installed ""
        - runs ran ""
        - vastai sdk installed (by bootstrap script)
    - where config.json, deployments are in tarball
      - bootstrap script knows deployment_id to grab from tarball with from S3

Serve <-> Webserver:
    - deployment_id in env
    - bootstrap script in onstart
    - Image, search params from deployments

Serve <-> Deploy:
    - Tarball as promised
Deploy <-> webserver

"""

DEBUG_DEPLOYMENT_TAR = os.environ.get("DEBUG_DEPLOYMENT_TAR")


@dataclass
class _FullDeployment:
    root_module: str
    deployment: ManagedDeployment


P = ParamSpec("P")


class Deployment(Deployment_):  # TODO: Async Context Manager compatible with client
    def __init__(
        self,
        api_key: str | object = _APIKEY_SENTINEL,
        ttl: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client = AsyncClient(api_key).serverless()
        self._image: Image | None = None
        self._autoscaling: Autoscaling | None = None
        self._ttl = ttl
        self._deployment_file: str | None = None
        self._inner: _FullDeployment | None = None

    def _into_deployment_config_and_tarball(self, tar_path: str) -> DeploymentConfig:
        # should error if _image, _autoscaling, _deployment_file, or any other field needed to calculate DeploymentConfig is None
        if not isinstance(self._image, Image):
            raise Exception(
                "Trying to deploy a deployment without an image configured."
            )
        if not isinstance(self.name, str):
            raise Exception(
                "Trying to deploy an unbound deployment. Have any remote functions been registered?"
            )
        if not isinstance(self._autoscaling, dict):
            raise Exception(
                "Trying to deploy a deployment without autoscaling configured."
            )
        hash, size = self._compute_hash_and_filesize_and_make_tar(
            tar_path, self.name, self._image
        )
        return DeploymentConfig(
            name=self.name,
            image=self._image.image_,
            file_hash=hash,
            file_size=size,
            tag=self.tag,
            search_params=json.dumps(self._image.requires_.query),
            storage=self._image.storage_,
            ttl=self._ttl,
            version_label=self.version_label,
            **self._image.docker_login,
            **self._autoscaling,
        )

    def configure_autoscaling(self, **kwargs: Unpack[Autoscaling]):
        if self._autoscaling is None:
            self._autoscaling = kwargs
        else:
            self._autoscaling.update(kwargs)

    def image(self, from_image: str, storage: int) -> Image:
        self._image = Image(from_image, storage)
        return self._image

    def _collate_config(self, checked_name: str, checked_image: Image) -> Config:
        # should error if any required fields are still None
        return Config(
            checked_name,
            checked_image.pip_installs_,
            checked_image.apt_gets_,
            list(checked_image.envs_.items()),
            checked_image.runs_,
        )

    def _compute_hash_and_filesize_and_make_tar(
        self, tar_path: str, checked_name: str, checked_image: Image
    ) -> tuple[str, int]:
        if not isinstance(self._deployment_file, str):
            raise Exception(
                "Trying to deploy a deployment not yet bound to a Python module. Have any remote functions been registered?"
            )
        config = self._collate_config(checked_name, checked_image)
        hash = compute_deployment_hash(
            config, self._deployment_file, checked_image.copies
        )
        create_deployment_tarball(
            tar_path, config, self._deployment_file, checked_image.copies
        )
        size = getsize(tar_path)
        return (hash, size)

    async def async_ensure_ready(self):
        if not isinstance(self.root_module, str):
            raise Exception(
                "Trying to deploy a deployment not yet bound to a Python module. Have any remote functions been registered?"
            )

        with tempfile.NamedTemporaryFile(
            delete_on_close=False
        ) as f:  # deletes at end of context manager instead of at f.close
            tar_path = f.name if not DEBUG_DEPLOYMENT_TAR else DEBUG_DEPLOYMENT_TAR
            f.close()  # _into_deployment_config_and_tarball will reopen it when it makes the tarball
            config = self._into_deployment_config_and_tarball(tar_path)
            deployment = await self.client.put_deployment(config)
            if deployment.needs_upload:
                await deployment.upload(tar_path)
            self._inner = _FullDeployment(self.root_module, deployment)

    def ensure_ready(self):
        asyncio.run(self.async_ensure_ready())

    async def _dispatch(self, f_name, globals, sig, args, kwargs) -> Any:
        if not isinstance(self._inner, _FullDeployment):
            raise Exception("Deployment is not ready. Call .ensure_ready() first!")
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return serialization.deserialize_unwrap_error(
            await self._inner.deployment.endpoint.request(
                "/remote/" + "/".join(f_name),
                {
                    k: serialization.serialize(v, self._inner.root_module)
                    for k, v in bound_args.arguments.items()
                },
            ),
            self._inner.root_module,
            globals,
        )

    def remote(
        self,
        f: Callable[P, Awaitable[Any]] | None = None,
        *,
        benchmark_dataset: list[dict] | None = None,
        benchmark_generator: Callable[[], dict] | None = None,
        benchmark_runs: int = 10,
    ) -> (
        Callable[P, Awaitable[Any]]
        | Callable[[Callable[P, Awaitable[Any]]], Callable[P, Awaitable[Any]]]
    ):
        def decorator(
            f: Callable[P, Awaitable[Any]], **_
        ) -> Callable[P, Awaitable[Any]]:
            f_rel_name = self.relativize(f)
            f_globals = f.__globals__
            sig = inspect.signature(f)

            def inner(*args: P.args, **kwargs: P.kwargs) -> Awaitable[Any]:
                return self._dispatch(f_rel_name, f_globals, sig, args, kwargs)

            return inner

        if f is not None:
            return decorator(f)
        else:
            return decorator
