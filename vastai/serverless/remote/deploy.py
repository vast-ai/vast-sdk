import inspect
from typing import Optional, Any, Callable, Awaitable
from dataclasses import dataclass
from vastai.data import query
from . import serialization
from ..client import client, endpoint
from pathlib import Path
import tarfile
import time
from io import BytesIO
from .base import Deployment_, Config

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


def tar_info_and_file_from_str(  # deletes owner id/name, as we want files to be owned by the extractor always
    path: str, contents: str, set_executable=False, permissions_override=None
) -> tuple[tarfile.TarInfo, BytesIO]:
    bytes = contents.encode("utf-8")
    size = len(bytes)
    info = tarfile.TarInfo(path)
    permissions = permissions_override if permissions_override is not None else 0o644
    if set_executable:
        permissions = permissions & 0o100
    info.mode = permissions
    info.size = size
    info.mtime = time.time()

    return (info, BytesIO(bytes))


def tar_add_string(
    tarball: tarfile.TarFile,
    path: str,
    contents: str,
    set_executable=False,
    permissions_override=None,
):
    header, membuf = tar_info_and_file_from_str(
        path, contents, set_executable, permissions_override
    )
    tarball.addfile(header, membuf)


def _cache_dir() -> Path: ...


class Image:
    def __init__(self, from_image: str):
        self.image_ = from_image
        self.envs_ = {}
        self.runs_ = []
        self.pip_installs_: list[str] = []
        self.apt_gets_: list[str] = []
        self.requires_ = query.Query.search_defaults()

    def post_deployment_kwargs(self) -> dict[str, str]:
        ret = {
            "image": self.image_,
            "search_params": self.requires_.query,
        }
        return ret

    def pip_install(self, *args: str) -> "Image":
        for arg in args:
            self.pip_installs_.append(arg)
        return self

    def apt_get(self, *args: str) -> "Image":
        for arg in args:
            self.apt_gets_.append(arg)
        return self

    def run_script(self, script) -> "Image":
        self.runs_.append(script)
        return self

    def run_cmd(self, *args) -> "Image":
        self.runs_.append(args)
        return self

    def env(self, **kwargs: str) -> "Image":
        for k, v in kwargs.items():
            self.envs_[k] = v
        return self

    def require(self, *args: query.Query) -> "Image":
        for arg in args:
            self.requires_.extend(arg)
        return self

    def copy(self, src: str, dst: Optional[str] = None) -> "Image": ...

    def compile_config(
        self,
    ) -> str: ...  # for now, stuff all installs and env exports into onstart.

    def make_tarball(
        self, name: str
    ) -> tuple[str, str, int]:  # (filename, sha256sum, filesize)
        # TODO: implement caching logic so we skip disk writes if unnecessary.
        # TODO support compression?
        onstart = self.compile_onstart()
        cache_dir = _cache_dir()
        tarpath = cache_dir / (name + ".tar")
        tarball = tarfile.open(tarpath, mode="w")
        tar_add_string(tarball, "onstart.sh", onstart, set_executable=True)


@dataclass
class Autoscaling:
    # Scaling params
    cold_workers: Optional[int] = None
    max_workers: Optional[int] = None
    min_load: Optional[int] = None
    min_cold_load: Optional[int] = None
    target_util: Optional[float] = None
    cold_mult: Optional[int] = None
    max_queue_time: Optional[float] = None
    target_queue_time: Optional[float] = None
    inactivity_timeout: Optional[int] = None
    autoscaler_instance: Optional[str] = None


class _PartialDeployment:  # Nullable fields, lacks info returned by api.deployments.PUT
    def __init__(
        self,
        image: Image,
        name: Optional[str] = None,  # Module name by default
        tag: str = "default",
        version_label: Optional[str] = None,
    ):
        self.name = name  # name for webserver, goes to module name by default
        self.image = image
        self.version_label = version_label
        self.tag = tag
        self.autoscaling: Optional[Autoscaling] = None
        self.root_module: str | None = None
        self.file = None


@dataclass
class _FullDeployment:
    name: str
    image: Image
    version_label: Optional[str]
    tag: str
    autoscaling: Autoscaling
    root_module: str
    endpoint: endpoint.Endpoint


class Deployment(Deployment_["Deployment"]):
    def __init__(
        self,
        image: Image,
        name: Optional[str] = None,  # Module name by default
        tag: str = "default",
        version_label: Optional[str] = None,
    ):
        self.inner_ = _PartialDeployment(image, name, tag, version_label)

    def _compute_file_hash_and_size(self, path: str) -> tuple[str, int]: ...

    def _post_deployment(
        self, api_key: str, file_hash: str, file_size: int
    ) -> dict: ...

    def ensure_ready(self, ttl: Optional[int] = None): ...

    def _check_ready(self) -> _FullDeployment:
        if not isinstance(self.inner_, _FullDeployment):
            raise TypeError("Deployment is not ready!")
        return self.inner_

    async def dispatch_(
        self, relative_name: tuple[str], sig, args, kwargs, globals
    ) -> Any:
        inner_ = self._check_ready()
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        payload = {
            "input": {
                k: serialization.serialize(v, inner_.root_module)
                for k, v in bound_args.arguments.items()
            }
        }
        response = await inner_.endpoint.request("/".join(relative_name), payload)
        return serialization.deserialize_unwrap_error(
            response["response"], inner_.root_module, globals
        )  # TODO: what is right globals???

    # Will override in future for a multi-module deployment subclass
    # For now, require all remote functions be in the same module.
    def relativize(self, f: Callable[..., Any]) -> tuple[str]:
        mod = f.__module__
        name = f.__name__
        file = f.__globals__.get("__file__")
        if self.inner_.root_module is None:
            self.inner_.root_module = mod
            if file is None:
                raise Exception(
                    f"Cowardly refusing to deploy function {name} defined in your interactive Python session! We need it to be defined in a .py file to deploy from."
                )
            self.file = file
        if f.__module__ != self.inner_.root_module:
            raise Exception(
                f"Using a single-file deployment object to deploy remote functions from multiple .py files! Remote function {name} is defined in {file} but we're already trying to deploy from {self.file}."
            )
        return (name,)

    def remote(self, f: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        relative_name = self.relativize(f)
        sig = inspect.signature(f)
        globals = f.__globals__

        def inner(*args, **kwargs):
            return self.dispatch_(relative_name, sig, args, kwargs, globals)

        return inner

    # TODO this is probably serve time only?
    def context(self, *args, **kwargs) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            return cls

        return decorator
