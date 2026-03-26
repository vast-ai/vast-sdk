import inspect
from typing import Optional, Any, Callable, Awaitable, ParamSpec, TypeVar
from dataclasses import asdict, dataclass
from vastai.data import query
from vastai import AsyncClient
from vastai._base import _APIKEY_SENTINEL
from vastai.data.deployment import DeploymentConfig
from vastai.serverless.client import ManagedDeployment
from . import serialization
from ..client import client, endpoint
from pathlib import Path
import tarfile
import time
from io import BytesIO
from .base import Deployment_, Config
from .utils import create_deployment_tarball, compute_deployment_hash()
from os.path import getsize

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


@dataclass
class DockerLogin:
    docker_login_user: str
    docker_login_pass: str
    docker_login_repo: str


class Image:
    def __init__(self, from_image: str):
        self.image_ = from_image
        self.envs_ : dict[str,str]= {}
        self.runs_ : list[str | tuple[str,...]]= []
        self.pip_installs_: list[str] = []
        self.apt_gets_: list[str] = []
        self.docker_login: DockerLogin | None = None
        self.requires_ = query.Query.search_defaults()
        self.storage_ = 50
        self.copies : list[tuple[str,str]] = []

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
        self.file : str | None = None


@dataclass
class _FullDeployment:
    root_module : str
    deployment : ManagedDeployment


P = ParamSpec("P")
class Deployment(Deployment_): #TODO: Async Context Manager compatible with client
    def __init__(
        self,
        api_key: str | object = _APIKEY_SENTINEL,
        ttl: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client = AsyncClient(api_key).serverless()
        self._image = None
        self._autoscaling = None
        self._ttl = ttl
        self._deployment_file = None
        self._inner : _FullDeployment | None = None

    def _into_deployment_config_and_tarball(self) -> tuple[DeploymentConfig,str]:
        # should error if _image, _autoscaling, _deployment_file, or any other field needed to calculate DeploymentConfig is None
        if not isinstance(self._image, Image):
            raise Exception("Trying to deploy a deployment without an image configured.")
        if not isinstance(self.name, str):
            raise Exception("Trying to deploy an unbound deployment. Have any remote functions been registered?")
        if not isinstance(self._autoscaling, Autoscaling):
            raise Exception("Trying to deploy a deployment without autoscaling configured.")
        hash, size, tar_path = self._compute_hash_filesize_and_tarball(self.name, self._image)
        return (
            DeploymentConfig(
                name = self.name,
                image = self._image._image, 
                file_hash=hash, 
                file_size=size, 
                tag=self.tag, 
                search_params=self._image.requires_.query, 
                storage = self._image.storage_, 
                ttl = self._ttl, 
                version_label=self.version_label, 
                **(asdict(self._image.docker_login) if self._image.docker_login is not None else {}), 
                **asdict(self._autoscaling)
            ),
            tar_path
        )

    def _collate_config(self, checked_name : str, checked_image : Image) -> Config:
        # should error if any required fields are still None
        return Config(
            checked_name,
            checked_image.pip_installs_,
            checked_image.apt_gets_,
            list(checked_image.envs_.items()),
            checked_image.runs_
        )

    def _compute_hash_filesize_and_tarball(self, checked_name : str, checked_image : Image) -> tuple[str, int, str]:
        if not isinstance(self._deployment_file, str):
            raise Exception("Trying to deploy a deployment not yet bound to a Python module. Have any remote functions been registered?")
        config = self._collate_config(checked_name,checked_image)
        hash = compute_deployment_hash(config, self._deployment_file, checked_image.copies)
        tar_path = create_deployment_tarball(config, self._deployment_file, checked_image.copies)
        size = getsize(tar_path)
        return (hash, size, tar_path)
    
    async def ensure_ready(self):
        if not isinstance(self.root_module, str):
            raise Exception("Trying to deploy a deployment not yet bound to a Python module. Have any remote functions been registered?")
        config, tar_path = self._into_deployment_config_and_tarball()
        deployment = await self.client.put_deployment(config)
        if deployment.needs_upload:
            await deployment.upload(tar_path)
        self._inner = _FullDeployment(self.root_module, deployment)

    async def _dispatch(self, f_name, globals, sig, args, kwargs) -> Any:
        if not isinstance(self._inner, _FullDeployment):
            raise Exception("Deployment is not ready. Call .ensure_ready() first!")
        bound_args = sig.bind(*args,**kwargs)
        bound_args.apply_defaults()
        return serialization.deserialize_unwrap_error(
            await self._inner.deployment.endpoint.request(
                '/remote/' + '/'.join(f_name), 
                {
                    k : serialization.serialize(v, self._inner.root_module) for k,v in bound_args.arguments.items()
                }
            ),
            self._inner.root_module,
            globals 
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
            
        def decorator(f : Callable[P, Awaitable[Any]], **_) -> Callable[P, Awaitable[Any]]:
            f_rel_name = self.relativize(f)
            f_globals = f.__globals__ 
            sig = inspect.signature(f)
            def inner(*args : P.args, **kwargs : P.kwargs) -> Awaitable[Any]:
                return self._dispatch(f_rel_name, f_globals, sig, args,kwargs)
            return inner

        if f is not None:
            return decorator(f)
        else: 
            return decorator


