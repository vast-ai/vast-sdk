from abc import ABC, abstractmethod
from typing import (
    Callable,
    Any,
    Awaitable,
    ParamSpec,
    Optional,
    TypeVar,
    Type,
    AsyncContextManager,
    TypedDict,
)
from typing_extensions import Unpack
from vastai.data import Query
from dataclasses import dataclass


class DockerLogin(TypedDict, total=False):
    docker_login_user: str
    docker_login_pass: str
    docker_login_repo: str


class Image:
    def __init__(
        self,
        from_image: str,
        storage: float = 50,
        worker_port: tuple[int, str] = (3000, "tcp"),
        **docker_login: Unpack[DockerLogin],
    ):
        self.image_ = from_image
        self.envs_: dict[str, str] = {}
        self.runs_: list[str | tuple[str, ...]] = []
        self.pip_installs_: list[str] = []
        self.apt_gets_: list[str] = []
        self.docker_login: DockerLogin = docker_login
        self.requires_ = Query.search_defaults()
        self.storage_ = storage
        self.copies: list[tuple[str, str]] = []
        self.ports: set[tuple[int, str]] = {worker_port}

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

    def require(self, *args: Query) -> "Image":
        for arg in args:
            self.requires_.extend(arg)
        return self

    def copy(self, src: str, dst: str) -> "Image":
        self.copies.append((src, dst))
        return self

    def publish_port(self, number: int, type_: str = "tcp") -> "Image":
        self.ports.add((number, type_))
        return self


class Autoscaling(TypedDict, total=False):
    # Scaling params
    cold_workers: int
    max_workers: int
    min_load: int
    min_cold_load: int
    target_util: float
    cold_mult: int
    max_queue_time: float
    target_queue_time: float
    inactivity_timeout: int
    autoscaler_instance: str


@dataclass
class Config:
    name: str
    pip_installs: list[str]
    apt_gets: list[str]
    envs: list[list[str]] | list[tuple[str, ...]]  # format [(KEY,VALUE)]
    runs: (
        list[str | list[str]] | list[str | tuple[str, ...]]
    )  # tuples get turned into list after JSON serialize->deserialize round trip


T = TypeVar("T")
DeploymentT = TypeVar("DeploymentT", bound="Deployment_")


class Deployment_(ABC):
    by_name: dict[Type, dict[str, "Deployment_"]] = {}

    def __init__(
        self,
        name: Optional[str] = None,
        tag: str = "default",
        version_label: Optional[str] = None,
        **kwargs,
    ):
        self.name = name
        self.tag = tag
        self.version_label: str | None = version_label
        self.root_module: str | None = None
        self.is_main = False

    def context(
        self, *args, **kwargs
    ) -> Callable[
        [Type[AsyncContextManager[T]]], Type[AsyncContextManager[T]]
    ]:  # registers context_class as deployment context
        return lambda c: c  # no-op when not in serve mode

    def get_context(self, context_class: Type[AsyncContextManager[T]]) -> T:
        raise NotImplementedError(
            f"{context_class}"
        )  # should not be run when not in serve modej

    @property
    def name(self):
        return self.name_

    @name.setter
    def name(self, name: str | None):
        self.name_ = name
        if isinstance(name, str):
            if self.__class__ not in self.by_name:
                self.by_name[self.__class__] = {}
            self.by_name[self.__class__][name] = self

    @classmethod
    def lookup(cls: Type[DeploymentT], name: str) -> DeploymentT | None:
        if cls not in cls.by_name:
            return None
        ret = cls.by_name[cls].get(name)
        return ret  # type: ignore

    def relativize(self, f: Callable[..., Any]) -> tuple[str]:
        mod = f.__module__
        name = f.__name__
        file = f.__globals__.get("__file__")
        # assume single file deployment for now.
        if self.root_module is None:
            self.root_module = mod
            if self.name_ is None:
                self.name = ""
            if file is None:
                raise Exception(
                    f"Cowardly refusing to deploy function {name} defined in your interactive Python session! We need it to be defined in a .py file to deploy from."
                )
            self.file = file
        if f.__module__ != self.root_module:
            raise Exception(
                f"Using a single-file deployment object to deploy remote functions from multiple .py files! Remote function {name} is defined in {file} but we're already trying to deploy from {self.file}."
            )
        return (name,)

    P = ParamSpec("P")

    # TODO: add type checked version of remote
    @abstractmethod
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
    ): ...

    # Stubs for methods that are expected to be present in serve mode, but shouldn't do anything.
    def configure_autoscaling(self, **kwargs: Unpack[Autoscaling]):
        pass

    def image(self, from_image: str, storage: int) -> Image:
        return Image("")

    def ensure_ready(self):  # NO-OP in serve mode
        pass
