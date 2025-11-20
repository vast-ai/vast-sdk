from vastai import Remote
import asyncio 

@remote(endpoint="my-remote-endpoint")
async def remote_func(a : int, b : str, c: dict[str, int]):
    # code here
    return a + 1


ep = Remote.Endpoint("my-remote-endpoint")
ep.apt_get("tmux")
ep.pip_install("transformers")
ep.on_start("echo \"Hello world!\"")

ep.set_max_workers(20)
ep.set_min_load(500)
# ... more endpoint config ...

# Remote function JSON contact
'''
{
    "func_name": "remote_func",
    "params":
        {
            "a" : # some val,
            "b" : # some val,
            "c" : # some dict
        }
}
'''
# Pyworker setup
ep.calculate_load(lambda request: request["params"]["a"])
benchmark_data = [
    {
        "a" : 1,
        "b" : "test",
        "c" : { "hello" : 0}
    },
        {
        "a" : 2,
        "b" : "test",
        "c" : { "world" : 2}
    },
        {
        "a" : 3,
        "b" : "test",
        "c" : { "bing" : 5}
    },
]
ep.benchmark_dataset(benchmark_data)

ep.ready()

# result = await remote_func(1, "hello", {"test" : 1})