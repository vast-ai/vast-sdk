from vastai.serverless.remote import Deployment
from vastai.data.query import (
    gpu_name,
    static_ip,
    RTX_4090,
    RTX_4090_Ti,
    RTX_5090_Ti,
    RTX_5090,
)

app = Deployment(webserver_url="https://alpha-server.vast.ai", ttl=60)


@app.remote(benchmark_dataset=[{"x": 2}])
async def square(x):
    return x**2


app.configure_autoscaling()  # use defaults
app.image("pytorch/pytorch", 16).require(
    gpu_name.in_(
        [RTX_4090, RTX_4090_Ti, RTX_5090_Ti, RTX_5090]
    )  # unnec, but filters out really bad machines
)
app.ensure_ready()
