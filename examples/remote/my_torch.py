from vastai_sdk.serverless.remote.endpoint import *

@benchmark(
    endpoint_name="hello_pytorch2",
    dataset=[{'a': [[2, 0], [0, 2]], 'b': [[1, 2], [3, 4]]}]
)
@remote(endpoint_name='hello_pytorch2')
async def matmul(a,b):
    import torch as t
    a_t = t.tensor(a, dtype=t.float32, device='cuda:0')
    b_t = t.tensor(b, dtype=t.float32, device='cuda:0')
    c_t = a_t @ b_t
    return c_t.to('cpu').tolist()

ep = Endpoint(
    'hello_pytorch2',
    image_name='pytorch/pytorch'
)
ep.ready()

