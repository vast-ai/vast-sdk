from vastai.serverless.remote.endpoint import *

@benchmark(
    endpoint_name="pytorch",
    dataset=[{'a': [[2, 0], [0, 2]], 'b': [[1, 2], [3, 4]]}]
)
@remote(endpoint_name='pytorch')
async def matmul(a,b):
    import torch as t
    a_t = t.tensor(a, dtype=t.float32, device='cuda:0')
    b_t = t.tensor(b, dtype=t.float32, device='cuda:0')
    c_t = a_t @ b_t
    return c_t.to('cpu').tolist()

ep = Endpoint(
    'pytorch',
    image_name='pytorch/pytorch'
)
ep.ready()

