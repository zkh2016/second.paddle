import time 
import contextlib 
import paddle 

@contextlib.contextmanager
def paddle_timer(name=''):
    paddle.device.cuda.synchronize()
    t = time.time()
    yield
    paddle.device.cuda.synchronize()
    print(name, "time:", time.time() - t)
