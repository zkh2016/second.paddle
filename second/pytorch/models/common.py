import sys
from collections import OrderedDict

import paddle
from paddle.nn import functional as F

class Empty(paddle.nn.Layer):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args

class Sequential(paddle.nn.Layer):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
        
        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_sublayer(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_sublayer(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._sub_layers:
                raise ValueError("name exists.")
            self.add_sublayer(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._sub_layers.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._sub_layers)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._sub_layers))
            if name in self._sub_layers:
                raise KeyError("name exists")
        self.add_sublayer(name, module)

    def forward(self, input):
        # i = 0
        for module in self._sub_layers.values():
            # print(i)
            input = module(input)
            # i += 1
        return input
