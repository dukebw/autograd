import warnings
from contextlib import contextmanager
from collections import defaultdict
from .util import subvals, toposort
from .wrap_util import wraps

def trace(start_node, fun, x):
    with trace_stack.new_trace() as t:
        start_box = new_box(x, t, start_node)
        end_box = fun(start_box)
        if isbox(end_box) and end_box._trace == start_box._trace:
            return end_box._value, end_box._node
        else:
            warnings.warn("Output seems independent of input.")
            return end_box, None

class Node(object):
    __slots__ = []
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        assert False

    def initialize_root(self, *args, **kwargs):
        assert False

    @classmethod
    def new_root(cls, *args, **kwargs):
        root = cls.__new__(cls)
        root.initialize_root(*args, **kwargs)
        return root

def primitive(f_raw):
    """
    Wraps a function so that its gradient can be specified and its invocation
    can be recorded. For examples, see the docs."""
    # NOTE(brendan): Decorators take a function as argument, and return another
    # function, i.e.,
    # @bar
    # def foo(x):
    # ...
    # is equivalent to:
    # def foo(x):
    # ...
    # foo = bar(foo)
    #
    # wraps is a function that returns a decorator, which replaces __name__ and
    # __doc__ of the function (f_wrapped) returned by primitive with those of
    # f_raw.
    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        # NOTE(brendan): Finds the boxed args topmost on the stack.
        boxed_args, trace, node_constructor = find_top_boxed_args(args)
        if boxed_args:
            # NOTE(brendan): Extract out the values from the boxes, and call
            # the wrapped function.
            argvals = subvals(args, [(argnum, box._value) for argnum, box in boxed_args])
            if f_wrapped in notrace_primitives[node_constructor]:
                return f_wrapped(*argvals, **kwargs)
            parents = tuple(box._node for _     , box in boxed_args)
            argnums = tuple(argnum    for argnum, _   in boxed_args)
            ans = f_wrapped(*argvals, **kwargs)
            # NOTE(brendan): This seems to be creating the node, which houses a
            # function (node.vjp) that calls a bunch of corresponding VJPs for
            # all the arguments to this Node that are on top of the trace
            # stack (i.e., all have arg._trace the highest of any boxed arg).
            #
            # Why might this be done? Possibly because those top boxed args are
            # what needs to be evaluated next.
            #
            # See defvjp in autograd/core.py.
            node = node_constructor(ans, f_wrapped, argvals, kwargs, argnums, parents)
            return new_box(ans, trace, node)
        else:
            return f_raw(*args, **kwargs)
    f_wrapped.fun = f_raw
    f_wrapped._is_autograd_primitive = True
    return f_wrapped

notrace_primitives = defaultdict(set)
def register_notrace(trace_type, primitive_fun):
    notrace_primitives[trace_type].add(primitive_fun)

def notrace_primitive(f_raw):
    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        argvals = map(getval, args)
        return f_raw(*argvals, **kwargs)
    f_wrapped._is_primitive = True
    return f_wrapped

def find_top_boxed_args(args):
    top_trace = -1
    top_boxes = []
    top_node_type = None
    for argnum, arg in enumerate(args):
        if isbox(arg):
            trace = arg._trace
            if trace > top_trace:
                top_boxes = [(argnum, arg)]
                top_trace = trace
                top_node_type = type(arg._node)
            elif trace == top_trace:
                top_boxes.append((argnum, arg))
    return top_boxes, top_trace, top_node_type

class TraceStack(object):
    def __init__(self):
        self.top = -1
    @contextmanager
    def new_trace(self):
        self.top += 1
        yield self.top
        self.top -= 1
trace_stack = TraceStack()

class Box(object):
    type_mappings = {}
    types = set()

    __slots__ = ['_value', '_trace', '_node']
    def __init__(self, value, trace, node):
        self._value = value
        self._node = node
        self._trace = trace

    def __bool__(self):
        return bool(self._value)

    __nonzero__ = __bool__

    def __str__(self):
        return "Autograd {0} with value {1}".format(
            type(self).__name__, str(self._value))

    @classmethod
    def register(cls, value_type):
        Box.types.add(cls)
        Box.type_mappings[value_type] = cls
        Box.type_mappings[cls] = cls

box_type_mappings = Box.type_mappings
def new_box(value, trace, node):
    try:
        return box_type_mappings[type(value)](value, trace, node)
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))

box_types = Box.types
isbox  = lambda x: type(x) in box_types  # almost 3X faster than isinstance(x, Box)
getval = lambda x: getval(x._value) if isbox(x) else x
