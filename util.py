import torch
import numpy as np
import random
#from termcolor import colored
import inspect
import os
import sys
import enum
import time
import math
#from functools import reduce
import operator
import datetime
import torch.multiprocessing

#from .distributions import Categorical

#torch.multiprocessing.set_sharing_strategy('file_system')

_device = torch.device('cpu')
_dtype = torch.float
_cuda_enabled = False
_verbosity = 2
_print_refresh_rate = 0.25  # seconds
_epsilon = 1e-8
_log_epsilon = math.log(_epsilon)  # log(1e-8) = -18.420680743952367


class TraceMode(enum.Enum):
    PRIOR = 1
    POSTERIOR = 2
    PRIOR_FOR_INFERENCE_NETWORK = 3


class PriorInflation(enum.Enum):
    DISABLED = 0
    ENABLED = 1


class InferenceEngine(enum.Enum):
    IMPORTANCE_SAMPLING = 0  # Type: IS; Importance sampling with proposals from prior
    IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK = 1  # Type: IS; Importance sampling with proposals from inference network
    LIGHTWEIGHT_METROPOLIS_HASTINGS = 2  # Type: MCMC; Lightweight (single-site) Metropolis Hastings sampling, http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf and https://arxiv.org/abs/1507.00996
    RANDOM_WALK_METROPOLIS_HASTINGS = 3  # Type: MCMC; Lightweight Metropolis Hastings with single-site proposal kernels that depend on the value of the site


class InferenceNetwork(enum.Enum):
    FEEDFORWARD = 0
    LSTM = 1


class ObserveEmbedding(enum.Enum):
    FEEDFORWARD = 0
    CNN2D5C = 1
    CNN3D5C = 2


class Optimizer(enum.Enum):
    ADAM = 0
    SGD = 1
    ADAM_LARC = 2
    SGD_LARC = 3


class LearningRateScheduler(enum.Enum):
    NONE = 0
    POLY1 = 1
    POLY2 = 2


class ImportanceWeighting(enum.Enum):
    IW0 = 0  # use prior as proposal for all replace=True addresses
    IW1 = 1


def set_random_seed(seed=None):
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)
    global _random_seed
    _random_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_random_seed()



def set_verbosity(v=2):
    global _verbosity
    _verbosity = v


def to_tensor(value, dtype=_dtype):
    if not torch.is_tensor(value):
        if type(value) == np.int64:
            value = torch.tensor(float(value))
        elif type(value) == np.float32:
            value = torch.tensor(float(value))
        else:
            value = torch.tensor(value)
    return value.to(device=_device, dtype=dtype)


def to_numpy(value):
    if torch.is_tensor(value):
        return value.cpu().numpy()
    elif isinstance(value, np.ndarray):
        return value
    else:
        try:
            return np.array(value)
        except Exception as e:
            print(e)
            raise TypeError('Cannot convert to Numpy array.')


def to_size(value):
    if isinstance(value, torch.Size):
        return value
    elif isinstance(value, int):
        return torch.Size([value])
    elif isinstance(value, list):
        return torch.Size(value)
    else:
        raise TypeError('Expecting a torch.Size, int, or list of ints.')


def fast_np_random_choice(values, probs_cumsum):
    # See https://mobile.twitter.com/RadimRehurek/status/928671225861296128
    return values[min(np.searchsorted(probs_cumsum, random.random()), len(values) - 1)]


def eval_print(*expressions):
    #print('\n\n' + colored(inspect.stack()[1][3], 'white', attrs=['bold']))
    frame = sys._getframe(1)
    max_str_length = 0
    for expression in expressions:
        if len(expression) > max_str_length:
            max_str_length = len(expression)
    for expression in expressions:
        val = eval(expression, frame.f_globals, frame.f_locals)
        if isinstance(val, np.ndarray):
            val = val.tolist()
        print('  {} = {}'.format(expression.ljust(max_str_length), repr(val)))


def progress_bar(i, len):
    bar_len = 20
    filled_len = int(round(bar_len * i / len))
    # percents = round(100.0 * i / len, 1)
    return '#' * filled_len + '-' * (bar_len - filled_len)


progress_bar_num_iters = None
progress_bar_len_str_num_iters = None
progress_bar_time_start = None
progress_bar_prev_duration = None


def progress_bar_init(message, num_iters, iter_name='Items'):
    global progress_bar_num_iters
    global progress_bar_len_str_num_iters
    global progress_bar_time_start
    global progress_bar_prev_duration
    if num_iters < 1:
        raise ValueError('num_iters must be a positive integer')
    progress_bar_num_iters = num_iters
    progress_bar_time_start = time.time()
    progress_bar_prev_duration = 0
    progress_bar_len_str_num_iters = len(str(progress_bar_num_iters))
    print(message)
    sys.stdout.flush()
    print('Time spent  | Time remain.| Progress             | {} | {}/sec'.format(iter_name.ljust(progress_bar_len_str_num_iters * 2 + 1), iter_name))


def progress_bar_update(iter):
    global progress_bar_prev_duration
    duration = time.time() - progress_bar_time_start
    if (duration - progress_bar_prev_duration > _print_refresh_rate) or (iter >= progress_bar_num_iters - 1):
        progress_bar_prev_duration = duration
        traces_per_second = (iter + 1) / duration
        print('{} | {} | {} | {}/{} | {:,.2f}       '.format(days_hours_mins_secs_str(duration), days_hours_mins_secs_str((progress_bar_num_iters - iter) / traces_per_second), progress_bar(iter, progress_bar_num_iters), str(iter).rjust(progress_bar_len_str_num_iters), progress_bar_num_iters, traces_per_second), end='\r')
        sys.stdout.flush()


def progress_bar_end(message=None):
    progress_bar_update(progress_bar_num_iters)
    print()
    if message is not None:
        print(message)


def days_hours_mins_secs_str(total_seconds):
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return '{0}d:{1:02}:{2:02}:{3:02}'.format(int(d), int(h), int(m), int(s))


def is_sorted(lst):
    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))


def truncate_str(s, length=50):
    return (s[:length] + '...') if len(s) > length else s


def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_time_stamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')


def one_hot(dim, i):
    t = torch.zeros(dim)
    t.narrow(0, i, 1).fill_(1)
    return to_tensor(t)


def is_hashable(v):
    try:
        hash(v)
    except TypeError:
        return False
    return True


#def check_gnu_dbm():
#    try:
#        import dbm.gnu
#    except ModuleNotFoundError:
#        return False
#    return True


#if not check_gnu_dbm():
    #print(colored(r'Warning: Empirical distributions on disk may perform slow because GNU DBM is not available. Please install and configure gdbm library for Python for better speed.', 'red', attrs=['bold']))


def tile_rows_cols(num_items):
    cols = math.ceil(math.sqrt(num_items))
    rows = 0
    while num_items > 0:
        rows += 1
        num_items -= cols
    return rows, cols


def create_path(path, directory=False):
    if directory:
        dir = path
    else:
        dir = os.path.dirname(path)
    if not os.path.exists(dir):
        print('{} does not exist, creating'.format(dir))
        os.makedirs(dir)


def address_id_to_int(address_id):
    if '__' not in address_id:
        return 0.
    else:
        divider_i = address_id.find('__')
        value_id = address_id[1:divider_i]
        return int(value_id)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def init_distributed_print(rank, world_size, debug_print=True):
    if not debug_print:
        if rank > 0:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
    else:
        # labelled print with info of [rank/world_size]
        old_out = sys.stdout

        class LabeledStdout:
            def __init__(self, rank, world_size):
                self._rank = rank
                self._world_size = world_size
                self.flush = sys.stdout.flush

            def write(self, x):
                if x == '\n':
                    old_out.write(x)
                else:
                    old_out.write('[r%d/ws%d] %s' % (self._rank, self._world_size, x))

        sys.stdout = LabeledStdout(rank, world_size)


def drop_items(l, num_items_to_drop):
    if num_items_to_drop > len(l):
        raise ValueError('Cannot drop more items than the list length')
    ret = l.copy()
    for _ in range(num_items_to_drop):
        del(ret[random.randrange(len(ret))])
    return ret


def get_source(obj):
    try:
        return inspect.getsource(obj)
    except:
        return obj.__name__
