from typing import List, Tuple
from types import FunctionType
import torch
import numpy as np


def np2tensor(
    x: np.ndarray,
    make_batch: bool=False,
    ):
    ''' Converts numpy array to tensor, adding a batch
        dimension if "make_batch" is "True".
    '''
    if make_batch:
        x = x[None, :]
    x = torch.from_numpy(x).float()
    return x


def resize_range(
    x: np.ndarray,
    min_val: np.uint8=0,
    max_val: np.uint8=255
    ):
    ''' Converts values from min_val - max_val to 0 - 1 interval.
    '''
    assert max_val > min_val
    x = (x - min_val) / (max_val - min_val)
    return x


def experience2batches(
    exp: List[Tuple]
    ):
    ''' Unpacks a experiences into batch tensors.
    '''
    state = []
    action = []
    reward = []
    new_state = []
    for s, a, r, s_ in exp:
        state.append(s)
        action.append(a)
        reward.append(r)
        new_state.append(s_)
    state = torch.cat(state, dim=0)
    new_state = torch.cat(new_state, dim=0)
    action = torch.tensor(action)
    reward = torch.cat(reward, dim=0)
    assert len(state) == len(new_state) == len(reward) == len(action)
    return state, action, reward, new_state


class NumpyArrayTransform:
    ''' Class for preprocessing input data for DL model.
    '''
    def __init__(
        self,
        transforms: List[FunctionType]
        ):
        self._transforms = transforms

    def add(
        self,
        transform: FunctionType
        ):
        ''' Adds the new transform to the sequence of transforms.
        '''
        self._transforms.append(transform)

    def __call__(
        self,
        x: np.ndarray
        ):
        ''' Applies all the transforms to x and returns the result.
        '''
        for t in self._transforms:
            x = t(x)
        return x
