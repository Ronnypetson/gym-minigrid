from typing import List
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
    x = torch.from_numpy(x, requires_grad=False).float()
    return x


def resize_range(
    x: np.ndarray,
    min_val: np.uint8=0,
    max_val: np.uint8=255
    ):
    ''' Converts values from min_val - max_val to 0 - 1 interval.
    '''
    x = (x - min_val) / (max_val - min_val)
    return x


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
