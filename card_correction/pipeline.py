from abc import ABC
from .utils.device import (device_placement, create_device, verify_device) 
from .utils.constant import Frameworks
from .utils.torch_utils import compile_model
import torch
import numpy as np
from threading import Lock
from typing import Any, Dict, Mapping, Union

Input = Union[str, tuple, 'Image.Image', 'numpy.ndarray']


class Pipeline(ABC):
    """Base class for all pipelines.

    Args:
        model (str or Model): Model name or model instance.
        device (str, optional): Device to execute the model.
        **kwargs: Additional arguments to pass to the model.
    """
    def _process_single(self, input: Input, *args, **kwargs) -> Dict[str, Any]:
        preprocess_params = kwargs.get('preprocess_params', {})
        forward_params = kwargs.get('forward_params', {})
        postprocess_params = kwargs.get('postprocess_params', {})
        # self._check_input(input)
        out = self.preprocess(input, **preprocess_params)

        with device_placement(self.framework, self.device_name):
            if self.framework == Frameworks.torch:
                with torch.no_grad():
                    if self._auto_collate:
                        out = self._collate_fn(out)
                    out = self.forward(out, **forward_params)
            else:
                out = self.forward(out, **forward_params)

        out = self.postprocess(out, **postprocess_params)
        # self._check_output(out)
        return out

    def prepare_model(self):
        """ Place model on certain device for pytorch models before first inference
        """
        self._model_prepare_lock.acquire(timeout=600)

        def _prepare_single(model):
            if not isinstance(model, torch.nn.Module) and hasattr(
                    model, 'model'):
                model = model.model
            if not isinstance(model, torch.nn.Module):
                return
            model.eval()
            from .utils.torch_utils import is_on_same_device
            if is_on_same_device(model):
                model.to(self.device)

        if not self._model_prepare:
            # prepare model for pytorch
            if self.framework == Frameworks.torch:
                _prepare_single(self.model)
                if self._compile:
                    self.model = compile_model(self.model,
                                                   **self._compile_options)
            self._model_prepare = True
        self._model_prepare_lock.release()

    def _collate_fn(self, data):
        return collate_fn(data, self.device)
    
    def __call__(self, input, *args, **kwargs):
        if not self._model_prepare:
            self.prepare_model()
        output = self._process_single(input, *args, **kwargs)
        return output

    def __init__(self,
                 model: str,
                 device: str = 'gpu',
                 auto_collate=True,
                 device_map=None,
                 **kwargs):
        
        verify_device(device)
        self.framework = Frameworks.torch
        self.model = model
        self.device_name = device
        self.device = create_device(self.device_name)

        # 模型是否准备完毕
        self._model_prepare = False
        # 设置一个锁
        self._model_prepare_lock = Lock()
        self._auto_collate = auto_collate
        self._compile = kwargs.get('compile', False)
        self._compile_options = kwargs.get('compile_options', {})



def collate_fn(data, device):
    """Prepare the input just before the forward function.
    This method will move the tensors to the right device.
    Usually this method does not need to be overridden.

    Args:
        data: The data out of the dataloader.
        device: The device to move data to.

    Returns: The processed data.

    """
    from torch.utils.data.dataloader import default_collate

    def get_class_name(obj):
        return obj.__class__.__name__

    if isinstance(data, dict) or isinstance(data, Mapping):
        # add compatibility for img_metas for mmlab models
        return type(data)({
            k: collate_fn(v, device) if k != 'img_metas' else v
            for k, v in data.items()
        })
    elif isinstance(data, (tuple, list)):
        if 0 == len(data):
            return torch.Tensor([])
        if isinstance(data[0], (int, float)):
            return default_collate(data).to(device)
        else:
            return type(data)(collate_fn(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        if data.dtype.type is np.str_:
            return data
        else:
            return collate_fn(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (bytes, str, int, float, bool, type(None))):
        return data
    elif get_class_name(data) == 'InputFeatures':
        return data
    elif get_class_name(data) == 'DataContainer':
        # mmcv.parallel.DataContainer
        return data
    else:
        raise ValueError(f'Unsupported data type {type(data)}')