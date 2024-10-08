import argparse
import torch.nn as nn
from .normalization_scalingonly import *
from ..utils import str2dict


# TODO:加入SOLN和SOBN -> DONE
def _GroupNorm(num_features, num_groups=32, eps=1e-5, affine=True, *args, **kwargs):
    return nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine)


def _BatchNorm(num_features, dim=4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, *args, **kwargs):
    return (nn.BatchNorm2d if dim == 4 else nn.BatchNorm1d)(num_features, eps=eps, momentum=momentum, affine=affine,
                                                            track_running_stats=track_running_stats)

def _LayerNorm(normalized_shape, eps=1e-5, affine=True, *args, **kwargs):
    return nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)


def _myLayerNorm(normalized_shape, eps=1e-5, affine=True, *args, **kwargs):
    return myLayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)


def _SOLayerNorm(normalized_shape, eps=1e-5, affine=True, *args, **kwargs):
    return SOLayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)


def _SOGroupNorm(num_features, num_groups=32, eps=1e-5, affine=True, *args, **kwargs):
    return SOGroupNorm(num_groups, num_features, eps=eps, affine=affine)


'''def _IdentityModule(x, *args, **kwargs):
    """return first input"""
    return IdentityModule()'''


def _Identity_fn(x, *args, **kwargs):
    """return first input"""
    return x


class _config:
    norm = 'BN'
    norm_cfg = {}
    norm_methods = {'GN': _GroupNorm, 'LN': _LayerNorm, 'SOGN': _SOGroupNorm, 'SOLN': _SOLayerNorm,
                    'myLN': _myLayerNorm, 'BN': _BatchNorm, 'None': None}


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Normalization Options')
    group.add_argument('--norm', default='No', help='Use which normalization layers? {' + ', '.join(
        _config.norm_methods.keys()) + '}' + ' (defalut: {})'.format(_config.norm))
    group.add_argument('--norm-cfg', type=str2dict, default={}, metavar='DICT', help='layers config.')
    return group

def getNormConfigFlag():
    flag = ''
    flag += _config.norm
    if str.find(_config.norm, 'GW')>-1 or str.find(_config.norm, 'GN') > -1:
        if _config.norm_cfg.get('num_groups') != None:
            flag += '_NG' + str(_config.norm_cfg.get('num_groups'))
    if str.find(_config.norm,'ItN') > -1:
        if _config.norm_cfg.get('T') != None:
            flag += '_T' + str(_config.norm_cfg.get('T'))
        if _config.norm_cfg.get('num_channels') != None:
            flag += '_NC' + str(_config.norm_cfg.get('num_channels'))

    if str.find(_config.norm,'DBN') > -1:
        flag += '_NC' + str(_config.norm_cfg.get('num_channels'))
    if _config.norm_cfg.get('affine') == False:
        flag += '_NoA'
    if _config.norm_cfg.get('momentum') != None:
        flag += '_MM' + str(_config.norm_cfg.get('momentum'))
    #print(_config.normConv_cfg)
    return flag

def setting(cfg: argparse.Namespace):
    print(_config.__dict__)
    for key, value in vars(cfg).items():
        #print(key)
        #print(value)
        if key in _config.__dict__:
            setattr(_config, key, value)
    #print(_config.__dict__)
    flagName = getNormConfigFlag()
    print(flagName)
    return flagName


def Norm(*args, **kwargs):
    kwargs.update(_config.norm_cfg)
    if _config.norm == 'None':
        return None
    return _config.norm_methods[_config.norm](*args, **kwargs)

