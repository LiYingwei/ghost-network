import argparse
import os

from tensorpack import logger, QueueInput
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import (
    TrainConfig, SyncMultiGPUTrainerReplicated, launch_train_with_config)
from tensorpack.dataflow import FakeData
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu

from networks.lib.tp.imagenet_utils import (
    fbresnet_augmentor, get_imagenet_dataflow, ImageNetModel,
    eval_on_ILSVRC12)
from networks.lib.tp.resnet_model import (
    preresnet_group, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck, se_resnet_bottleneck,
    resnet_backbone)

class Model(ImageNetModel):
    def __init__(self, depth, mode='resnet'):
        if mode == 'se':
            assert depth >= 50

        self.mode = mode
        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'se': se_resnet_bottleneck}[mode]
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]

    def get_logits(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            return resnet_backbone(
                image, self.num_blocks,
                preresnet_group if self.mode == 'preact' else resnet_group, self.block_func)