import numpy as np
import os
from easydict import EasyDict as edict

config = edict()

config.bn_mom = 0.9
config.workspace = 256
config.emb_size = 512
config.ckpt_embedding = False
config.net_se = 0
config.net_act = 'prelu'
config.net_unit = 3
config.net_input = 1
config.net_blocks = [1,4,6,2]
config.net_output = 'E'
config.net_multiplier = 1.0
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
config.ce_loss = True     # cross entropy
config.fc7_lr_mult = 10.0
config.fc7_wd_mult = 1.0
config.fc7_no_bias = False
config.max_steps = 0
config.data_rand_mirror = True
config.data_cutoff = False
config.data_color = 1
config.data_images_filter = 0   # filter fewer images per identity, set threadshold
config.count_flops = True
config.memonger = False #not work now


# network settings
network = edict()

network.r100 = edict()
network.r100.net_name = 'fresnet'
network.r100.num_layers = 100

network.r34 = edict()
network.r34.net_name = 'fresnet'
network.r34.num_layers = 34

network.rs100 = edict()
network.rs100.net_name = 'fresnet'
network.rs100.num_layers = 100

network.rs34 = edict()
network.rs34.net_name = 'fresnet'
network.rs34.num_layers = 34

network.r100fc = edict()
network.r100fc.net_name = 'fresnet'
network.r100fc.num_layers = 100
network.r100fc.net_output = 'FC'

network.r50 = edict()
network.r50.net_name = 'fresnet'
network.r50.num_layers = 50

network.r50v1 = edict()
network.r50v1.net_name = 'fresnet'
network.r50v1.num_layers = 50
network.r50v1.net_unit = 1

network.d169 = edict()
network.d169.net_name = 'fdensenet'
network.d169.num_layers = 169
network.d169.per_batch_size = 64
network.d169.densenet_dropout = 0.0

network.d201 = edict()
network.d201.net_name = 'fdensenet'
network.d201.num_layers = 201
network.d201.per_batch_size = 64
network.d201.densenet_dropout = 0.0

network.y1 = edict()
network.y1.net_name = 'fmobilefacenet'
network.y1.emb_size = 128
network.y1.net_output = 'GDC'

network.y2 = edict()
network.y2.net_name = 'fmobilefacenet'
network.y2.emb_size = 256
network.y2.net_output = 'GDC'
network.y2.net_blocks = [2,8,16,4]

network.m1 = edict()
network.m1.net_name = 'fmobilenet'
network.m1.emb_size = 256
network.m1.net_output = 'GDC'
network.m1.net_multiplier = 1.0

network.m05 = edict()
network.m05.net_name = 'fmobilenet'
network.m05.emb_size = 256
network.m05.net_output = 'GDC'
network.m05.net_multiplier = 0.5

network.mnas = edict()
network.mnas.net_name = 'fmnasnet'
network.mnas.emb_size = 256
network.mnas.net_output = 'GDC'
network.mnas.net_multiplier = 1.0

network.mnas05 = edict()
network.mnas05.net_name = 'fmnasnet'
network.mnas05.emb_size = 256
network.mnas05.net_output = 'GDC'
network.mnas05.net_multiplier = 0.5

network.mnas025 = edict()
network.mnas025.net_name = 'fmnasnet'
network.mnas025.emb_size = 256
network.mnas025.net_output = 'GDC'
network.mnas025.net_multiplier = 0.25

network.vargfacenet = edict()
network.vargfacenet.net_name = 'vargfacenet'
network.vargfacenet.net_multiplier = 1.25
network.vargfacenet.emb_size = 512
network.vargfacenet.net_output='J'

# dataset settings
dataset = edict()

dataset.emore = edict()
dataset.emore.dataset = 'emore'
dataset.emore.dataset_path = '../datasets/faces_emore'
dataset.emore.num_classes = 85742
dataset.emore.image_shape = (112,112,3)
dataset.emore.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.retina = edict()
dataset.retina.dataset = 'retina'
dataset.retina.dataset_path = '../datasets/ms1m-retinaface-t1'
dataset.retina.num_classes = 93431
dataset.retina.image_shape = (112,112,3)
dataset.retina.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.glint = edict()
dataset.glint.dataset = 'glint'
dataset.glint.dataset_path = '../datasets/faces_glint'
dataset.glint.num_classes = 180855
dataset.glint.image_shape = (112,112,3)
dataset.glint.val_targets = ['lfw']

dataset.glintmask = edict()
dataset.glintmask.dataset = 'glintmask'
dataset.glintmask.dataset_path = '../datasets/faces_glint_mask'
dataset.glintmask.num_classes = 180855
dataset.glintmask.image_shape = (112,112,3)
dataset.glintmask.val_targets = ['lfw']


loss = edict()
loss.softmax = edict()
loss.softmax.loss_name = 'softmax'

loss.nsoftmax = edict()
loss.nsoftmax.loss_name = 'margin_softmax'
loss.nsoftmax.loss_s = 64.0
loss.nsoftmax.loss_m1 = 1.0
loss.nsoftmax.loss_m2 = 0.0
loss.nsoftmax.loss_m3 = 0.0

loss.arcface = edict()
loss.arcface.loss_name = 'margin_softmax'
loss.arcface.loss_s = 64.0
loss.arcface.loss_m1 = 1.0
loss.arcface.loss_m2 = 0.5
loss.arcface.loss_m3 = 0.0

loss.cosface = edict()
loss.cosface.loss_name = 'margin_softmax'
loss.cosface.loss_s = 64.0
loss.cosface.loss_m1 = 1.0
loss.cosface.loss_m2 = 0.0
loss.cosface.loss_m3 = 0.35

loss.combined = edict()
loss.combined.loss_name = 'margin_softmax'
loss.combined.loss_s = 64.0
loss.combined.loss_m1 = 1.0
loss.combined.loss_m2 = 0.3
loss.combined.loss_m3 = 0.2

loss.triplet = edict()
loss.triplet.loss_name = 'triplet'
loss.triplet.images_per_identity = 5
loss.triplet.triplet_alpha = 0.3
loss.triplet.triplet_bag_size = 7200
loss.triplet.triplet_max_ap = 0.0
loss.triplet.per_batch_size = 60
loss.triplet.lr = 0.05

loss.atriplet = edict()
loss.atriplet.loss_name = 'atriplet'
loss.atriplet.images_per_identity = 5
loss.atriplet.triplet_alpha = 0.35
loss.atriplet.triplet_bag_size = 7200
loss.atriplet.triplet_max_ap = 0.0
loss.atriplet.per_batch_size = 60
loss.atriplet.lr = 0.05

# default settings
default = edict()

# default network
default.network = 'r34'
default.pretrained = './pretrain_model/model'   # pretrained model, need pretrained model prefix
default.pretrained_epoch = 0
# default.pretrained = './ckpt/r34-arcface-glint/model'   # pretrained model, need pretrained model prefix
# default.pretrained_epoch = 5
# default dataset
default.dataset = 'glint'
default.loss = 'combined'
default.frequent = 20     # Specifies how frequently training speed and evaluation metrics must be logged, 20 batch
default.verbose = 5000    # how frequently do testing and save ckpt, 5000 batch
default.kvstore = 'device'

default.end_epoch = 10000
default.lr = 0.05
default.wd = 0.0005
default.mom = 0.9
default.per_batch_size = 256
default.ckpt = 2
default.lr_steps = '30000,90000,150000,210000'
# default.lr_steps = '85000,145000,225000'
default.models_root = './ckpt'    # save model path


def generate_config(_network, _dataset, _loss):
    for k, v in loss[_loss].items():     # loss is a edict(), and loss[_loss] is also a edict()
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in network[_network].items():  # network is a edict(), and network[_network] is also a edict()
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in dataset[_dataset].items():  # dataset is a edict(), and dataset[_dataset] is also a edict()
      config[k] = v
      if k in default:
        default[k] = v
    config.loss = _loss
    config.network = _network
    config.dataset = _dataset
    config.num_workers = 1
    if 'DMLC_NUM_WORKER' in os.environ:
      config.num_workers = int(os.environ['DMLC_NUM_WORKER'])

