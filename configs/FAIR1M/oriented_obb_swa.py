_base_ = ['/home/hnu1/GGM/OBBDetection/configs/FAIR1M/oriented_obb.py',
          '/home/hnu1/GGM/OBBDetection/configs/_base_/swa.py']

# swa optimizer
swa_optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))