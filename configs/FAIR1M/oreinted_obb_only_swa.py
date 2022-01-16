_base_ = ['/home/hnu1/GGM/OBBDetection/configs/FAIR1M/oriented_obb.py',
          '/home/hnu1/GGM/OBBDetection/configs/_base_/swa.py']

only_swa_training = True
swa_training = True
swa_load_from = '/home/hnu1/GGM/OBBDetection/work_dir/oriented_obb/epoch_12.pth'
swa_resume_from = None

# swa optimizer
swa_optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
# swa learning policy
swa_lr_config = dict(
    policy='cyclic',
    target_ratio=(1, 0.01),
    cyclic_times=12,
    step_ratio_up=0.0)
total_epochs = 12
max_epochs=12