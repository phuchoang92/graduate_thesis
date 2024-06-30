# Dataset configuration


dataset_config = {
    
    'ava_v2.2':{
        # dataset
        'frames_dir': 'frames/',
        'frame_list': 'frame_lists/',
        'annotation_dir': 'annotations/',
        'train_gt_box_list': 'train.csv',
        'val_gt_box_list': 'val.csv',
        'train_exclusion_file': 'train_excluded_timestamps.csv',
        'val_exclusion_file': 'val_excluded_timestamps.csv',
        'labelmap_file': 'action_list.pbtxt', # 'ava_v2.2/ava_action_list_v2.2.pbtxt',
        'class_ratio_file': 'config/ava_categories_ratio.json',
        'backup_dir': 'results/',
        # input size
        'train_size': 224,
        'test_size': 224,
        # transform
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 2,
        # cls label
        'multi_hot': True,  # multi hot
        # train config
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.0066667,
        'wp_iter': 8000,
        # class names
        'valid_num_classes': 7,
        'label_map': (
                    'normal actions', 'use smartphones', 'sleep', 'eat/drink', 'group discuss', 'walk', 'leave seat'
                ),
    }
}