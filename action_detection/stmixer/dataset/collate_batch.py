import math
import torch


def batch_different_videos(videos, size_divisible=0):
    '''
    :param videos: a list of video tensors
    :param size_divisible: output_size(width and height) should be divisble by this param
    :return: batched videos as a single tensor
    '''
    assert isinstance(videos, (tuple, list))
    max_size = tuple(max(s) for s in zip(*[clip.shape for clip in videos]))

    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size[3] = int(math.ceil(max_size[3] / stride) * stride)
        max_size = tuple(max_size)

    batch_shape = (len(videos),) + max_size
    batched_clips = videos[0].new(*batch_shape).zero_()
    for clip, pad_clip in zip(videos, batched_clips):
        pad_clip[:clip.shape[0], :clip.shape[1], :clip.shape[2], :clip.shape[3]].copy_(clip)

    return batched_clips


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched object images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.divisible = size_divisible
        self.size_divisible = self.divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        slow_clips = batch_different_videos(transposed_batch[0], self.size_divisible)
        if transposed_batch[1][0] is not None:
            fast_clips = batch_different_videos(transposed_batch[1], self.size_divisible)
        else:
            fast_clips = None
        whwh = torch.stack(transposed_batch[2])
        boxes = transposed_batch[3]
        label_arrs = transposed_batch[4]
        metadata = transposed_batch[5]
        clip_ids = transposed_batch[6]
        return slow_clips, fast_clips, whwh, boxes, label_arrs, metadata, clip_ids