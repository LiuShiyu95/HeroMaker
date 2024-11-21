from .distributed_weighted_sampler import DistributedWeightedSampler
from .video_dataset_deform import VideoDatasetDeform
dataset_dict = {"video_deform": VideoDatasetDeform}

custom_sampler_dict = {'weighted': DistributedWeightedSampler}