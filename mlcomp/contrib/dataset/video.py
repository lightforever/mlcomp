import os
import pickle
import random
import warnings
from collections import defaultdict
from numbers import Number
from os.path import join
from typing import Callable, Dict, List

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.video_utils import VideoClips

warnings.filterwarnings('ignore')


class VideoClipsFolder:
    def __init__(self, video_paths: List[str], clip_length_in_frames: int,
                 frames_between_clips: int,
                 _precomputed_metadata: dict = None):
        self.video_paths = video_paths
        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips = frames_between_clips
        self.cumulative_sizes = []
        self.clips = []
        self.metadata = _precomputed_metadata
        self.compute_clips()

    # noinspection PyTypeChecker
    def compute_clips(self):
        if self.metadata is not None:
            self.clips = self.metadata['clips']
            self.cumulative_sizes = self.metadata[
                'cumulative_sizes']
            self.video_paths = self.metadata['video_paths']
            return

        clips_size = 0
        for video_index, folder in enumerate(self.video_paths):
            files = sorted(os.listdir(folder))
            assert len(files) >= self.clip_length_in_frames, \
                f'folder = {folder} has only {len(files)} files'

            for i in range(0, len(files) - self.clip_length_in_frames,
                           self.frames_between_clips):
                clips_size += 1
                self.clips.append({
                    'video_index': video_index,
                    'min_index': i,
                    'max_index': i + self.clip_length_in_frames
                })

            self.cumulative_sizes.append(clips_size)

        self.metadata = {
            'clips': self.clips,
            'cumulative_sizes': self.cumulative_sizes,
            'video_paths': self.video_paths
        }

    def get_clip(self, index: int):
        clip = self.clips[index]
        files = sorted(os.listdir(self.video_paths[clip['video_index']]))
        imgs = []
        for index in range(clip['min_index'], clip['max_index']):
            file = join(self.video_paths[clip['video_index']], files[index])
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        return imgs, None, None, clip['video_index']


class VideoDataset(Dataset):
    def __init__(
            self,
            *,
            video_folder: str,
            fold_csv: str = None,
            fold: int = None,
            is_test: bool = False,
            clip_length_in_frames: int = 1,
            frames_between_clips: int = 1,
            num_classes=2,
            max_count=None,
            transforms=None,
            postprocess_func: Callable[[Dict], Dict] = None,
            clips_per_video: int = 1,
            metadata_path: str = None
    ):
        self.video_folder = video_folder

        if fold_csv:
            df = pd.read_csv(fold_csv)
            if fold is not None:
                if is_test:
                    self.data = df[df['fold'] == fold]
                else:
                    self.data = df[df['fold'] != fold]
            else:
                self.data = df
        else:
            self.data = pd.DataFrame(
                {'video': os.listdir(video_folder)}).sort_values(by='video')

        self.data = self.data.sample(frac=1)
        self.data = self.data.to_dict(orient='row')
        if max_count is not None:
            self.apply_max_count(max_count)

        for row in self.data:
            self.preprocess_row(row)

        self.transforms = transforms
        self.num_classes = num_classes
        self.postprocess_func = postprocess_func
        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips = frames_between_clips
        self.clips_per_video = clips_per_video
        self.metadata_path = metadata_path

        self.video_paths = [row['video'] for row in self.data]
        self.scan_mode = any(os.path.isdir(v) for v in self.video_paths)
        self.clips = self.create_clips()

    def create_clips(self):
        precomputed_metadata = None
        if self.metadata_path:
            if os.path.exists(self.metadata_path):
                precomputed_metadata = pickle.load(
                    open(self.metadata_path, 'rb'))
                if precomputed_metadata is not None:
                    if len(precomputed_metadata['video_paths']) != len(
                            self.video_paths):
                        precomputed_metadata = None

                    if precomputed_metadata is not None and any(
                            [p1 != p2 for p1, p2 in zip(self.video_paths,
                                                        precomputed_metadata[
                                                            'video_paths'])]):
                        precomputed_metadata = None

        if precomputed_metadata is None:
            print('computing metadata')

        cls = VideoClipsFolder if self.scan_mode else VideoClips
        clips = cls(
            video_paths=self.video_paths,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            _precomputed_metadata=precomputed_metadata
        )

        if precomputed_metadata is None and self.metadata_path:
            folder = os.path.dirname(self.metadata_path)
            os.makedirs(folder, exist_ok=True)
            pickle.dump(clips.metadata, open(self.metadata_path, 'wb'))

        return clips

    def apply_max_count(self, max_count):
        if isinstance(max_count, Number):
            self.data = self.data[:max_count]
        else:
            data = defaultdict(list)
            for row in self.data:
                data[row['label']].append(row)
            min_index = np.argmin(max_count)
            min_count = len(data[min_index])
            for k, v in data.items():
                count = int(min_count * (max_count[k] / max_count[min_index]))
                data[k] = data[k][:count]

            self.data = [v for i in range(len(data)) for v in data[i]]

    def preprocess_row(self, row: dict):
        row['video'] = join(self.video_folder, row['video'])

    def __len__(self):
        return len(self.clips.video_paths) * self.clips_per_video

    def __getitem__(self, index):
        video_index = index // self.clips_per_video
        max_clip_index = self.clips.cumulative_sizes[video_index]
        min_clip_index = self.clips.cumulative_sizes[
            video_index - 1] if video_index > 0 else 0

        clip_index = np.random.randint(min_clip_index, max_clip_index)
        video, audio, info, _ = self.clips.get_clip(clip_index)
        row = self.data[video_index]

        if self.transforms:
            seed = random.randint(0, 10 ** 6)
            frames = []
            if not self.scan_mode:
                # noinspection PyUnresolvedReferences
                video = video.numpy()

            for v in video:
                random.seed(seed)
                frames.append(self.transforms(image=v)['image'][None])

            video = np.vstack(frames)

        res = {
            'features': video,
            'video_file': row['video']
        }
        if 'label' in row:
            res['targets'] = row['label']

        if self.postprocess_func:
            res = self.postprocess_func(res)
        return res


__all__ = ['VideoDataset']
