import os
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import functorch
import patoolib
import wget
from torch import Tensor
from torchvision.io import read_video

from chronica.torch.utils.data import Dataset

__all__ = ["HMDB51"]


class HMDB51(Dataset):
    r"""
    `HMDB51 <https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>`_
    dataset.
    HMDB51 is an action recognition video dataset.
    Args:
        root (string): Root directory of the HMDB51 Dataset.
        annotation_path (str): Not used but for torchvision compatibility.
        frames_per_clip (int): Not used but for torchvision compatibility.
        step_between_clips (int): Not used but for torchvision compatibility.
        fold (int, optional): Which fold to use. Should be between 1 and 3.
        train (bool, optional): If ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that takes in a TxHxWxC video
            and returns a transformed version.
        output_format (str, optional): The format of the output video tensors (before transforms).
            Can be either "THWC" (default) or "TCHW".
        download (bool, optional): If ``True``, downloads the dataset from the internet and
            puts it in :attr:`root` directory.
    Returns:
        tuple: A 3-tuple with the following entries:
            - video (Tensor[T, H, W, C] or Tensor[T, C, H, W]): The `T` video frames
            - audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
              and `L` is the number of points
            - label (int): class of the video clip
    """

    data_url = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
    splits = {
        "url": "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
        "md5": "15e67781e70dcfbdce2d7dbb9b3344b5",
    }
    TRAIN_TAG = 1
    TEST_TAG = 2
    actions = ["brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs", "dive", "draw_sword", "dribble", "drink", "eat", "fall_floor", "fencing", "flic_flac", "golf", "handstand", "hit", "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse", "run", "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp", "smile", "smoke", "somersault", "stand", "swing_baseball", "sword", "sword_exercise", "talk", "throw", "turn", "walk", "wave"]

    def __init__(
        self,
        root: str,
        annotation_path: str,
        frames_per_clip: int,
        step_between_clips: int = 1,
        frame_rate: Optional[int] = None,
        fold: int = 1,
        train: bool = True,
        transform: Optional[Callable] = None,
        _precomputed_metadata: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,
        _video_width: int = 0,
        _video_height: int = 0,
        _video_min_dimension: int = 0,
        _audio_samples: int = 0,
        output_format: str = "THWC",
        download: bool = False,
    ) -> None:
        if fold not in (1, 2, 3):
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.root = root
        self.transform = transform
        self.output_format = output_format

        if download:
            # download and extract archives to :attr:`root`.
            filename = wget.download(self.data_url, out=self.root)
            patoolib.extract_archive(filename, outdir=self.root)
            os.remove(filename)

            for action in self.actions:
                filename = os.path.join(self.root, "{}.rar".format(action))
                patoolib.extract_archive(filename, outdir=self.root)
                os.remove(filename)

            filename = wget.download(self.splits.get("url"), out=self.root)
            patoolib.extract_archive(filename, outdir=self.root)
            os.remove(filename)

        self.videos = list()
        self.labels = list()

        for label, action in enumerate(self.actions):
            filename = os.path.join(self.root, "testTrainMulti_7030_splits/", "{}_test_split{}.txt".format(action, fold))
            lines = map(lambda line: line.split(), open(filename).readlines())
            for line in lines:
                filename, tag = os.path.join(self.root, action, line[0]), int(line[1])
                if train and tag == self.TRAIN_TAG or not train and tag == self.TEST_TAG:
                    self.videos.append(filename)
                    self.labels.append(label)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, _ = read_video(self.videos[index], pts_unit="sec", output_format=self.output_format)
        if self.transform is not None:
            # apply :attr:`transform` to each of the video frames.
            video = functorch.vmap(self.transform)(video)
        return video, audio, self.labels[index]

    def __sizeof__(self, index: int) -> int:  # type: ignore[override]
        return min(int(cv2.VideoCapture(self.videos[index]).get(cv2.CAP_PROP_FRAME_COUNT)), 266)

    def __len__(self) -> int:
        return len(self.videos)

    def __repr__(self) -> str:
        head = "Dataset {}".format(self.__class__.__name__)
        lines = list()
        lines.append(head)
        lines.append("   Number of videos: {}".format(len(self)))
        lines.append("   Root location: {}".format(self.root))
        return "\n".join(lines)
