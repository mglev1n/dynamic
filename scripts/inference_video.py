import os
import numpy as np
import torch
import torchvision
from pathlib import Path
from typing import Union
from argparse import ArgumentParser
import argparse
import tqdm

from echonet.utils import loadvideo


## Video: Prediction of Ejection Fraction from Subsampled Clips
# model_name: r2plus1d_18
# frames: 32
# (sampling) period: 2


# relative paths to weights for various models
weights_path = Path(__file__).parent.parent / 'weights'
model_name = {'segmentation': 'deeplabv3_resnet50_random',
              'video': 'r2plus1d_18_32_2_pretrained'}


class VideoInferenceEngine:

    def __init__(
            self, weights_path=weights_path, model_name: str = model_name['video'], device=None
    ) -> None:

        model_path = weights_path / (model_name + '.pt')
        if isinstance(model_path, str):
            model_path = Path(model_path)
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        """Loads the model to be used for inference into memory.

        Returns:
            NamedTuple: See torch.Module.load_state_dict() for details
        """

        model = torchvision.models.video.r2plus1d_18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.fc.bias.data[0] = 55.6
        if self.device == "cuda":
            model = torch.nn.DataParallel(model)
        model.to(self.device)
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        if self.device == "cpu":
            checkpoint['state_dict']
            from collections import OrderedDict
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            self.model = model.load_state_dict(new_state_dict)
        else:
            self.model = model.load_state_dict(checkpoint['state_dict'])
        self.model = model.eval()
        return self.model

    def run_on_dir(
            self, in_dir: Union[Path, str], out_dir: Union[Path, str], verbose=True) -> None:
        """Run inference on A4C videos in a directory

        Args:
            in_dir (Union[Path, str]): Directory of A4C videos to run inference on.
            out_dir (Union[Path, str]): Directory to save results to.
            verbose (bool, optional): Print progress and updates. Defaults to True.
        """

        # Prepare
        in_dir = Path(in_dir) if isinstance(in_dir, str) else in_dir
        out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
        p = lambda s: print(s) if verbose else None
        if not out_dir.exists():
            out_dir.mkdir()
        paths = list(in_dir.iterdir())
        paths = [path for path in paths if '.avi' in path.name]

        # Start inference threads. Run inference, save results to out_dir
        if verbose:
            p('Running inference')
        self._run_on_clips(paths, out_dir, verbose=verbose)
        if verbose:
            p('Inference Finished')

    def _run_on_clips(
            self, paths, out_dir, verbose=True
    ) -> None:
        """Internally used to iterate through a list of paths and run inference. Batches may share frames from
        several clips. Yields clips, predictions, and filenames when inference is finished. Used for performance.
        """

        with open(os.path.join(out_dir, "test_predictions.csv"), "w") as g:
            g.write("FileName,EF\n")
            for path in tqdm.tqdm(paths):
                filename = path.name
                video = loadvideo(str(path))
                # c, f, h, w = video.shape
                x = self._preprocess_video(video)
                x = torch.from_numpy(x.astype(np.float32)).to(torch.float)
                x = x.to(self.device)
                y = self.model(x.unsqueeze(0))
                g.write("{},{:.4f}\n".format(filename, float(y)))

    def _get_video_mean_and_std(self, video):
        video = video.transpose(1, 0, 2, 3)
        video = video.reshape(3, -1)
        mean = np.mean(video, axis=1)
        std = np.std(video, axis=1)  # type: np.ndarray
        mean = mean.astype(np.float32)
        std = std.astype(np.float32)
        return mean, std

    def _preprocess_video(self, video, length=32, period=2):
        # Set number of frames
        c, f, h, w = video.shape

        if period:
            video = video[:, np.arange(0, f, period), :, :]

        if length:
            if f < length * period:
                # Pad video with frames filled with zeros if too short
                # 0 represents the mean color (dark grey), since this is after normalization
                video = np.concatenate((video, np.zeros((length * period - f, c, h, w), video.dtype)), axis=1)
            else:
                video = video[:, :length, :, :]

        video = video.astype(np.float32)
        mean, std = self._get_video_mean_and_std(video)
        video -= mean.reshape(3, 1, 1, 1)
        video /= std.reshape(3, 1, 1, 1)

        return video


class BoolAction(argparse.Action):

    """Class used by argparse to parse binary arguements.
    Yes, Y, y, True, T, t are all accepted as True. Any other
    arguement is evaluated as False.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        b = values.lower()[0] in ['t', 'y', '1']
        setattr(namespace, self.dest, b)
        print(parser)


if __name__ == '__main__':
    # CLI Interface for running inference on a directory
    # and saving predictions to an output directory.
    args = {
        'device': ('cuda:0', 'Device to run inference on. Ex: "cuda:0" or "cpu"'),
        'verbose': (True, 'Print progress and statistics while running. y/n'),
        'weights_path': (weights_path, f'Path to weights folder.'),
        'model_name': (model_name['video'], f'Name of model.'),
    }
    parser = ArgumentParser()
    parser.add_argument('in_dir', type=str, help='Directory containing .avi\' to run inference on.')
    parser.add_argument('out_dir', type=str, help='Directory to output predictions to.')
    for k, (v, h) in args.items():
        h += f' default={v}'
        if isinstance(v, bool):
            parser.add_argument('--' + k.replace('_', '-'), action=BoolAction, default=v, help=h)
        else:
            parser.add_argument('--' + k.replace('_', '-'), type=type(v), default=v, help=h)
    args.update({k.replace('-', '_'): v for k, v in vars(parser.parse_args()).items()})
    if not torch.torch.cuda.is_available():
        args.update({'device': 'cpu'})
    get_args = lambda *l: {k: args[k] for k in l}

    # Run inference
    engine = VideoInferenceEngine(**get_args('weights_path', 'model_name', 'device'))
    engine.run_on_dir(**get_args('in_dir', 'out_dir', 'verbose'))
