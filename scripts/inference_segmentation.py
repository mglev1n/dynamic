import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
from pathlib import Path
from typing import Union
from argparse import ArgumentParser
import argparse
import tqdm

from echonet.utils import loadvideo, savevideo, latexify


## Segmentation:
# Def: Frame-by-frame Semantic Segmentation of the Left Ventricle
# model_name: 'deeplabv3_resnet50'
# frames: 64
# period: 2


# relative paths to weights for various models
weights_path = Path(__file__).parent.parent / 'weights'
model_name = {'segmentation': 'deeplabv3_resnet50_random',
              'video': 'r2plus1d_18_32_2_pretrained'}


class SegmentationInferenceEngine:

    def __init__(
            self, weights_path=weights_path, model_name: str = model_name['segmentation'], batch_size=10, device=None
    ) -> None:

        model_path = weights_path / (model_name + '.pt')
        if isinstance(model_path, str):
            model_path = Path(model_path)
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.model = self.load_model()
        self.batch_size = batch_size

    def load_model(self):
        """Loads the model to be used for inference into memory.

        Returns:
            NamedTuple: See torch.Module.load_state_dict() for details
        """

        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=False)
        model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1
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
        os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "size"), exist_ok=True)
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

        latexify()
        with open(os.path.join(out_dir, "size.csv"), "w") as g:
            g.write("Filename,Frame,Size,Systole,Diastole\n")
            for path in tqdm.tqdm(paths):
                filename = path.name
                video = loadvideo(str(path))
                # c, f, h, w = video.shape
                video = video.transpose(1, 0, 2, 3)
                # f, c, h, w = video.shape
                x = self._preprocess_video(video)
                x = torch.from_numpy(x.astype(np.float32)).to(torch.float)
                logit = np.concatenate([self.model(x[i:(i + self.batch_size), :, :, :].to(self.device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], self.batch_size)])
                logit = logit.squeeze()
                video = self._postprocess_video_with_segmentations(video, logit)

                # Compute size of segmentation per frame
                size = (logit > 0).sum((1, 2))

                trim_min, trim_max, trim_range = self._get_trim_min_max_range(size)
                systole = self._get_systole(size, trim_range)
                diastole = self._get_diastole(size, trim_range)

                # Write sizes and frames to file
                for (frame, s) in enumerate(size):
                    g.write("{},{},{},{},{}\n".format(filename, frame, s, 1 if frame in systole else 0, 1 if frame in diastole else 0))
                    g.flush()
                    

                # Plot sizes
                self._plot_size(size, systole, filename, out_dir)

                # Normalize size to [0, 1]
                size = self._normalize_size(size)

                # Iterate the frames in this video
                for (f, s) in enumerate(size):
                    video = self._postprocess_video_with_size(video, size, systole, f, s)

                # Rearrange dimensions and save
                video = video.transpose(1, 0, 2, 3)
                video = video.astype(np.uint8)
                savevideo(os.path.join(out_dir, "videos", filename), video, 50)

    def _get_video_mean_and_std(self, video):
        video = video.reshape(3, -1)
        mean = np.mean(video, axis=1)
        std = np.std(video, axis=1)  # type: np.ndarray
        mean = mean.astype(np.float32)
        std = std.astype(np.float32)
        return mean, std

    def _preprocess_video(self, video):
        video = video.astype(np.float32)
        mean, std = self._get_video_mean_and_std(video)
        video -= mean.reshape(1, 3, 1, 1)
        video /= std.reshape(1, 3, 1, 1)
        return video

    def _postprocess_video_with_segmentations(self, video, logit):
        # Get frames, channels, height, and width
        f, c, h, w = video.shape  # pylint: disable=W0612
        assert c == 3

        # Put two copies of the video side by side
        video = np.concatenate((video, video), 3)

        # If a pixel is in the segmentation, saturate blue channel
        # Leave alone otherwise
        video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])  # pylint: disable=E1111

        # Add blank canvas under pair of videos
        video = np.concatenate((video, np.zeros_like(video)), 2)
        return video

    def _get_trim_min_max_range(self, size):
        # Identify systole frames with peak detection
        trim_min = sorted(size)[round(len(size) ** 0.05)]
        trim_max = sorted(size)[round(len(size) ** 0.95)]
        trim_range = trim_max - trim_min
        return trim_min, trim_max, trim_range

    def _get_systole(self, size, trim_range):
        return set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])
      
    def _get_diastole(self, size, trim_range):
        return set(scipy.signal.find_peaks(size, distance=20, prominence=(0.50 * trim_range))[0])

    def _plot_size(self, size, systole, filename, out_dir):
        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
        ylim = plt.ylim()
        for s in systole:
            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
        plt.ylim(ylim)
        plt.title(os.path.splitext(filename)[0])
        plt.xlabel("Seconds")
        plt.ylabel("Size (pixels)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "size", os.path.splitext(filename)[0] + ".pdf"))
        plt.close(fig)

#     def _normalize_size(self, size):
#         size -= size.min()
#         size = size / size.max()
#         size = 1 - size
#         return size

    def _normalize_size(self, size):
        print(size)
        size -= size.min()
        max_size = size.max()
        if max_size == 0:
            return size  # return original size array if all elements are zero
        size = size / max_size
        size = 1 - size
        return size

    def _postprocess_video_with_size(self, video, size, systole, f, s):
        # On all frames, mark a pixel for the size of the frame
        video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

        if f in systole:
            # If frame is computer-selected systole, mark with a line
            video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

        def dash(start, stop, on=10, off=10):
            buf = []
            x = start
            while x < stop:
                buf.extend(range(x, x + on))
                x += on
                x += off
            buf = np.array(buf)
            buf = buf[buf < stop]
            return buf

        d = dash(115, 224)

        # Get pixels for a circle centered on the pixel
        r, c = skimage.draw.disk((int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))), 4.1)

        # On the frame that's being shown, put a circle over the pixel
        video[f, :, r, c] = 255.

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
        'batch_size': (10, 'Number of frames to run inference on at once.'),
        'weights_path': (weights_path, f'Path to weights folder.'),
        'model_name': (model_name['segmentation'], f'Name of model.'),
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
    engine = SegmentationInferenceEngine(**get_args('weights_path', 'model_name', 'batch_size', 'device'))
    engine.run_on_dir(
        **get_args('in_dir', 'out_dir', 'verbose'))
