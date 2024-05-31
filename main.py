import av
import os
import time
import torch
import random
import torchvision
import itertools
from Pix2Gif import Pix2GifModel
from giphy_scraper import downlad_dataset
from datasets import Dataset, IterableDataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def dataset_generator():
    n_frames = 24
    paths = [f"dataset/{f}" for f in os.listdir("dataset")]
    random.shuffle(paths)

    containers = (av.open(path) for path in paths)
    containers = (c for c in containers if c.streams.video[0].frames >= n_frames)
    gifs = (itertools.islice(c.decode(video=0), 0, n_frames, 1) for c in containers)
    gifs = ([np.array(Image.fromarray(frame.to_ndarray(format="rgb24")).resize((240, 240))) for frame in gif] for gif in gifs)
    gifs = (np.rollaxis(np.asarray(gif), -1, 1) / 256 for gif in gifs)
    for gif in gifs:
        yield {'frame': gif[0], 'gif': gif}


def save_video(frames, path):
    output_container = av.open(path, 'w')

    # Create a video stream with the desired properties (e.g., resolution, frame rate)
    output_stream = output_container.add_stream('h264', rate=10)
    output_stream.width = 240
    output_stream.height = 240

    # Write each frame to the video stream
    for frame in frames:
        # Convert the frame to the appropriate format for encoding
        frame_rgb = frame.astype(np.uint8)
        frame_yuv = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')

        # Encode and write the frame to the video stream
        for packet in output_stream.encode(frame_yuv):
            output_container.mux(packet)

    # Flush the remaining packets
    for packet in output_stream.encode():
        output_container.mux(packet)

    # Close the video stream
    output_container.close()


if __name__ == '__main__':
    # downlad_dataset('waterfall')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Pix2GifModel(3, 3, device=device)

    dataset = IterableDataset.from_generator(dataset_generator).with_format("torch")
    data_loader = DataLoader(dataset, batch_size=8, num_workers=1, prefetch_factor=100, pin_memory=True)
    path = 'results'
    for epoch in range(100):
        epoch_start_time = time.time()  # timer for entire epoch
        for input in data_loader:  # inner loop within one epoch
            model.set_input(input)
            output = model.optimize_parameters()

        print('losses:', model.get_current_losses().items())

        video = random.choice(output).detach().cpu()
        plt.imshow(torchvision.utils.make_grid(video, 6).permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(f"{path}/images/{epoch}.png")

        with torch.no_grad():
            video = torch.moveaxis(video * 256, 1, -1).numpy()
            save_video(video, f'{path}/videos/{epoch}.mp4')

        if epoch % 5 == 0:
            mode_path = f"{path}/model_{epoch}.pt"
            print(f"saving model {mode_path}")
            torch.save(model, mode_path)
