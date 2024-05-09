from data_loader.datasets.tii_dataset import TIIDataset
import os
from data_loader.utils import readlines
import torch
import random
from torch.utils.data import DataLoader
import multiprocessing
import timeit


class TIIGateDataset(TIIDataset):
    def __init__(self, batch_size, img_width=448, img_height=256, return_Ks=False):
        fpath = os.path.join("../../data_loader", "splits", "tii", "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))

        super().__init__("/data/aderik/drone-racing-dataset/data", train_filenames, img_height, img_width,
                         [0], batch_size, [0])

        self.K = torch.from_numpy(self.K)
        if self.scales != [0]:
            raise NotImplementedError("Gate TII Dataset only supports for [0] scales")

        self.K[0, :] *= self.width
        self.K[1, :] *= self.height
        self.inv_K = torch.linalg.inv(self.K)

        self.return_Ks = return_Ks
        self.batch_size = batch_size

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        """

        inputs = {}

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        # inputs[("folder")] = folder
        inputs[("frame_index")] = frame_index

        if self.is_train:
            # do not to forget to modify your GT labels
            do_flip = self.do_flip and random.random() > 0.5
            do_color_aug = self.do_color_aug and random.random() > 0.25
        else:
            do_flip = False
            do_color_aug = False

        inputs[("color", -1)] = self.get_color(folder, frame_index, do_flip)
        inputs[("tstamp")] = self.get_timestamp(folder, frame_index)

        # Assuming 0 scale
        if self.return_Ks:
            inputs[("K", 0)] = self.K
            inputs[("inv_K", 0)] = self.inv_K

        if self.do_color_aug and self.is_train:
            self.preprocess_with_augmentation(inputs, do_color_aug)
        else:
            self.preprocess(inputs)

        for _ in self.frame_idxs:
            del inputs[("color", -1)]

        return inputs

    def preprocess(self, inputs):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, i = k
                inputs[(n, 0)] = self.to_tensor(inputs[(n, i)])

    def preprocess_with_augmentation(self, inputs, do_color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, i = k
                inputs[(n, 0)] = self.to_tensor(inputs[(n, i)])

                if do_color_aug:
                    inputs[(n + "_aug", 0)] = self.to_tensor(self.color_aug(inputs[(n, i)]))
                    # inputs[(n + "_aug", im, 0)] = self.to_tensor(inputs[(n, im, i)])
                else:
                    inputs[(n + "_aug", 0)] = self.to_tensor(inputs[(n, i)])


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cpu_count = multiprocessing.cpu_count()
    gpu_count = torch.cuda.device_count()

    batch_size = 64

    dataset = TIIGateDataset(batch_size)

    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=cpu_count)

    # Warmup
    for batch_idx, inputs in enumerate(train_loader):
        if batch_idx == 0:
            inputs[('color', 0)] = inputs[('color', 0)].to(device)
            # plt.figure()
            # plt.imshow(inputs[('color', 0)][0].cpu().detach().squeeze(), cmap="gray")
            # plt.show()
    end_time = timeit.default_timer()

    start_time = timeit.default_timer()
    for batch_idx, inputs in enumerate(train_loader):
        if batch_idx == 0:
            inputs[('color', 0)] = inputs[('color', 0)].to(device)
            # plt.figure()
            # plt.imshow(inputs[('color', 0)][0].cpu().detach().squeeze(), cmap="gray")
            # plt.show()
    end_time = timeit.default_timer()
    print(end_time - start_time)