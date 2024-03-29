#!/usr/bin/env python3

import io
import os
import argparse
from pathlib import Path
from typing import Optional

import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import matplotlib.pyplot as plt
import seaborn as sns


def create_confusion_matrics_img(conf_mat):
    buf = io.BytesIO()

    fig, ax = plt.subplots()
    sns.heatmap(conf_mat, annot=True, ax=ax, cmap=plt.cm.Blues, fmt="d")

    plt.savefig(buf, format="jpeg")
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    img = transforms.ToTensor()(img)
    return img


class CCSNDataSet(Dataset):

    labels = ["Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St"]
    img_ext = ["*.png", "*.jpg", "*.jpeg"]

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []  # tuple(Path-img, index-label)

        self.setup()

    def setup(self):
        self.images = []

        for p in self.root_dir.iterdir():
            if not p.is_dir():
                continue
            if p.name not in self.labels:
                continue
            label_idx = self.labels.index(p.name)
            for img_path in self._get_file(p):
                self.images.append((img_path, label_idx))

    def _get_file(self, path: Path):
        for ext in self.img_ext:
            for p in path.glob(ext):
                yield p

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label_idx = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label_idx)


class CCSNDataModule(pl.LightningDataModule):
    """DataSet for Cirrus Cumulus Stratus Nimbus"""

    url_default = "https://dvn-cloud.s3.amazonaws.com/..."

    def __init__(
        self,
        data_dir: str = "./dataset",
        batch_size: int = 32,
        valid_percent: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_percent = valid_percent
        self.num_workers = 8
        self.transform = transforms.Compose(
            [
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomRotation((-10, 10)),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

        # check
        if 1 <= self.valid_percent < 0:
            raise ValueError("valid_percent out from range [0-1)")

        # color, w, h
        self.dims = (3, 256, 256)

    # def prepare_data(self):
    #     # download
    #     # CCSN_v2
    #     torchvision.datasets.utils.download_and_extract_archive(
    #         self.url_default,
    #         self.data_dir,
    #         filename="ccsn.zip",
    #         remove_finished=True
    #     )

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            ds = CCSNDataSet(self.data_dir, transform=self.transform)
            tlen = int(len(ds) * (1 - self.valid_percent))
            vlen = len(ds) - tlen
            self.ds_train, self.ds_val = random_split(ds, [tlen, vlen])

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if self.valid_percent == 0:
            return
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class Backbone(torch.nn.Module):
    def __init__(self, count_labels: int = 2):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(nn.Linear(512, count_labels))

    def forward(self, x):
        return self.backbone(x)


class CloudNet(pl.LightningModule):
    """Classifier Cloud"""

    def __init__(self, backbone, learning_rate=1e-3, count_labels=2):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.count_labels = count_labels

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

        self.valid_cm = torchmetrics.ConfusionMatrix(
            num_classes=self.count_labels, multilabel=False
        )
        self.train_cm = torchmetrics.ConfusionMatrix(
            num_classes=self.count_labels, multilabel=False
        )

    def forward(self, x):
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)

        # y_hat = y_hat.argmax(1)
        self.train_acc.update(y_hat, y)
        self.train_cm.update(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        tb = self.logger.experiment
        a = self.train_cm.compute().detach().cpu().type(torch.int)
        tb.add_image(
            "train_confusion_matrix",
            create_confusion_matrics_img(a),
            global_step=self.current_epoch,
        )
        self.train_cm.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)

        # y_hat = y_hat.argmax(1)
        self.valid_acc.update(y_hat, y)
        self.valid_cm.update(y_hat, y)

        self.log("valid_loss", loss)

    def validation_epoch_end(self, outputs):
        self.log("valid_acc", self.valid_acc, on_step=False, on_epoch=True)

        tb = self.logger.experiment
        a = self.valid_cm.compute().detach().cpu().type(torch.int)
        tb.add_image(
            "valid_confusion_matrix",
            create_confusion_matrics_img(a),
            global_step=self.current_epoch,
        )
        self.valid_cm.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False
        )
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        return parser

    def infer(self, img) -> int:
        """Infering for img

        Args
            img - PIL image

        Return
            label index
        """
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        img = transform(img).unsqueeze(0)
        y = self.forward(img)
        y = y.argmax(1)
        return int(y[0])


def cli_main():
    pl.seed_everything(42)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--valid-percent", default=0.1, type=float)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CloudNet.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    d_module = CCSNDataModule(
        "./dataset", args.batch_size, valid_percent=args.valid_percent
    )

    # model
    count_labels = len(CCSNDataSet.labels)
    model = CloudNet(Backbone(count_labels), args.learning_rate, count_labels)

    # training
    checkpoint_end = pl.callbacks.ModelCheckpoint()
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_end], num_sanity_val_steps=0
    )
    trainer.fit(model, d_module)

    # to onnx
    print("Convert to onnx")
    input_sampe = torch.randn((1, 3, 256, 256))
    model.to_onnx(
        os.path.join(checkpoint_end.dirpath, "model_last.onnx"),
        input_sampe,
        export_params=True,
    )

    # testing


if __name__ == "__main__":
    cli_main()
