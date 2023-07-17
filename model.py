import torch
from torch import nn, optim

import pytorch_lightning as pl
import torchmetrics
import torchvision


class TransferLearningModel(pl.LightningModule):
    def __init__(
            self,
            backbone: torchvision.models,
            image_size: tuple=(3, 150, 150),
            num_classes: int=2,
            learning_rate: float=0.0001
    ):
        super().__init__()
        self.backbone = backbone
        self.image_size = image_size
        self.num_classes = num_classes
        self.lr = learning_rate

        self.feature_extractor = self.backbone.features
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(
                in_features=self._get_classifier_input(),
                out_features=num_classes))

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def _get_classifier_input(self):
        for module in self.backbone.classifier:
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                break
        
        return in_features

    def forward(self, x):
        x = self.backbone(x)

        return x
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)

        loss = self.criterion(scores, y)
        accuracy = self.accuracy(preds, y)

        return loss, accuracy
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, accuracy = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        if batch_idx % 100 == 0:
            x = x[:4]
            grid = torchvision.utils.make_grid(x.view(-1, 3, self.image_size, self.image_size))
            self.logger.experiment.add_image("earth images", grid, self.global_step)

        return {"loss": loss, "accuracy": accuracy, "y": y} 
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "valid_loss": loss,
                "valid_accuracy": accuracy
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return loss
    
    def test_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, batch_idx)
        self.log("test_accuracy", accuracy)

        return loss, accuracy
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)