import torchvision
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime

import config
from data import IntelDataModule
from model import TransferLearningModel
from callbacks import PrintingCallback


if __name__ == "__main__":
    time = datetime.now().strftime("%Y%m%d%H%M")
    logger = TensorBoardLogger("tb_logs", name=f"model_{time}")

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    backbone = torchvision.models.efficientnet_b0(weights=weights)
    
    backbone_transforms = weights.transforms()
    train_transforms = transforms.Compose(
        [
            backbone_transforms,
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=45)
        ]
    )

    data_module = IntelDataModule(
        data_dir=config.DATA_DIR,
        train_transform=train_transforms,
        test_transform=backbone_transforms,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    model = TransferLearningModel(
        backbone=backbone,
        image_size = backbone_transforms.crop_size[0],
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE
    )

    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        logger=logger,
        callbacks=[
            PrintingCallback(),
            EarlyStopping(monitor="valid_loss")
        ]
    )

    trainer.fit(model, data_module)
    trainer.validate(model, data_module)
    trainer.test(model, data_module)