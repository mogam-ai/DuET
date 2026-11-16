"""
Created By  : Dinghai Zheng
Created Date: 08/04/2023
Description : Module for transfer learning
"""

from typing import Dict, Optional

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.model_selection import KFold

from src.model import RiboNN
from src.data import RiboNNDataModule

torch.set_float32_matmul_precision("high")


def unfreeze_batchnorm_layers(model):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            for param in child.parameters():
                param.requires_grad = True
        unfreeze_batchnorm_layers(child)


def create_transfer_learning_model(
    config: Dict,
    local_state_dict_path: str = "models/human/0ad046d67ba7481881d5c7f918acedb4/model.pth",
) -> pl.LightningModule:

    # Create an instance of the pretrained multi-task model
    pretrain_config = config.copy()
    pretrain_config["num_targets"] = 78
    loaded_model = RiboNN(**pretrain_config)

    # Load the state dict
    loaded_model.load_state_dict(torch.load(local_state_dict_path))

    # Freeze the model except the BatchNorm layers
    loaded_model.freeze()
    unfreeze_batchnorm_layers(loaded_model)

    # Create a new model
    model = RiboNN(**config)

    # Replace the initial_conv and middle_convs of the new model with the pretrained model
    model.initial_conv = loaded_model.initial_conv
    model.middle_convs = loaded_model.middle_convs
    model.train()

    return model


def transfer_learning(
    config: Dict,
    pretrain_run_df: pd.DataFrame,
    outer_cv_folds: int = 10,
    inner_cv_folds: int = 9,
    stop_after_n_inner_cv_folds: Optional[int] = None,
    random_state: int = 42,
    phase_1_epochs: int = 50,
    max_epochs: int = 200,
    patience: int = 50,
    experiment_name: str = "experiment_name",
) -> None:
    """Train single-task models using transfer learning"""

    # Create the data module
    config["outer_cv_folds"] = outer_cv_folds
    config["inner_cv_folds"] = inner_cv_folds
    target_column = config["target_column_pattern"]
    dm = RiboNNDataModule(config)
    dm.df = dm.df.dropna(subset=[target_column])
    config["num_targets"] = dm.num_targets
    config["len_after_conv"] = dm.get_sequence_length_after_ConvBlocks()

    if not "fold" in dm.df.columns:
        np.random.seed(random_state)
        dm.df["fold"] = np.random.randint(0, outer_cv_folds - 1, size=len(dm.df))

    client = mlflow.MlflowClient()

    # Create a new experiment if it does not exist
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    print(f"Using experiment_id: {experiment_id}")


    # Start and log the parent run
    with mlflow.start_run(
        experiment_id=experiment_id,
    ) as parent_run:
        for test_fold in sorted(dm.df.fold.unique()):
            data_test = dm.df.query("fold == @test_fold").reset_index(drop=True)
            data_test["split"] = "test"
            data_train_valid = dm.df.query("fold != @test_fold").reset_index(
                drop=True
            )
            k_fold = KFold(
                n_splits=inner_cv_folds,
                shuffle=True,
                random_state=random_state,
            )
            for valid_fold, (train_idx, val_idx) in enumerate(
                k_fold.split(data_train_valid)
            ):
                if stop_after_n_inner_cv_folds is not None:
                    if valid_fold >= stop_after_n_inner_cv_folds:
                        break

                config["test_fold"] = test_fold
                config["valid_fold"] = valid_fold

                # Update the splits
                data_train_valid.loc[train_idx, "split"] = "train"
                data_train_valid.loc[val_idx, "split"] = "valid"
                dm.df = pd.concat(
                    [data_test, data_train_valid], axis=0, ignore_index=True
                )

                # Create callbacks
                lr_logger = LearningRateMonitor(logging_interval="step")
                checkpoint = ModelCheckpoint(
                    save_top_k=1,
                    verbose=True,
                    monitor="val_r2",
                    mode="max",
                )
                early_stopping = EarlyStopping(
                    monitor="val_r2", mode="max", verbose=True, patience=patience
                )

                # Start and log the child run
                with mlflow.start_run(
                    experiment_id=experiment_id,
                    nested=True,
                    description=f"Test fold: {test_fold}. Validation fold: {valid_fold}",
                ) as child_run:
                    # The MLFlow logger can log more hyperparameters than mlflow autolog.
                    mlf_logger = MLFlowLogger(
                        experiment_name=experiment_name,
                        run_id=child_run.info.run_id,
                    )

                    # Autolog. Avoding logging compiled models
                    mlflow.pytorch.autolog(log_models=False)

                    # Create the model
                    sub_run_df = pretrain_run_df.query(
                        "`params.test_fold` == @test_fold and `params.valid_fold` == @valid_fold"
                    ).reset_index(drop=True)

                    if config.get("use_random_weights", False):
                        model = RiboNN(**config)
                    else:
                        model = create_transfer_learning_model(
                            config,
                            f"models/human/{sub_run_df.run_id.values[0]}/state_dict.pth"
                        )
                    
                    # Train the head
                    trainer = pl.Trainer(
                        max_epochs=phase_1_epochs,
                        gradient_clip_val=0.5,
                        callbacks=[
                            lr_logger,
                        ],
                        logger=mlf_logger,
                        log_every_n_steps=10,
                    )
                    trainer.fit(
                        model,
                        datamodule=dm,
                    )

                    # Train the whole model with lower learning rate
                    model.unfreeze()
                    model.lr = 0.00001
                    (
                        model.optimizers,
                        model.lr_schedulers,
                    ) = model.configure_optimizers()
                    trainer = pl.Trainer(
                        max_epochs=max_epochs - phase_1_epochs,
                        gradient_clip_val=0.5,
                        callbacks=[
                            lr_logger,
                            early_stopping,
                            checkpoint,
                        ],
                        logger=mlf_logger,
                        log_every_n_steps=10,
                    )
                    trainer.fit(
                        model,
                        datamodule=dm,
                    )

                    # Reload the best model
                    saved_checkpoint = torch.load(checkpoint.best_model_path)
                    model.load_state_dict(saved_checkpoint["state_dict"])
                    model.eval()

                    # Manually log the model
                    mlflow.pytorch.log_model(model, "model")

                    # Test
                    trainer = pl.Trainer(
                        devices=1,
                        num_nodes=1,
                        logger=mlf_logger,  # important
                    )
                    trainer.test(model, datamodule=dm)