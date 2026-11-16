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
from typing import Dict, Optional

from src.model import RiboNN
from src.data import RiboNNDataModule

torch.set_float32_matmul_precision('high')


def train_model_nested_cv(
    config: Dict,
    outer_cv_folds: int = 10,  # ignored if 'fold' column already exists in dm.df
    inner_cv_folds: int = 9,
    stop_after_n_inner_cv_folds: Optional[int] = None,
    # The following three arguments are useful if your training was interrupted
    # and you want to resume training
    max_test_fold_to_skip: int = -1,
    max_valid_fold_to_skip: int = -1,
    ckpt_path: Optional[str] = None,
    random_state: int = 42,
    max_epochs: int = 200,
    patience: int = 20,
    log_model: bool = True,
    experiment_name: str = "experiment_name",
) -> None:
    """Train the RiboNN model using parameters specified in config"""

    # Create the data module
    dm = RiboNNDataModule(config)
    config["num_targets"] = dm.num_targets
    config["len_after_conv"] = dm.get_sequence_length_after_ConvBlocks()
    config["outer_cv_folds"] = outer_cv_folds
    config["inner_cv_folds"] = inner_cv_folds
    # config['patience'] = patience
    # config['max_epochs'] = max_epochs


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
                if test_fold < max_test_fold_to_skip or (
                    test_fold == max_test_fold_to_skip
                    and valid_fold <= max_valid_fold_to_skip
                ):
                    continue

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
                early_stopping = EarlyStopping(
                    monitor="val_r2", mode="max", verbose=True, patience=patience
                )
                checkpoint = ModelCheckpoint(
                    save_top_k=1,
                    verbose=True,
                    monitor="val_r2",
                    mode="max",
                )

                # Start and log the child run
                with mlflow.start_run(
                    experiment_id=experiment_id,
                    nested=True,
                    run_name=f"{test_fold}-{valid_fold}",
                    # description=f"Test fold: {test_fold}. Validation fold: {valid_fold}",
                ) as child_run:
                    # The MLFlow logger can log more hyperparameters than mlflow autolog.
                    mlf_logger = MLFlowLogger(
                        experiment_name=experiment_name,
                        run_name=f"{test_fold}-{valid_fold}",
                        run_id=child_run.info.run_id,
                    )

                    # Create the trainer
                    trainer = pl.Trainer(
                        accelerator='gpu', 
                        devices=1,
                        max_epochs=max_epochs,
                        # fast_dev_run=True,
                        gradient_clip_val=0.5,
                        callbacks=(
                            [
                                lr_logger,
                                checkpoint,
                                early_stopping,
                            ]
                            if patience > 0
                            else [
                                lr_logger,
                                checkpoint,
                            ]
                        ),
                        logger=mlf_logger,
                        log_every_n_steps=20,
                    )

                    # Autolog. Avoding logging compiled models
                    mlflow.pytorch.autolog(log_models=False)

                    # Create the model
                    model = RiboNN(**config)

                    # Train
                    trainer.fit(
                        model,
                        datamodule=dm,
                        ckpt_path=ckpt_path,
                    )

                    # The ckpt_path is only used once when resuming the training
                    ckpt_path = None

                    # Reload the best model
                    model = RiboNN.load_from_checkpoint(
                        checkpoint.best_model_path
                    )

                    # # Save state_dict for later use
                    # state_dict = model.state_dict()

                    # # Correct the key names
                    # for key in list(state_dict.keys()):
                    #     state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(
                    #         key
                    #     )

                    # # Recreate an eager model that can be logged by mlflow
                    # model = RiboNN(**config)

                    # # Load the state_dict
                    # model.load_state_dict(state_dict)
                    model.eval()

                    # Manually log the model
                    if log_model:
                        mlflow.pytorch.log_model(model, "model")

                    # Test
                    trainer = pl.Trainer(
                        accelerator='gpu', 
                        devices=1,
                        num_nodes=1,
                        logger=mlf_logger,  # important
                    )
                    trainer.test(model, datamodule=dm)
