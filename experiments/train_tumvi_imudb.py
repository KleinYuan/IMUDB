import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from models.imudb import Model
from data.data import get_TUMVI_IMUDB
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import pytz
import datetime
import uuid
import re
import socket

tz = pytz.timezone('US/Pacific')


def train(config_fp='configs/tumvi_imudb.yaml'):
    with open(config_fp) as f:
        config = yaml.safe_load(f)
    time = datetime.datetime.now(tz=tz).isoformat()
    version = "{}_{}_{}".format(config["stage"], config["experiment_name"], time)
    datasets_name = config["datasets_name"]
    experiment_uuid = uuid.uuid4().hex
    hostname = socket.gethostname()
    hostname = re.sub(r'[\W]+', '', hostname)  # remove invalid characters.
    note = config['note']
    logdir = config["logs_dir"] + '_' + hostname
    model_name = config['model_name']
    if 'experiment_name' in config:
        experiment_name = config['experiment_name']
    else:
        experiment_name = datasets_name + '_' + model_name
    debug_mode = False
    if 'debug' in config:
        debug_mode = config['debug']
    # Adding the experiment info to the experiment management log csv
    if not debug_mode:
        experiment_management_log_content = f"{experiment_uuid},{hostname},{version}," \
                                            f"{experiment_name},{logdir}/{experiment_name}/{version}/hparams.yaml,{note}\n"
        experiment_management_log = open(config["experiment_management_log"], "a")
        experiment_management_log.write(experiment_management_log_content)
        experiment_management_log.close()
        print(experiment_management_log_content)
    else:
        experiment_name = "debug/" + experiment_name
        logdir = "debug/" + logdir

    train_loader, val_loader, test_loader = get_TUMVI_IMUDB(config=config)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(logdir, name=experiment_name, version=version)
    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints_{hostname}/{experiment_name}/{version}",
        filename=experiment_name + "-{epoch:02d}-{val_loss:.6f}",
        save_top_k=3,
        mode="min",
    )
    model = Model(config)
    trainer = pl.Trainer(gpus=1, max_epochs=100000, logger=logger, callbacks=[lr_monitor, checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    train()
