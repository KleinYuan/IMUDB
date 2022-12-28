from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataloaders import euroc_limu_bert
from dataloaders import euroc_imudb
from dataloaders import tumvi_imudb

pl.seed_everything(1234)


def get_EUROC_LIMU_BERT(config):
    train_dataset = euroc_limu_bert.EUROCDataset(config=config,  mode='train')
    train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=config["data"]["shuffle"]["train"], drop_last=True, num_workers=16)
    val_dataset = euroc_limu_bert.EUROCDataset(config=config,  mode='val')
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=config["data"]["shuffle"]["val"], drop_last=True, num_workers=16)
    test_dataset = euroc_limu_bert.EUROCDataset(config=config, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"], shuffle=config["data"]["shuffle"]["test"], drop_last=True, num_workers=16)
    return train_loader, val_loader, test_loader


def get_EUROC_IMUDB(config):
    train_dataset = euroc_imudb.EUROCDataset(config=config,  mode='train')
    train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=config["data"]["shuffle"]["train"], drop_last=True, num_workers=16)
    val_dataset = euroc_imudb.EUROCDataset(config=config,  mode='val')
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=config["data"]["shuffle"]["val"], drop_last=True, num_workers=16)
    test_dataset = euroc_imudb.EUROCDataset(config=config, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"], shuffle=config["data"]["shuffle"]["test"], drop_last=True, num_workers=16)
    return train_loader, val_loader, test_loader


def get_TUMVI_IMUDB(config):
    train_dataset = tumvi_imudb.TUMVIDataset(config=config, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"],
                              shuffle=config["data"]["shuffle"]["train"], drop_last=True, num_workers=16)
    val_dataset = tumvi_imudb.TUMVIDataset(config=config, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"],
                            shuffle=config["data"]["shuffle"]["val"], drop_last=True, num_workers=16)
    test_dataset = tumvi_imudb.TUMVIDataset(config=config, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"],
                             shuffle=config["data"]["shuffle"]["test"], drop_last=True, num_workers=16)
    return train_loader, val_loader, test_loader
