import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from networks.limu_bert import LIMUBertModel4Pretrain
from box import Box


class Model(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        # Below shall be moved to configs
        self.save_hyperparameters("config")
        self.starting_learning_rate = float(config['model']['hyper_params']['starting_learning_rate'])
        self.hyper_params = Box(config['model']['hyper_params'])
        self.limu_bert = LIMUBertModel4Pretrain(self.hyper_params)

        self.mse_loss = F.mse_loss
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        """
        mask_seqs.size(): torch.Size([B, seq, 6])
        masked_pos.size(): torch.Size([B, seq * mask_ratio])
        gt_imu_seq.size(): torch.Size([B, seq * mask_ratio, 6])
        normed_imu_seq.size(): torch.Size([B, seq, 6])

        """
        mask_seqs = batch['inputs']['mask_seqs'] # (B, seq, 6)
        masked_pos = batch['inputs']['masked_pos'] # (B, seq * mask_ratio)
        gt_imu = batch['outputs']['seq']  # (B, seq * mask_ratio, 6)
        normed_imu_seq = batch['outputs']['normed_imu_seq']  # (B, Seq, 6)

        hat_imu_MLM = self.limu_bert.forward(mask_seqs, masked_pos)
        hat_imu_denoise = self.limu_bert.forward(normed_imu_seq)

        MLM_loss = self.mse_loss(gt_imu, hat_imu_MLM)
        denoise_loss = self.mse_loss(normed_imu_seq[:, -1, :], hat_imu_denoise[:, -1, :]) * float(self.hyper_params.denoise_loss_weights)

        loss = MLM_loss + denoise_loss

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_MLM_loss", MLM_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_denoise_loss", denoise_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        mask_seqs.size(): torch.Size([B, seq, 6])
        masked_pos.size(): torch.Size([B, seq * mask_ratio])
        gt_imu_seq.size(): torch.Size([B, seq * mask_ratio, 6])
        normed_imu_seq.size(): torch.Size([B, seq, 6])
        """
        mask_seqs = batch['inputs']['mask_seqs']  # (B, seq, 6)
        masked_pos = batch['inputs']['masked_pos']  # (B, seq * mask_ratio)
        gt_imu = batch['outputs']['seq']  # (B, seq * mask_ratio, 6)
        normed_imu_seq = batch['outputs']['normed_imu_seq']  # (B, Seq, 6)

        hat_imu_MLM = self.limu_bert.forward(mask_seqs, masked_pos)
        hat_imu_denoise = self.limu_bert.forward(normed_imu_seq)

        MLM_loss = self.mse_loss(gt_imu, hat_imu_MLM)
        denoise_loss = self.mse_loss(normed_imu_seq[:, -1, :], hat_imu_denoise[:, -1, :]) * float(self.hyper_params.denoise_loss_weights)

        loss = MLM_loss + denoise_loss

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_MLM_loss", MLM_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_denoise_loss", denoise_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.starting_learning_rate)

        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                              T_0=int(self.hyper_params.T_0),
                                                                              T_mult=int(self.hyper_params.T_mult),
                                                                              eta_min=float(self.hyper_params.eta_min)),
            "interval": "epoch",
            "frequency": 1,
            'name': 'learning_rate'
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
