from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import math
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from src.models.trainers.train_lstm import SeqDataset
from src.models.architectures.lstm_baseline import LSTMBaseline

class AdvancedTrainer:
    def __init__(self,
                 output_dir: Path,
                 gradient_accum_steps: int = 1,
                 amp: bool = True,
                 max_epochs: int = 50,
                 early_stop_patience: int = 10,
                 resume: bool = True):
        self.output_dir = output_dir
        self.gradient_accum_steps = gradient_accum_steps
        self.amp = amp and torch.cuda.is_available()
        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience
        self.resume = resume
        self.scaler = GradScaler(enabled=self.amp)
        self.best_loss = float('inf')
        self.no_improve_epochs = 0
        self.start_epoch = 1

    def _init_ddp(self):
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if world_size > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
        return world_size

    def _rank(self):
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def train(self, df, features, target, seq_len: int, batch_size: int, lr: float):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        world_size = self._init_ddp()
        ds = SeqDataset(df, features, target, seq_len)
        if len(ds) == 0:
            raise ValueError('Not enough data')
        sampler = DistributedSampler(ds) if world_size > 1 else None
        dl = DataLoader(ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)
        model = LSTMBaseline(input_dim=len(features)).to(device)
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ.get('LOCAL_RANK','0'))])
        opt = AdamW(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.output_dir / 'checkpoint.pt'
        if self.resume and ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state['model'])
            opt.load_state_dict(state['opt'])
            self.scaler.load_state_dict(state['scaler'])
            self.best_loss = state.get('best_loss', self.best_loss)
            self.no_improve_epochs = state.get('no_improve', 0)
            self.start_epoch = state.get('epoch', 1)
            print(f"Resumed from epoch {self.start_epoch}")

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            total = 0.0; count = 0
            opt.zero_grad()
            for step, (xb, yb) in enumerate(dl, start=1):
                xb = xb.to(device); yb = yb.to(device)
                with autocast(enabled=self.amp):
                    pred = model(xb).squeeze(-1)
                    loss = loss_fn(pred, yb) / self.gradient_accum_steps
                self.scaler.scale(loss).backward()
                if step % self.gradient_accum_steps == 0:
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.scaler.step(opt)
                    self.scaler.update()
                    opt.zero_grad()
                total += loss.item() * len(xb)
                count += len(xb)
            epoch_loss = total / count if count else math.inf
            if self._rank() == 0:
                print(f"Epoch {epoch} loss={epoch_loss:.6f}")
            # Early stopping
            improved = epoch_loss < self.best_loss - 1e-6
            if improved:
                self.best_loss = epoch_loss
                self.no_improve_epochs = 0
                if self._rank() == 0:
                    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': self.scaler.state_dict(), 'epoch': epoch+1, 'best_loss': self.best_loss, 'no_improve': self.no_improve_epochs}, ckpt_path)
            else:
                self.no_improve_epochs += 1
            if self.no_improve_epochs >= self.early_stop_patience:
                if self._rank() == 0:
                    print('Early stopping triggered')
                break
        return model
