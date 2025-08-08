from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import math
import time
import psutil
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
                 resume: bool = True,
                 elastic_batch: bool = True,
                 base_batch_size: int = 64,
                 max_batch_size: int = 512,
                 thermal_limit_c: int = 85,
                 lr_decay_patience: int = 3,
                 lr_decay_factor: float = 0.5,
                 profile_interval: int = 0,
                 keep_last_n: int = 3,
                 shard_threshold_mb: int = 200):
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
        self.elastic_batch = elastic_batch
        self.base_batch_size = base_batch_size
        self.max_batch_size = max_batch_size
        self.thermal_limit_c = thermal_limit_c
        self.lr_decay_patience = lr_decay_patience
        self.lr_decay_factor = lr_decay_factor
        self.profile_interval = profile_interval
        self._since_improve = 0
        self.keep_last_n = keep_last_n
        self.shard_threshold_mb = shard_threshold_mb
        self._initial_lr = None

    def _save_checkpoint(self, model, opt, ckpt_path: Path, epoch: int, tag: str, extra: Dict[str, Any]):
        state = {
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'scaler': self.scaler.state_dict(),
            'epoch': epoch,
            'best_loss': self.best_loss,
            'no_improve': self.no_improve_epochs,
            'meta': extra
        }
        torch.save(state, ckpt_path)
        # Shard if file large
        size_mb = ckpt_path.stat().st_size / (1024*1024)
        if size_mb > self.shard_threshold_mb:
            shard_dir = ckpt_path.parent / f"{ckpt_path.stem}_shards"
            shard_dir.mkdir(exist_ok=True)
            sd = model.state_dict()
            for k, v in sd.items():
                torch.save(v, shard_dir / f"{k.replace('.', '_')}.pt")
        # Rotate old checkpoints
        self._rotate_checkpoints(ckpt_path.parent, prefix="epoch_")

    def _rotate_checkpoints(self, directory: Path, prefix: str):
        ckpts = sorted([p for p in directory.glob(f"{prefix}*.pt")], key=lambda p: p.stat().st_mtime)
        if len(ckpts) > self.keep_last_n:
            for p in ckpts[:-self.keep_last_n]:
                try: p.unlink()
                except Exception: pass

    def _profile_epoch(self, model, dl, device: str, epoch: int):
        if self.profile_interval <= 0 or epoch % self.profile_interval != 0:
            return
        try:
            import torch.profiler as profiler
            activities = [profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(profiler.ProfilerActivity.CUDA)
            with profiler.profile(activities=activities, record_shapes=True) as prof:
                model.train()
                for i, (xb, yb) in enumerate(dl):
                    xb = xb.to(device); yb = yb.to(device)
                    with autocast(enabled=self.amp):
                        pred = model(xb).squeeze(-1)
                        loss = nn.functional.mse_loss(pred, yb)
                    if i > 5: break
            report = prof.key_averages().table(sort_by="cpu_time_total", row_limit=15)
            out_file = self.output_dir / f"profile_epoch{epoch}.txt"
            out_file.write_text(report)
            print(f"Saved profiler report to {out_file}")
        except Exception as e:
            print(f"Profiling failed: {e}")

    def _suggest_hyperparams(self, epoch_loss: float, epoch: int):
        # Simple heuristic suggestions
        suggestions = []
        if self._initial_lr and epoch_loss > self.best_loss * 1.02:
            suggestions.append("Consider lowering learning rate or increasing gradient_accum_steps")
        if self.elastic_batch and epoch % 5 == 0:
            suggestions.append("Evaluate memory headroom to further increase batch size")
        if suggestions:
            (self.output_dir / "suggestions.txt").open("a").write(f"Epoch {epoch}: " + " | ".join(suggestions) + "\n")

    def _init_ddp(self):
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if world_size > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
        return world_size

    def _rank(self):
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def _gpu_temp(self) -> Optional[float]:
        # Placeholder: would query nvidia-smi; on non-GPU systems return None
        return None

    def _should_throttle(self) -> bool:
        cpu_temp = None
        try:
            # psutil sensors_temperatures may not be available on macOS; guard access
            temps = psutil.sensors_temperatures() if hasattr(psutil, 'sensors_temperatures') else {}
            for k, arr in temps.items():
                if arr:
                    cpu_temp = max([t.current for t in arr if hasattr(t, 'current')])
        except Exception:
            cpu_temp = None
        gpu_temp = self._gpu_temp()
        over = False
        if cpu_temp and cpu_temp > self.thermal_limit_c:
            over = True
        if gpu_temp and gpu_temp > self.thermal_limit_c:
            over = True
        return over

    def _adjust_batch(self, current_bs: int, improvement: bool) -> int:
        if not self.elastic_batch:
            return current_bs
        if improvement and current_bs < self.max_batch_size:
            return min(current_bs * 2, self.max_batch_size)
        if self._should_throttle() and current_bs > self.base_batch_size:
            return max(current_bs // 2, self.base_batch_size)
        return current_bs

    def train(self, df, features, target, seq_len: int, batch_size: int, lr: float):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        world_size = self._init_ddp()
        ds = SeqDataset(df, features, target, seq_len)
        if len(ds) == 0:
            raise ValueError('Not enough data')
        sampler = DistributedSampler(ds) if world_size > 1 else None
        current_bs = batch_size
        dl = DataLoader(ds, batch_size=current_bs, shuffle=(sampler is None), sampler=sampler)
        model = LSTMBaseline(input_dim=len(features)).to(device)
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ.get('LOCAL_RANK','0'))])
        opt = AdamW(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=self.lr_decay_factor, patience=self.lr_decay_patience, verbose=True)

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
        if not self._initial_lr:
            self._initial_lr = opt.param_groups[0]['lr']

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            total = 0.0; count = 0
            opt.zero_grad()
            start_time = time.time()
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
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} loss={epoch_loss:.6f} bs={current_bs} lr={opt.param_groups[0]['lr']:.2e} time={elapsed:.2f}s")
            # Early stopping
            improved = epoch_loss < self.best_loss - 1e-6
            if improved:
                self.best_loss = epoch_loss
                self.no_improve_epochs = 0
                if self._rank() == 0:
                    epoch_ckpt = ckpt_path.parent / f"epoch_{epoch}.pt"
                    self._save_checkpoint(model, opt, epoch_ckpt, epoch+1, tag="epoch", extra={"improved": True})
                    # update main pointer
                    if ckpt_path.exists():
                        try: ckpt_path.unlink()
                        except Exception: pass
                    epoch_ckpt.rename(ckpt_path)
            else:
                self.no_improve_epochs += 1
            scheduler.step(epoch_loss)
            # Elastic batch adjust after scheduler step
            new_bs = self._adjust_batch(current_bs, improved)
            if new_bs != current_bs:
                current_bs = new_bs
                dl = DataLoader(ds, batch_size=current_bs, shuffle=(sampler is None), sampler=sampler)
            # Profiling & suggestions
            if self._rank() == 0:
                self._profile_epoch(model, dl, device, epoch)
                self._suggest_hyperparams(epoch_loss, epoch)
            if self.no_improve_epochs >= self.early_stop_patience:
                if self._rank() == 0:
                    print('Early stopping triggered')
                break
        return model
