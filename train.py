

import torch
import random
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from collections import deque
import os
from config import Config
from utils.dataloader import VQADataset, get_video_frames, seed_worker
from model import QwenVQAModel
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

def set_seed(seed: int):

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, config, output_path=None):
        self.config = config
        set_seed(config.SEED)
        
        self.output_dir = Path(output_path) if output_path else config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs_dir = self.output_dir / "epochs"
        self.epochs_dir.mkdir(exist_ok=True)
        
        self.wrapper = QwenVQAModel(config)
        self.optimizer = AdamW(list(self.wrapper.model.parameters()) + 
                               list(self.wrapper.regressor.parameters()), lr=config.LR)
        self.scaler = torch.amp.GradScaler(device=config.DEVICE)
        self.mse_loss = torch.nn.MSELoss()
        self.pair_buffer = deque(maxlen=config.PAIR_BUFFER_SIZE)
        

        g = torch.Generator()
        g.manual_seed(config.SEED)
        self.train_loader = DataLoader(VQADataset(config.TRAIN_CSV, config.TRAIN_VID_DIR),
                                       batch_size=config.BATCH_SIZE, shuffle=True,
                                       worker_init_fn=seed_worker, generator=g)
        

        full_val_ds = VQADataset(config.VAL_CSV, config.VAL_VID_DIR)
        

        val_subset = torch.utils.data.Subset(full_val_ds, range(min(config.VAL_LEN, len(full_val_ds))))
        

        self.val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)

        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.train_losses, self.val_losses = [], []
        
        self.val_srcc = []
        self.val_plcc = []
        self.val_krcc = []
        self.val_rmse = []
        
        self.load_checkpoint()
        self.print_parameter_summary()

    def print_parameter_summary(self):
        trainable_params = 0
        all_param = 0
        
        for name, param in self.wrapper.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"\n{'='*40}")
        print(f"PARAMETER SUMMARY")
        print(f"{'='*40}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"All parameters:   {all_param:,}")
        print(f"Percentage:       {100 * trainable_params / all_param:.4f}%")
        print(f"{'='*40}\n")

    def load_checkpoint(self):
        ckpt_path = self.output_dir / "checkpoint_latest.pt"
        if ckpt_path.exists():
            print(f"Loading checkpoint from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=self.config.DEVICE)
            self.wrapper.model.load_state_dict(ckpt['model_state'])
            self.wrapper.regressor.load_state_dict(ckpt['regressor_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.scaler.load_state_dict(ckpt['scaler_state'])
            self.start_epoch = ckpt['epoch'] + 1
            self.train_losses = ckpt.get('train_losses', [])
            self.val_losses = ckpt.get('val_losses', [])
            self.best_val_loss = ckpt.get('best_val_loss', float("inf"))
            self.val_srcc = ckpt.get('val_srcc', [])
            self.val_plcc = ckpt.get('val_plcc', [])
            self.val_krcc = ckpt.get('val_krcc', [])
            self.val_rmse = ckpt.get('val_rmse', [])

    def save_checkpoint(self, epoch):
        ckpt = {
            'epoch': epoch,
            'model_state': self.wrapper.model.state_dict(),
            'regressor_state': self.wrapper.regressor.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'val_srcc': self.val_srcc,
            'val_plcc': self.val_plcc,
            'val_krcc': self.val_krcc,
            'val_rmse': self.val_rmse
        }
        torch.save(ckpt, self.output_dir / "checkpoint_latest.pt")
        
        epoch_dir = self.epochs_dir / f"epoch{epoch}"
        epoch_dir.mkdir(exist_ok=True)
        self.wrapper.model.save_pretrained(epoch_dir / "lora")
        torch.save(self.wrapper.regressor.state_dict(), epoch_dir / "regressor.pt")

    def run_step(self, video_path, mos_score, backward=True):
        frames, fps, _ = get_video_frames(video_path, num_frames=self.config.NUM_KEYFRAMES)
        conv = [self.wrapper.embedder.format_model_input(
            text=self.config.PROMPT, video=frames, fps=fps, max_frames=len(frames)
        )]
        processed = {k: v.to(self.config.DEVICE) for k, v in self.wrapper.embedder._preprocess_inputs(conv).items()}
        
        with torch.amp.autocast(device_type="cuda", dtype=self.config.DTYPE):
            pred = self.wrapper(processed)
            target = torch.as_tensor(mos_score, device=pred.device, dtype=self.config.DTYPE)
            loss_mse = self.mse_loss(pred, target)
            
            loss_rank = torch.tensor(0.0, device=pred.device)
            if self.config.USE_RANKING and len(self.pair_buffer) > 0:
                p_pred, p_target = random.choice(self.pair_buffer)
                y = 1.0 if target.item() > p_target else -1.0
                loss_rank = torch.clamp(self.config.RANK_MARGIN - y * (pred - p_pred), min=0.0)
            
            loss = loss_mse + self.config.LAMBDA_RANK * loss_rank

        if backward:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        self.pair_buffer.append((pred.detach(), target.item()))
        
        return loss.item(), pred.item(), target.item()

    def train(self):
        for epoch in range(self.start_epoch, self.config.MAX_EPOCHS + 1):
            self.wrapper.model.train(); self.wrapper.regressor.train()
            t_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
            for vid_path, mos in pbar:
                step_loss, _, _ = self.run_step(vid_path[0], mos[0], backward=True)
                t_loss += step_loss
            
            avg_t = t_loss / len(self.train_loader)
            self.train_losses.append(avg_t)

            self.wrapper.model.eval(); self.wrapper.regressor.eval()
            v_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for vid_path, mos in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                    step_loss, pred, target = self.run_step(vid_path[0], mos[0], backward=False)
                    v_loss += step_loss
                    val_preds.append(pred)
                    val_targets.append(target)
            
            avg_v = v_loss / len(self.val_loader)
            self.val_losses.append(avg_v)
            
            y_pred = np.array(val_preds)
            y_true = np.array(val_targets)
            
            if np.std(y_pred) == 0 or np.std(y_true) == 0:
                plcc, srcc, krcc = 0.0, 0.0, 0.0
            else:
                plcc, _ = pearsonr(y_pred, y_true)
                srcc, _ = spearmanr(y_pred, y_true)
                krcc, _ = kendalltau(y_pred, y_true)
                
            rmse = np.sqrt(np.mean((y_pred - y_true)**2))
            
            self.val_plcc.append(float(plcc))
            self.val_srcc.append(float(srcc))
            self.val_krcc.append(float(krcc))
            self.val_rmse.append(float(rmse))

            print(f"Epoch {epoch} complete. Train Loss: {avg_t:.4f}, Val Loss: {avg_v:.4f}")
            print(f"Metrics -> PLCC: {plcc:.4f} | SRCC: {srcc:.4f} | KRCC: {krcc:.4f} | RMSE: {rmse:.4f}")
            
            if avg_v < self.best_val_loss:
                self.best_val_loss = avg_v
                best_dir = self.output_dir / "best_model"
                best_dir.mkdir(exist_ok=True)
                self.wrapper.model.save_pretrained(best_dir / "lora")
                torch.save(self.wrapper.regressor.state_dict(), best_dir / "regressor.pt")
            
            self.save_checkpoint(epoch)

            metrics = {
                "train_loss": self.train_losses, 
                "val_loss": self.val_losses,
                "val_srcc": self.val_srcc,
                "val_plcc": self.val_plcc,
                "val_krcc": self.val_krcc,
                "val_rmse": self.val_rmse
            }
            with open(self.output_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Train Loss")
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("VQA Training Progress")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.output_dir / "loss_curve.png")
            plt.close()

if __name__ == "__main__":
    trainer = Trainer(Config())
    trainer.train()