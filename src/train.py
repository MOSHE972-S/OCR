"""
מנוע אימון עם:
- שמירת Checkpoint בכל epoch
- המשך מנקודת עצירה
- Mixed Precision (GPU בלבד)
- זיהוי GPU חלש/CPU אוטומטי
- קריאה דרך QThread עם signals
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PyQt6.QtCore import QThread, pyqtSignal

from src.model import MyCRNN
from src.dataset import HandwritingDataset, NUM_CLASSES, collate_fn

CHECKPOINT_DIR = "checkpoints"
WEIGHTS_DIR    = "weights"


def _get_device():
    """בחר מכשיר לפי הזמינות — CUDA > Intel XPU > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Intel GPU דרך IPEX
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
    except Exception:
        pass
    return torch.device("cpu")


def _get_batch_size(device):
    """Batch size לפי סוג המכשיר."""
    if device.type == "cuda":
        try:
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return 8 if gb < 2 else 16
        except Exception:
            return 8
    if device.type == "xpu":
        # Intel HD Graphics — זיכרון משותף עם RAM, נשמור על batch קטן
        return 8
    return 8   # CPU


class TrainWorker(QThread):
    """Worker שרץ בthread נפרד ושולח עדכונים ל-GUI."""
    epoch_done   = pyqtSignal(int, int, float)   # epoch, total, loss
    batch_done   = pyqtSignal(int, int)           # batch, total_batches
    log_msg      = pyqtSignal(str)
    finished_ok  = pyqtSignal()
    error        = pyqtSignal(str)

    def __init__(self, csv_path, epochs=50, resume=True,
                 input_type="word", parent=None):
        super().__init__(parent)
        self.csv_path   = csv_path
        self.epochs     = epochs
        self.resume     = resume
        self.input_type = input_type
        self._stop      = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self._train()
        except Exception as e:
            self.error.emit(str(e))

    def _train(self):
        device     = _get_device()
        batch_size = _get_batch_size(device)
        self.log_msg.emit(f"🖥️  מכשיר: {device} | Batch: {batch_size}")

        model = MyCRNN(NUM_CLASSES).to(device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=1e-4
        )
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        # Mixed Precision: CUDA או Intel XPU (דרך IPEX)
        use_amp = device.type in ("cuda", "xpu")
        if device.type == "xpu":
            try:
                import intel_extension_for_pytorch as ipex
                model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float16)
            except Exception:
                use_amp = False
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

        start_epoch = 0
        ckpt_path   = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")

        # ── טעינת Checkpoint ──────────────────────────────
        if self.resume and os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch = ck["epoch"] + 1
            self.log_msg.emit(
                f"▶️  ממשיך מ-Epoch {start_epoch} (checkpoint נטען)"
            )
        else:
            self.log_msg.emit("🆕 מתחיל אימון חדש")

        # ── DataLoader ────────────────────────────────────
        dataset = HandwritingDataset(self.csv_path, self.input_type, True)
        if len(dataset) == 0:
            self.error.emit("❌ אין נתוני אימון ב-CSV.")
            return
        loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn,
            num_workers=0
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3,
            steps_per_epoch=len(loader),
            epochs=self.epochs - start_epoch,
            pct_start=0.1
        )

        best_loss = float("inf")

        for epoch in range(start_epoch, self.epochs):
            if self._stop:
                self.log_msg.emit("⏹️  האימון נעצר על ידי המשתמש.")
                break

            model.train()
            total_loss = 0.0

            for batch_idx, (images, targets, tgt_len) in enumerate(loader):
                if self._stop:
                    break

                images  = images.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    preds = model(images).permute(1, 0, 2)
                    in_len = torch.full(
                        (preds.size(1),), preds.size(0), dtype=torch.long
                    )
                    loss = criterion(
                        preds.log_softmax(2), targets, in_len, tgt_len
                    )

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                self.batch_done.emit(batch_idx + 1, len(loader))

            avg_loss = total_loss / max(len(loader), 1)
            self.epoch_done.emit(epoch + 1, self.epochs, avg_loss)

            # ── שמירת Checkpoint בכל epoch ────────────────
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss":      avg_loss,
            }, ckpt_path)

            # ── שמירת המודל הטוב ביותר ─────────────────────
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs(WEIGHTS_DIR, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(WEIGHTS_DIR, "best_model.pth")
                )

        self.finished_ok.emit()
