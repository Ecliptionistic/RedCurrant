import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight


def seed_everything(seed):
    random.seed(seed); 
    np.random.seed(seed)
    torch.manual_seed(seed); 
    torch.cuda.manual_seed_all(seed)


class Trainer:
    '''
    Цикл обучения для нейросетевых моделей (вроде wav2vec 2.0)
    '''
    def __init__(self, epochs=30, lr=1e-4, encoder_lr=None, weight_decay=1e-4, patience=6,
                 unfreeze_epoch=None, grad_clip=1.0, amp=True, balanced_loss=True, device=None):
        self.epochs = epochs
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.wd = weight_decay
        self.patience = patience
        self.unfreeze_epoch = unfreeze_epoch
        self.grad_clip = grad_clip
        self.balanced_loss = balanced_loss
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")

        cuda = self.device.type == "cuda"
        self.amp = amp and cuda

        if cuda and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16 
        else:
            self.amp_dtype = torch.float16

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp)

    def _criterion(self, y_train, n_classes):
        if not self.balanced_loss:
            return nn.CrossEntropyLoss()
        
        present = np.unique(y_train)
        cw = np.ones(n_classes, dtype=np.float32)
        cw[present] = compute_class_weight("balanced", classes=present, y=y_train)

        return nn.CrossEntropyLoss(weight=torch.tensor(cw, device=self.device))

    def fit(self, net, train_loader, val_loader, y_train, n_classes):
        net.to(self.device)
        groups = (net.param_groups(self.lr, self.encoder_lr) if hasattr(net, "param_groups")
                  else [{"params": list(net.parameters()), "lr": self.lr}])
        opt = torch.optim.AdamW(groups, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.amp_dtype == torch.float16)
        crit = self._criterion(y_train, n_classes)
        best_ua, best_state, bad = -1.0, None, 0

        for ep in range(1, self.epochs + 1):
            if self.unfreeze_epoch and ep == self.unfreeze_epoch:
                net.set_encoder_trainable(True)
                print(f"    [ep {ep}] encoder unfrozen")
            net.train()
            total = 0.0

            for x, lens, y in train_loader:
                x, lens, y = (t.to(self.device, non_blocking=True) for t in (x, lens, y))
                opt.zero_grad(set_to_none=True)
                with self._autocast():
                    loss = crit(net(x, lens).float(), y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                scaler.step(opt); scaler.update()
                total += loss.item()

            sched.step()

            probs, yv = self.predict(net, val_loader)
            ua = balanced_accuracy_score(yv, probs.argmax(1))
            print(f"    ep {ep:02d}  loss={total / len(train_loader):.4f}  val_UA={ua:.4f}")

            if ua > best_ua:
                best_ua, bad = ua, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

            elif not (self.unfreeze_epoch and ep < self.unfreeze_epoch):
                bad += 1
                if bad >= self.patience:
                    print(f"    early stop at ep {ep}")
                    break
                
        net.load_state_dict(best_state)
        return best_ua

    @torch.no_grad()
    def predict(self, net, loader):
        net.to(self.device).eval()
        P, Y = [], []
        for x, lens, y in loader:
            with self._autocast():
                logits = net(x.to(self.device), lens.to(self.device))
            P.append(logits.float().softmax(-1).cpu()); Y.append(y)
        return torch.cat(P).numpy(), torch.cat(Y).numpy()
