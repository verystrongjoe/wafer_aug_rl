import torch


class Trainer:
    def __init__(self, args, model, optim, fn_loss, criterions):
        self.args = args
        self.model = model
        self.optim = optim
        self.path_ckpt = args.path_ckpt
        self.best_ckpt = args.best_ckpt
        self.best_epoch = None
        self.fn_loss = fn_loss
        self.criterions = criterions

    def train_epoch(self, dataloader):
        train_loss = 0.
        self.model.train()
        metrics = {}
        trues, preds = [], []

        for batch in dataloader:
            xs = batch['x'].to(self.args.num_gpu)
            ys = batch['y'].to(self.args.num_gpu)
            self.optim.zero_grad()
            y_preds = self.model(xs)
            loss = self.fn_loss(y_preds, ys)
            loss.backward()
            self.optim.step()
            train_loss += loss.item()
            trues += [ys.cpu().detach()]
            preds += [y_preds.cpu().detach()]
        out = {'loss': train_loss / len(dataloader)}
        trues = torch.cat(trues, dim=0)
        preds = torch.cat(preds, dim=0)
        for metric_name, metric_function in self.criterions.items():
            out[metric_name] = metric_function(preds, trues).item()
        return out

    def valid_epoch(self, dataloader):
        valid_loss = 0.
        trues, preds = [], []
        self.model.eval()
        for batch in dataloader:
            xs = batch['x'].to(self.args.num_gpu)
            ys = batch['y'].to(self.args.num_gpu)
            y_preds = self.model(xs)
            loss = self.fn_loss(y_preds, ys)
            valid_loss += loss.item()
            trues += [ys.cpu().detach()]
            preds += [y_preds.cpu().detach()]
        out = {'loss': valid_loss / len(dataloader)}
        trues = torch.cat(trues, dim=0)
        preds = torch.cat(preds, dim=0)
        for metric_name, metric_function in self.criterions.items():
            out[metric_name] = metric_function(preds, trues).item()
        return out

    def save_checkpoint(
            self,
            epoch: int
            ):
        ckpt = {
            'classifier': self.model.state_dict(),
            'optimizer': self.optim.state_dict()
        }
        torch.save(ckpt, self.args.path_ckpt + f"/{epoch}.ckpt")

    def load_checkpoint(self, epoch):
        ckpt = torch.load(self.args.path_ckpt + f"/{epoch}.ckpt")
        self.model.load_state_dict(ckpt['classifier'])
        self.optim.load_state_dict(ckpt['optimizer'])
