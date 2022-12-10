import torch
from metrics import MultiAccuracy, MultiAUPRC, MultiF1Score, MultiRecall, MultiPrecision, TopKAccuracy

class Trainer:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.path_ckpt = args.path_ckpt
        self.best_ckpt = args.best_ckpt
        self.best_epoch = None
        self.criterions = {
            'MultiAccuracy': MultiAccuracy(num_classes=args.num_classes),
            'MultiAUPRC': MultiAUPRC(num_classes=args.num_classes),
            'MultiF1Score': MultiF1Score(num_classes=args.num_classes, average='macro'),
            'MultiRecall': MultiRecall(num_classes=args.num_classes, average='macro'),
            'MultiPrecision': MultiPrecision(num_classes=args.num_classes, average='macro'),
            'TopKAccuracy': TopKAccuracy(num_classes=args.num_classes, k=3),
        }
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def train_epoch(self, dataloader):
        train_loss = 0.
        self.model.train()
        metrics = {}
        trues, preds = [], []

        for batch in dataloader:
            xs = batch['x'].to(self.args.num_gpu)
            ys = batch['y'].to(self.args.num_gpu).long()
            self.optimizer.zero_grad()
            y_preds = self.model(xs)
            loss = self.model.fn_loss(y_preds, ys)
            loss.backward()
            self.optimizer.step()
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
            loss = self.model.fn_loss(y_preds, ys)
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
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(ckpt, self.args.path_ckpt + f"/{epoch}.ckpt")

    def load_checkpoint(self, epoch):
        ckpt = torch.load(self.args.path_ckpt + f"/{epoch}.ckpt")
        self.model.load_state_dict(ckpt['classifier'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
