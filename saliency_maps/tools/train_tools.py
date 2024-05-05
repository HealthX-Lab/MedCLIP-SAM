import torch

class Recorder:
    def __init__(self, out_path, batch_size, lr, weight_decay, best_only=False, save_period=1, within_epoch=0):
        self.out_path = out_path
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.best_only = best_only
        self.lr = lr
        self.best_acc = 0
        self.best_loss = 0
        self.best_val_acc = 0
        self.save_period = save_period if within_epoch ==0 else save_period*10
        self.within_epoch = within_epoch
    def update(self, acc, loss, val_acc, epoch):
        self.save_epoch = (epoch+1)%self.save_period == 0 or (self.within_epoch and (epoch%10)%self.within_epoch==0) if not self.best_only else False

        if acc >= self.best_acc or val_acc>= self.best_val_acc or loss <= self.best_loss or self.save_epoch:
            self.save = True
        else:
            self.save = False

        if acc >= self.best_acc:
            self.save_best_train_acc = True
            self.best_acc = acc
        else:   
            self.save_best_train_acc = False

        if loss <= self.best_loss:
            self.best_loss = loss
            self.save_best_loss = True
        else:
            self.save_best_loss = False

        if val_acc >= self.best_val_acc:
            self.save_best_val_acc = True
            self.best_val_acc = val_acc
        else:
            self.save_best_val_acc = False


    @staticmethod
    def save_model(epoch, avg_loss, model_sd, opt_sd, out_path):
        torch.save(
                {
                'epoch': epoch+1,
                'model_state_dict': model_sd,
                'optimizer_state_dict': opt_sd,
                'loss': avg_loss,
                }, out_path
                )

    def save_checkpoints(self, epoch, avg_loss, model, optimizer):
        if self.save_best_train_acc:
            out_name = f"{self.out_path}/best_train_acc_clip_b{self.batch_size}_lr{self.lr}_wd{self.weight_decay}.pt"
            self.save_model(epoch, avg_loss, model.state_dict(), optimizer.state_dict(), out_name)
        if self.save_best_val_acc:
            out_name = f"{self.out_path}/best_val_acc_clip_b{self.batch_size}_lr{self.lr}_wd{self.weight_decay}.pt"
            self.save_model(epoch, avg_loss, model.state_dict(), optimizer.state_dict(), out_name)
        if self.save_best_loss:
            out_name = f"{self.out_path}/best_loss_clip_b{self.batch_size}_lr{self.lr}_wd{self.weight_decay}.pt"
            self.save_model(epoch, avg_loss, model.state_dict(), optimizer.state_dict(), out_name)
        if self.save_epoch:   
            out_name = f"{self.out_path}/clip_e{epoch:03d}_b{self.batch_size}_lr{self.lr}_wd{self.weight_decay}.pt"
            self.save_model(epoch, avg_loss, model.state_dict(), optimizer.state_dict(), out_name)