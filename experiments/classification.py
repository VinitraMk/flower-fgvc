from models.ots_models import get_model
import torch
from common.utils import get_exp_params, get_config, save_experiment_output, save_model_helpers, save_experiment_chkpt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os

class Classification:
    
    def __init__(self, train_dataset, val_dataset, test_dataset):
        cfg = get_config()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.exp_params = get_exp_params()
        self.model_params = self.exp_params['model']
        self.device = cfg['device']
        
    def __loss_fn(self, loss_name = 'cross-entropy'):
        if loss_name == 'cross-entropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_name == 'mse':
            return torch.nn.MSELoss()
        elif loss_name == 'l1':
            return torch.nn.L1Loss()
        else:
            raise SystemExit("Error: no valid loss function name passed! Check run.yaml")
        
    def __save_model_checkpoint(self, model_state, optimizer_state, chkpt_info):
        save_experiment_output(model_state, chkpt_info, self.exp_params,
            True)
        save_model_helpers(optimizer_state, True)
        #os.remove(os.path.join(self.root_dir, "models/checkpoints/current_model.pt"))
        
    def __conduct_training(self, model, optimizer, train_loader, val_loader, tr_len, val_len):
        num_epochs = self.exp_params['train']['num_epochs']
        epoch_interval = self.exp_params['train']['epoch_interval']
        loss_fn = self.__loss_fn()
        trlosshistory, vallosshistory, valacchistory = [], [], []
        
        for epoch in range(num_epochs):
            
            model.train()
            tr_loss, val_loss, val_acc = 0.0, 0.0, 0.0

            for _, batch in enumerate(tqdm(train_loader, desc = '\t\tRunning through training set', position = 0, leave = True, disable = True)):
                optimizer.zero_grad()
                imgs = batch['img'].float().to(self.device)
                olabels = batch['olabel']
                op = model(imgs)
                print('op sz', op.size(), olabels.size())
                loss = loss_fn(op, olabels)
                loss.backward()
                optimizer.step()
                tr_loss += (loss.item() * batch.size(0))

                
            tr_loss /= tr_len
            trlosshistory.append(tr_loss)
            
            model.eval()
            
            for _, batch in enumerate(tqdm(val_loader, desc = '\t\tRunning through validation set', position = 0, leave = True, disable = True)):
                imgs = batch['img'].float().to(self.device)
                olabels = batch['olabel']
                op = model(imgs)
                loss = loss_fn(op, olabels)
                val_loss += (loss.item() * batch.size(0))
                correct_label = batch['label']
                pred_label = torch.argmax(op, 1).item()
                val_acc += (correct_label == pred_label).sum()
                
            val_loss /= val_len
            val_acc /=  val_len
            vallosshistory.append(val_loss)
            valacchistory.append(val_acc)
            
            if epoch % epoch_interval == 0:
                print(f'\tEpoch {epoch+1} Training Loss: {tr_loss}')
                print(f"\tEpoch {epoch+1} Validation Loss: {val_loss}")
                
        model_info = {
            'trlosshistory': torch.tensor(trlosshistory),
            'vallosshistory': torch.tensor(vallosshistory),
            'valacchistory': torch.tensor(valacchistory),
            'last_epoch': -1
        }
        self.__save_model_checkpoint(model.state_dict(), optimizer.state_dict(), model_info)
                
            
    def run_fgvc_pipeline(self):
        model_name = self.model_params['name']
        model, _ = get_model(model_name)
        optimizer = torch.optim.Adam(lr = self.model_params['lr'], weight_decay = self.model_params['weight_decay'], amsgrad = self.model_params['amsgrad'])
        
        batch_size = self.exp_params['train']['batch_size']
        
        train_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = False)
        val_loader = DataLoader(self.val_dataset, batch_size = batch_size, shuffle = False)
        tr_len = len(self.train_dataset)
        val_len = len(self.val_dataset)
        
        self.__conduct_training(model, optimizer, train_loader, val_loader, tr_len, val_len)
        
        torch.cuda.empty_cache()
        