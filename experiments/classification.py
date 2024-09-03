
from models.custom_models import get_model
import torch
from common.utils import get_exp_params, get_config, save_experiment_output, save_model_helpers, save_model_chkpt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
from common.loss_utils import ECCLoss
from transformers import CLIPProcessor, CLIPModel
from torch.optim.lr_scheduler import StepLR

class Classification:

    def __init__(self, train_dataset, val_dataset):
        cfg = get_config()
        self.data_dir = cfg['data_dir']
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        #self.test_dataset = test_dataset
        self.exp_params = get_exp_params()
        self.model_params = self.exp_params['model']
        self.device = cfg['device']
        self.num_classes = 102
        self.lamda1 = self.exp_params['train']['lamda1']
        self.lamda2 = self.exp_params['train']['lamda2']
        self.dim = 0
        #self.data_transform = transforms

    def __loss_fn(self, loss_name = 'cross-entropy'):
        if loss_name == 'cross-entropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_name == 'mse':
            return torch.nn.MSELoss()
        elif loss_name == 'l1':
            return torch.nn.L1Loss()
        elif loss_name == 'ecc':
            loss_fn = ECCLoss(self.num_classes, self.dim)
            return loss_fn
        else:
            raise SystemExit("Error: no valid loss function name passed! Check run.yaml")

    def __save_model_checkpoint(self, model_state, optimizer_state, chkpt_info):
        save_experiment_output(model_state, chkpt_info, False)
        save_model_helpers(optimizer_state, True)
        #os.remove(os.path.join(self.root_dir, "models/checkpoints/current_model.pt"))

    def __get_clip_features4classes(self):
        model_name = "openai/clip-vit-base-patch32"  # You can choose other available models
        num_text_feats = 512
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        lines = []
        with open(os.path.join(self.data_dir, 'processed_class_descriptions.txt')) as fp:
            lines = fp.readlines()
        text_features = torch.zeros(self.num_classes, num_text_feats)
        for i, line in enumerate(lines):
            desc = line.split(" : ")[1]
            inputs = processor(text=desc, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model.get_text_features(**inputs)
            outputs = outputs / outputs.norm(p = 2, dim = -1, keepdim = True)
            outputs = outputs.cpu()
            text_features[i] = outputs 
        return text_features



    def __conduct_training(self, model, optimizer, train_loader, val_loader, tr_len, val_len):
        num_epochs = self.exp_params['train']['num_epochs']
        epoch_interval = self.exp_params['train']['epoch_interval']
        
        self.text_features = self.__get_clip_features4classes().to(self.device)
        ce_loss_fn = self.__loss_fn()
        loss_fn = self.__loss_fn(self.exp_params['train']['loss'])
        trlosshistory, vallosshistory, valacchistory = [], [], []
        print('text features size', self.text_features.size(), '\n')
        scheduler = StepLR(optimizer,
            step_size = self.exp_params['train']['lr_step'],
            gamma = self.exp_params['train']['lr_decay'])

        for epoch in range(num_epochs):

            model.train()
            tr_loss, val_loss, val_acc = 0.0, 0.0, 0.0

            for _, batch in enumerate(tqdm(train_loader, desc = '\t\tRunning through training set', position = 0, leave = True, disable = True)):
                optimizer.zero_grad()
                imgs = batch['img'].float().to(self.device)
                olabels = batch['olabel'].to(self.device)
                lbls = batch['label'].to(self.device)
                op,feats = model(imgs)
                #print('op sz', op.size(), olabels.size(), batch['label'].size(), feats.size())
                celoss = ce_loss_fn(op, olabels)
                mcc, clg, _, _ = loss_fn(feats, op, lbls)
                loss = celoss + (self.lamda1 * mcc) + (self.lamda2 * clg)
                loss.backward()
                optimizer.step()
                tr_loss += (loss.item() * imgs.size(0))
            if self.exp_params['train']['enable_lr_decay']:
                scheduler.step()


            tr_loss /= tr_len
            trlosshistory.append(tr_loss)

            model.eval()

            for _, batch in enumerate(tqdm(val_loader, desc = '\t\tRunning through validation set', position = 0, leave = True, disable = True)):
                imgs = batch['img'].float().to(self.device)
                olabels = batch['olabel'].to(self.device)
                lbls = batch['label'].to(self.device)
                op, feats = model(imgs)
                celoss = ce_loss_fn(op, olabels)
                mcc, clg, _, _ = loss_fn(feats, op, lbls)
                loss = celoss + (self.lamda1 * mcc) + (self.lamda2 * clg)
                #loss = loss_fn(op, olabels)
                val_loss += (loss.item() * imgs.size(0))
                pred_label = torch.argmax(op, 1)
                #print('label size', correct_label.size(), pred_label.size())
                val_acc += (lbls == pred_label).sum()

            val_loss /= val_len
            val_acc /=  val_len
            vallosshistory.append(val_loss)
            valacchistory.append(val_acc.item())

            if epoch % epoch_interval == 0:
                print(f'\tEpoch {epoch} Training Loss: {tr_loss}')
                print(f"\tEpoch {epoch} Validation Loss: {val_loss}\n")

        model_info = {
            'trlosshistory': trlosshistory,
            'vallosshistory': vallosshistory,
            'valacchistory': valacchistory,
            'last_epoch': -1
        }
        self.__save_model_checkpoint(model, optimizer, model_info)


    def run_fgvc_pipeline(self):
        model_name = self.model_params['name']
        model = get_model(102, model_name)
        self.dim = model.dim
        optimizer = torch.optim.Adam(model.parameters(),
            lr = self.model_params['lr'],
            weight_decay = self.model_params['weight_decay'],
            amsgrad = self.model_params['amsgrad'])
        batch_size = self.exp_params['train']['batch_size']

        train_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = False)
        val_loader = DataLoader(self.val_dataset, batch_size = batch_size, shuffle = False)
        tr_len = len(self.train_dataset)
        val_len = len(self.val_dataset)

        print('Training of classifier...\n')

        self.__conduct_training(model, optimizer, train_loader, val_loader, tr_len, val_len)

        torch.cuda.empty_cache()
