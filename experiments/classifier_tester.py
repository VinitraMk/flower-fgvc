#script for testing the model

from common.utils import get_exp_params, get_accuracy, get_config, get_model_filename
from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from common.loss_utils import ECCLoss

class ModelTester:

    def __init__(self, model, te_dataset, data_transform = None, result_sample_len = 5):
        cfg = get_config()
        
        self.device = cfg['device']
        self.te_dataset = te_dataset
        self.model = model.to(self.device)
        self.model.eval()
        self.exp_params = get_exp_params()
        self.lamda1 = self.exp_params['train']['lamda1']
        self.lamda2 = self.exp_params['train']['lamda2']
        self.telen = len(te_dataset)
        self.te_loader = DataLoader(self.te_dataset,
            batch_size = self.exp_params['train']['batch_size'],
            shuffle = False
        )
        self.output_dir = cfg['output_dir']
        self.data_transform = data_transform
        self.result_sample_len = result_sample_len
        self.num_classes = 102
        self.dim = model.dim

    def __loss_fn(self, loss_name = 'cross-entropy'):
        if loss_name == 'cross-entropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_name == 'mse':
            return torch.nn.MSELoss()
        elif loss_name == 'l1':
            return torch.nn.L1Loss()
        elif loss_name == 'ecc':
            loss_fn = ECCLoss(102, self.dim)
            return loss_fn
        else:
            raise SystemExit("Error: no valid loss function name passed! Check run.yaml")

    def test_and_plot(self):
        loss_fn = self.__loss_fn(self.exp_params["train"]["loss"])
        ce_loss_fn = self.__loss_fn()
        running_loss = 0.0
        acc = 0
        for _, batch in enumerate(tqdm(self.te_loader, desc = '\t\tRunning through validation set', position = 0, leave = True, disable = True)):
            with torch.no_grad():
                olabels = batch['olabel'].to(self.device)
                imgs = batch['img'].to(self.device)
                op, feats = self.model(imgs)
                lbls = batch['label'].to(self.device)
                celoss = ce_loss_fn(op, olabels)
                mcc, clg, _, _ = loss_fn(feats, op, lbls)
                loss = celoss + (self.lamda1 * mcc) + (self.lamda2 * clg)
                running_loss += (loss.item() * imgs.size(0))
                pred_label = torch.argmax(op, 1)
                acc += (get_accuracy(pred_label, lbls) * lbls.size(0))
        acc /= self.telen
        running_loss /= self.telen
        print("\nTest Loss:", running_loss/len(self.te_loader))
        print("Test Accuracy:", acc, "\n")




