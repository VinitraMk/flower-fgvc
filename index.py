# %%
#set up some imports

import numpy as np
import torch
import random

# custom imports

from common.utils import init_config, get_exp_params
from datautils.dataset import FlowerDataset
from datautils.datareader import get_file_paths

# %%
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# %%
config_params = init_config()
print(config_params)

# %%
# read experiment params

exp_params = get_exp_params()
print('Experiment parameters\n')
print(exp_params)

# %%
train_fns, val_fns, test_fns, _ = get_file_paths(config_params['data_dir'])
ftr_dataset = FlowerDataset(config_params['data_dir'], train_fns)
val_dataset = FlowerDataset(config_params['data_dir'], val_fns)
test_dataset = FlowerDataset(config_params['data_dir'], test_fns)
sm_trlen = int(0.1 * len(ftr_dataset))
sm_telen = int(0.01 * len(test_dataset))
sm_vlen = int(0.1 * len(val_dataset))

sm_ftr_dataset = torch.utils.data.Subset(ftr_dataset, list(range(sm_trlen)))
sm_val_dataset = torch.utils.data.Subset(val_dataset, list(range(sm_vlen)))
sm_test_dataset = torch.utils.data.Subset(test_dataset, list(range(sm_telen)))
print('Full train dataset length', len(ftr_dataset))
print('Subset train dataset length', sm_trlen)
print('Full validation dataset length', len(val_dataset))
print('Subset validation dataset length', sm_vlen)
print('Full test dataset length', len(test_dataset))
print('Subset test dataset length', sm_telen)

# %%
# model finetuning

