import matplotlib.pyplot as plt
import torch
from models.ots_models import get_model
import torchvision.transforms as transforms
from common.utils import get_transforms, convert_to_grascale, get_exp_params

def layer_visualizer(img, args):

    model, _ = get_model(args.model_name, True)
    module_names = list(model.features.named_modules())[0][1]
    #print('model features', list(model.features.named_modules())[0][1])
    if model.features != None:
        model_features = model.features
        model_features.type(torch.FloatTensor)
    else:
        raise("model doesn't have features attribute")
        exit()

    for param in model_features.parameters():
        param.requires_grad = False

    c_transform, c_inv_transform = get_transforms(args.content_size)
    img = c_transform(img)[None]

    unnormalize = transforms.Compose([
        transforms.Lambda(convert_to_grascale),
        transforms.ToPILImage(),
        ])

    #img = img[None]
    x = img
    #print(list(model_features._modules.keys()))
    for i, layer in enumerate(model_features._modules.values()):
        #print('layer', layer)
        print(f'\nOutput from module {i}: {module_names[i]}')
        op = layer(x)
        x = op
        feature_map = op.squeeze(0)
        feature_sum = torch.sum(feature_map, 0)
        processed_img = feature_sum / feature_map.shape[0]
        #print('feature op size', op.size(), feature_map.size(), processed_img.size())

        plt.axis('off')
        rescaled_img = unnormalize(processed_img.data.cpu())
        plt.imshow(rescaled_img)
        plt.show()

class Visualization:

    def __init__(self, model_info, model_history = {}):
        self.exp_params = get_exp_params()
        num_epochs = self.exp_params['train']['num_epochs']
        self.bestm_tlh = model_info['trlosshistory'][-num_epochs:]
        self.bestm_vlh = model_info['vallosshistory'][-num_epochs:]
        self.model_history = model_history if model_history != {} else None

    def __plot_loss_history(self):
        num_epochs = self.exp_params["train"]["num_epochs"]
        plt.clf()
        print("\nModel results\n\n")
        plt.plot(list(range(num_epochs)), self.bestm_tlh, color="red", label="Best model training loss history")
        plt.plot(list(range(num_epochs)), self.bestm_vlh, color="orange", label="Best model validation loss history")
        plt.title("Model loss history")
        plt.legend()
        plt.savefig('./output/loss_history.png')
        plt.show()


    def get_results(self):
        self.__plot_loss_history()

 

 