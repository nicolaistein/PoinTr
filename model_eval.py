from utils.config import *
from models.AdaPoinTr import AdaPoinTr 
import torch


config_path = 'cfgs/PCN_models/AdaPoinTrPart.yaml'
checkpoint_path = 'experiments/AdaPoinTrPart/PCN_models/first_train/ckpt-best.pth'

config = cfg_from_yaml_file(config_path)
print(config)
# Initialize the model
model = AdaPoinTr(config)
model.eval()
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

