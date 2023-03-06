import torch, logging, argparse, os, time, sys
sys.path.append(os.path.dirname(__file__))

import torch
from torch.nn import CosineSimilarity
from omegaconf import OmegaConf
from lib.data_io import ScepterViTDataset
from recognition.spatiotemporal_vit import VisionTransformer

def criterion(x1: torch.Tensor, x2: torch.Tensor, task='Recognition') -> torch.Tensor:
    """Computes evaluation metrics on input tensors for the defined task.

    Args:
        x1 (torch.Tensor): Output of the model.
        x2 (torch.Tensor): Target value.
        task (str, optional): Type of experiment ['Recognition', 'DensePrediction']. Defaults to 'Recognition'.

    Returns:
        torch.Tensor: Evaluation of given tensors.
    """ 
    if task == 'Recognition':
        accuracy_ = CosineSimilarity(dim=1, eps=1e-6)
        return accuracy_(x1, x2)
    else:
        cos = CosineSimilarity(dim=1, eps=1e-6)
        pearson = cos(x1 - x1.mean(dim=1, keepdim=True), x2 - x2.mean(dim=1, keepdim=True))
        return 1. - pearson


def main():
    parser = argparse.ArgumentParser(description='Training ViT recognition model')
    parser.add_argument('-c', '--config', required=True, help='Path to the config file') 
    parser.add_argument('-m', '--mask', required=True, help='Path to the mask file')  
    parser.add_argument('-t', '--testset', required=True, help='Path to pandas dataframe that keeps list of images')
    parser.add_argument('-p', '--pretrained_model', required=True, help='Path to the pretrained model')
    parser.add_argument('-s', '--save_dir', required=True, help='Path to save evaluation metrics') 
    args = parser.parse_args()

    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET, format="[ %(asctime)s ]  %(levelname)s : %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    for i in range(torch.cuda.device_count()):
        logging.debug("Available processing unit ({} : {})".format(i, torch.cuda.get_device_name(i)))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not (os.path.exists(args.mask)):
        raise FileNotFoundError(f"Mask file not found: {args.mask}")       
    if not (os.path.exists(args.config)):
        raise FileNotFoundError(f"Config file not found: {args.config}")  
    if not (os.path.exists(args.testset)):
        raise FileNotFoundError(f"DataTable: file not found: {args.testset}") 
    if not (os.path.exists(args.pretrained_model)):
        raise FileNotFoundError(f"Save directory does not exist, {args.pretrained_model}")
    if not (os.path.exists(args.save_dir)):
        raise FileNotFoundError(f"Save directory does not exist, {args.save_dir}")
    
    logging.info("Loading configuration data ...")
    conf = OmegaConf.load(args.config)
    checkpoint = torch.load(args.pretrained_model)
    mask_file_path = os.path.abspath(args.mask)
    dataset_file = os.path.abspath(args.dataset)
    logging.info("Loading test dataset for evaluating pretrained model.")
    main_dataset = ScepterViTDataset(image_list_file=dataset_file,
                                     mask_file=mask_file_path,
                                     **conf.DATASET)
    logging.info("Loading model and optimizer parameters.")
    pretrained_mdl = VisionTransformer(n_timepoints=main_dataset.time_bound, **conf.MODEL)
    if torch.cuda.is_available():
        pretrained_mdl = pretrained_mdl.cuda()    
    optimizer = torch.optim.Adam(pretrained_mdl.parameters())
    pretrained_mdl.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])





if __name__ == "__main__":
    main()