import os, sys, argparse, logging, torch
sys.path.append(os.path.dirname(__file__))

from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from lib.data_io import ScepterViTDataset
from densePrediction.spatiotemporal_cmixer_dense_e1 import ScepterConvMixer
from densePrediction.spatiotemporal_vit_dense_e1 import ScepterVisionTransformer


def main():
    parser = argparse.ArgumentParser(prog='Scepter inference module.',
                                     description='Dense prediction of dynamic maps.',
                                     epilog='Check ReadMe file for more information.',)
    parser.add_argument('-c', '--config', required=True, help='Path to the config file.')
    parser.add_argument('-m', '--mask', required=True, help='Path to the config file.')
    parser.add_argument('-d', '--dataset', required=True, help='Path to the config file.')
    parser.add_argument('-s', '--save_dir', required=True, help='Path to the config file.')
    parser.add_argument('-p', '--checkpoint', required=True, help='Path to the config file.')
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
    if not (os.path.exists(args.dataset)):
        raise FileNotFoundError(f"DataTable: file not found: {args.dataset}") 
    if not (os.path.exists(args.save_dir)):
        raise FileNotFoundError(f"Save directory does not exist, {args.save_dir}")  
    if not (os.path.exists(args.checkpoint)):
        raise FileNotFoundError(f"Checkpoint directory does not exist, {args.checkpoint}")
    
    logging.info("Loading configuration data ...")
    conf = OmegaConf.load(args.config)
    experiment_tag = conf.EXPERIMENT.tag
    checkpoint_path = os.path.abspath(args.checkpoint)
    model_architecture = conf.EXPERIMENT.architecture
    data_storage = os.path.join(args.save_dir, experiment_tag)
    mask_file_path = os.path.abspath(args.mask)
    dataset_file = os.path.abspath(args.dataset)
    logging.info("Loading subjects fMRI files and component maps")    
    main_dataset = ScepterViTDataset(image_list_file=dataset_file,
                                     mask_file=mask_file_path,
                                     **conf.DATASET)
    dataloaders = DataLoader(main_dataset, batch_size=int(conf.TEST.test_size), shuffle=conf.TEST.shuffling,)
    logging.info(f"Test size: {conf.TEST.test_size} and shuffling strategy: {conf.TEST.shuffling}") 
    if not os.path.exists(data_storage):
        os.mkdir(data_storage)
        logging.info(f'Save directory created in *savedir/{experiment_tag}')    

    if model_architecture == 'ViT':
        base_model = ScepterVisionTransformer(**conf.MODEL) 
    else:
        base_model = ScepterConvMixer(**conf.MODEL)
    checkpoint = torch.load(checkpoint_path)
    base_model.load_state_dict(checkpoint['state_dict'], strict=False)
    base_model.eval()
    pbar = tqdm(dataloaders)  
    if torch.cuda.is_available():
        base_model = base_model.cuda()
    for i, sample in enumerate(pbar):  
        inp = sample['img'].to(dev, non_blocking=True)
        with torch.set_grad_enabled(False):
            preds = base_model(inp)
        storage_path = os.path.join(data_storage, '{}.pt'.format(sample['SubjectID'][0].strip()))
        preds = preds.squeeze()
        torch.save(preds, storage_path)
    logging.info("Data generation is done.") 

if __name__ == "__main__":
    main()