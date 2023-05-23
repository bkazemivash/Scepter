import torch, logging, argparse, os, time, sys
sys.path.append(os.path.dirname(__file__))

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.nn.parallel import DataParallel
from torch.nn import CrossEntropyLoss, CosineSimilarity, BCEWithLogitsLoss, MSELoss
from torch.utils.tensorboard.writer import SummaryWriter

from lib.data_io import ScepterViTDataset
from densePrediction.spatiotemporal_vit_dense_e1 import ScepterVisionTransformer
from tools.utils import weights_init
from omegaconf import OmegaConf

def criterion(x1: torch.Tensor, 
              x2: torch.Tensor, 
              task='Recognition', 
              sample_weight=None) -> torch.Tensor:
    """Computes loss value based on the defined task.

    Args:
        x1 (torch.Tensor): Output of the model.
        x2 (torch.Tensor): Target value.
        task (str, optional): Type of experiment ['Recognition', 'DensePrediction']. Defaults to 'Recognition'.
        sample_weight (torch.Tensor, optional): Class weights in case of imbalanced dataset. Defaults to None.

    Returns:
        torch.Tensor: Loss value of given tensors.
    """ 
    if task == 'Recognition':
        entropy = CrossEntropyLoss(weight=sample_weight)
        if x1.shape[1] == 1:
            x1 = x1.squeeze()
            x2 = x2.float()
            entropy = BCEWithLogitsLoss(weight=sample_weight) 
        return entropy(x1, x2)
    else:
        # x1, x2 = x1.flatten(1), x2.flatten(1)
        # cos = CosineSimilarity(dim=1, eps=1e-6)
        # pearson = 1. - cos(x1 - x1.mean(dim=1, keepdim=True), x2 - x2.mean(dim=1, keepdim=True))
        # return pearson.sum()
        distance_ = MSELoss()
        return distance_(x1, x2)


def main():
    parser = argparse.ArgumentParser(description='Training ViT recognition model')
    parser.add_argument('-c', '--config', required=True, help='Path to the config file') 
    parser.add_argument('-m', '--mask', required=True, help='Path to the mask file')  
    parser.add_argument('-t', '--dataset', required=True, help='Path to pandas dataframe that keeps list of images')    
    parser.add_argument('-s', '--save_dir', required=True, help='Path to save checkpoint')  
    parser.add_argument('-l', '--log_dir', required=True, help='Path to save logfile')  
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
    if not (os.path.exists(args.log_dir)):
        raise FileNotFoundError(f"Log directory does not exist, {args.log_dir}")  
    
    logging.info("Loading configuration data ...")
    conf = OmegaConf.load(args.config)
    save_flag = conf.EXPERIMENT.save_model
    experiment_tag = conf.EXPERIMENT.tag
    checkpoints_directory = os.path.abspath(args.save_dir)
    mask_file_path = os.path.abspath(args.mask)
    dataset_file = os.path.abspath(args.dataset)
    log_directory = os.path.abspath(args.log_dir)
    logging.info("Loading subjects fMRI files and component maps")    
    main_dataset = ScepterViTDataset(image_list_file=dataset_file,
                                     mask_file=mask_file_path,
                                     **conf.DATASET)
    if main_dataset.class_dict:
        logging.info(f'Class name = {main_dataset.class_dict}')
    if main_dataset.imbalanced_weights is not None:
        logging.info(f'Class weights = {main_dataset.imbalanced_weights}')
        main_dataset.imbalanced_weights = main_dataset.imbalanced_weights.to(dev, non_blocking=True)
    data_pack = {}
    data_pack['train'], data_pack['val'] = random_split(main_dataset, [.8, .2], generator=torch.Generator().manual_seed(70))
    dataloaders = {x: DataLoader(data_pack[x], batch_size=int(conf.TRAIN.batch_size), shuffle=True, num_workers=int(conf.TRAIN.workers), pin_memory=True) for x in ['train', 'val']}       
    gpu_ids = list(range(torch.cuda.device_count()))
    writer = SummaryWriter(log_dir=log_directory, comment=conf.EXPERIMENT.name)
    base_model = ScepterVisionTransformer(n_timepoints=main_dataset.time_bound, **conf.MODEL)
    base_model.apply(weights_init)
    if torch.cuda.device_count() > 1:
        base_model = DataParallel(base_model, device_ids = gpu_ids)
        logging.info(f"Pytorch Distributed Data Parallel activated using gpus: {gpu_ids}")
    if torch.cuda.is_available():
        base_model = base_model.cuda()
    optimizer = torch.optim.Adam(base_model.parameters(), lr=float(conf.TRAIN.base_lr))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(conf.TRAIN.step_lr), gamma=float(conf.TRAIN.weight_decay))
    best_loss = float('inf')
    logging.info(f"Optimizer: Adam , Criterion: {conf.TRAIN.loss} , lr: {conf.TRAIN.base_lr} , decay: {conf.TRAIN.weight_decay}")
    num_epochs = int(conf.TRAIN.epochs)
    phase_error = {'train': 0., 'val': 0.}    
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                base_model.train() 
            else:
                base_model.eval() 
            running_loss = 0.0
            for inp, label in dataloaders[phase]:
                inp = inp.to(dev, non_blocking=True)
                label = label.to(dev, non_blocking=True)
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()
                    preds = base_model(inp)
                    loss = criterion(preds, label, experiment_tag, main_dataset.imbalanced_weights)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()                     
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / len(data_pack[phase])
            phase_error[phase] = epoch_loss
        logging.info("Epoch {}/{} - Train Loss: {:.10f} and Validation Loss: {:.10f}".format(epoch+1, num_epochs, phase_error['train'], phase_error['val']))
        writer.add_scalars("Loss", {'train': phase_error['train'], 'validation': phase_error['val']}, epoch)
        if phase == 'val' and save_flag and epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({'epoch': epoch,
                        'state_dict': base_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': phase_error,
                        'edition': conf.EXPERIMENT.edition}, 
                        os.path.join(checkpoints_directory, 'checkpoint_{}_{}.pth'.format(epoch, time.strftime("%m%d%y_%H%M%S"))))
    
    writer.flush()
    writer.close()
    logging.info("Training procedure is done!")


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    main()