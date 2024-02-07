import torch, logging, argparse, os, time, sys
sys.path.append(os.path.dirname(__file__))

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.nn.parallel import DataParallel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss, CosineSimilarity
from torch.utils.tensorboard.writer import SummaryWriter

from lib.data_io import ScepterViTDataset
from densePrediction.spatiotemporal_cmixer_dense_e1 import ScepterConvMixer
from densePrediction.spatiotemporal_vit_dense_e1 import ScepterVisionTransformer
from tools.utils import weights_init, fetch_list_of_backup_files
from omegaconf import OmegaConf

def criterion(x1: torch.Tensor, 
              x2: torch.Tensor, 
              loss_function: str = 'MSE', 
              **kwargs) -> torch.Tensor:
    """Computes loss value based on the defined task. We have CrossEntrop loss for Recognition, 
        while MSE, COS, and HYB are loss functions for Dense Prediction.

    Args:
        x1 (torch.Tensor): Output of the model.
        x2 (torch.Tensor): Target value.
        loss_function (str, optional): Type of criterion being applied. Defaults to 'MSE'.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        torch.Tensor: Loss value of given tensors.
    """ 
    if loss_function == 'CrossEntropy':
        weights_ = kwargs['sample_weight'] if 'sample_weight' in kwargs else None
        metric = CrossEntropyLoss(weight=weights_)
        if x1.shape[1] == 1:
            x1 = x1.squeeze()
            x2 = x2.float()
            metric = BCEWithLogitsLoss(weight=weights_) 
        return metric(x1, x2)
    elif loss_function == 'Correlation':
        cos = CosineSimilarity(dim=1, eps=1e-6)
        metric = 1. - cos(x1 - x1.mean(dim=1, keepdim=True), x2 - x2.mean(dim=1, keepdim=True))
        return metric.sum()
    elif loss_function == 'MSE':
        metric = MSELoss(reduction='sum')
        return metric(x1, x2)
    elif loss_function == 'COS':
        bs = x1.shape[0]
        metric = CosineSimilarity(dim=1, eps=1e-6)
        return (1. - metric(x1.contiguous().view(bs, -1), x2.contiguous().view(bs, -1))).sum()
    else:
        mse_ = MSELoss(reduction='mean')
        cos_ = CosineSimilarity(dim=-1, eps=1e-6)
        return mse_(x1, x2) + (1. - cos_(x1, x2)).sum()


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
    experiment_name = conf.EXPERIMENT.name
    model_architecture = conf.EXPERIMENT.architecture
    checkpoints_directory = os.path.join(args.save_dir, experiment_tag)
    mask_file_path = os.path.abspath(args.mask)
    dataset_file = os.path.abspath(args.dataset)
    log_directory = os.path.join(args.log_dir, experiment_name)
    logging.info("Loading subjects fMRI files and component maps")    
    main_dataset = ScepterViTDataset(image_list_file=dataset_file,
                                     mask_file=mask_file_path,
                                     **conf.DATASET)
    if not os.path.exists(log_directory):
        os.mkdir(log_directory)
        logging.info(f'Log directory created in *logdir/{experiment_name}')
    if save_flag and not os.path.exists(checkpoints_directory):
        os.mkdir(checkpoints_directory)
        logging.info(f'Save directory created in *savedir/{experiment_tag}')    
    if main_dataset.class_dict:
        logging.info(f'Class name = {main_dataset.class_dict}')
    if main_dataset.imbalanced_weights is not None:
        logging.info(f'Class weights = {main_dataset.imbalanced_weights}')
        main_dataset.imbalanced_weights = main_dataset.imbalanced_weights.to(dev, non_blocking=True)
    if save_flag:
        backup_files = fetch_list_of_backup_files(model_architecture)
        os.system(f"cp -f {os.path.join(os.path.dirname(os.path.abspath(__file__)),'config', backup_files[0])} {os.path.join(os.path.dirname(os.path.abspath(__file__)),'densePrediction', backup_files[1])} {checkpoints_directory}")
    data_pack = {}
    data_pack['train'], data_pack['val'] = random_split(main_dataset, [.8, .2], generator=torch.Generator().manual_seed(70))
    dataloaders = {x: DataLoader(data_pack[x], batch_size=int(conf.TRAIN.batch_size), shuffle=True, num_workers=int(conf.TRAIN.workers), pin_memory=True) for x in ['train', 'val']}       
    gpu_ids = list(range(torch.cuda.device_count()))
    writer = SummaryWriter(log_dir=log_directory, comment=conf.EXPERIMENT.name)
    if model_architecture == 'ViT':
        base_model = ScepterVisionTransformer(**conf.MODEL)
    else:
        base_model = ScepterConvMixer(**conf.MODEL)
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
                    optimizer.zero_grad(set_to_none=True)
                    preds = base_model(inp)
                    loss = criterion(preds, label, conf.TRAIN.loss)
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
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    main()