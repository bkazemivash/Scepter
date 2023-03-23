import torch, logging, argparse, os, time, sys
sys.path.append(os.path.dirname(__file__))

from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from lib.data_io import ScepterViTDataset
from lib.summary_metrics import *
from tools.visualization import *
from recognition.spatiotemporal_vit import VisionTransformer

def criterion(x1: torch.Tensor, x2: torch.Tensor, 
              evalution='Confusion_Matrix', **kwargs) -> torch.Tensor:
    """Computes evaluation metrics on input tensors for the defined task.

    Args:
        x1 (torch.Tensor): Output of the model.
        x2 (torch.Tensor): Target value.
        evalution (str, optional): Type of evaluation. Defaults to 'Confusion_Matrix'.
        **kwargs: Arbitrary keyword arguments mandatory for evaluation.

    Raises:
        ValueError: If evaluation method is not defined.

    Returns:
        torch.Tensor: Evaluation of given tensors.
    """
    if evalution == 'Accuracy':
       return compute_accuracy(x1, x2)
    elif evalution == 'Correlation':
        return compute_correlation(x1, x2)
    elif evalution == 'Confusion_Matrix':
        return compute_confusion_matrix(x1, x2, n_cls=kwargs['nb_classes'])
    else:
        raise ValueError('Evalution metric is not defined.')


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
        raise FileNotFoundError(f"pre-train model: file not found: {args.pretrained_model}")
    if not (os.path.exists(args.save_dir)):
        raise FileNotFoundError(f"Save directory does not exist, {args.save_dir}")
    
    logging.info("Loading configuration data ...")
    torch.manual_seed(712)
    conf = OmegaConf.load(args.config)
    checkpoint = torch.load(args.pretrained_model)
    mask_file_path = os.path.abspath(args.mask)
    dataset_file = os.path.abspath(args.testset)
    saving_path = os.path.abspath(args.save_dir)
    metric_ = conf.TEST.metric
    logging.info("Loading test dataset for evaluating pretrained model.")
    main_dataset = ScepterViTDataset(image_list_file=dataset_file,
                                     mask_file=mask_file_path,
                                     **conf.DATASET)
    dataloader = DataLoader(main_dataset, batch_size=conf.TEST.test_size, shuffle=conf.TEST.shuffling, num_workers=conf.TEST.test_workers, pin_memory=True)
    logging.info("Loading model and optimizer parameters.")
    pretrained_mdl = VisionTransformer(n_timepoints=main_dataset.time_bound, **conf.MODEL)
    if torch.cuda.is_available():
        pretrained_mdl = pretrained_mdl.cuda()
        pretrained_mdl = nn.DataParallel(pretrained_mdl)
        logging.info("DataParallel mode activated.") 
    pretrained_mdl.load_state_dict(checkpoint['state_dict'], strict=True)
    params_ = {}
    if main_dataset.class_dict:
        logging.info(f'Class name : {main_dataset.class_dict} and class weights {main_dataset.imbalanced_weights}')
        nb_classes = len(main_dataset.class_dict)
        running_metric = torch.zeros(nb_classes, nb_classes)
        params_ = {'nb_classes': nb_classes}
    else:
        running_metric = 0.0
    logging.info('Changing model status to evaluation.')
    pretrained_mdl.eval()

    for inp, label in dataloader:
        inp = inp.to(dev, non_blocking=True)
        label = label.to(dev, non_blocking=True)
        with torch.set_grad_enabled(False):
            preds = pretrained_mdl(inp)
            running_metric += criterion(preds, label, metric_, **params_)

    logging.info(f'Evaluation is done on metric {metric_}')
    print(running_metric)
    print(f'test size is {main_dataset.info_dataframe.shape[0]}')
    print(running_metric.diag()/running_metric.sum(1))  # type: ignore
    logging.info(f'Plotting and saving summary metrics: {saving_path}')
    if main_dataset.class_dict and isinstance(running_metric, torch.Tensor):
        draw_confusion_matrix(running_metric, main_dataset.class_dict, saving_path)
    logging.info('Inference is finished!')

if __name__ == "__main__":
    main()