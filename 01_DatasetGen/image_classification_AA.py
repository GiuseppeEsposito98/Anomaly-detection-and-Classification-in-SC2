import argparse
import datetime
import json
import os
import time
import random

# Global counter
img_id = 0
fault_id = 0

import torch
from torch.backends import cudnn
from torchdistill.common import yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process,set_seed
from torchdistill.datasets import util
from torchdistill.eval.classification import compute_accuracy
from torchdistill.misc.log import setup_log_file, MetricLogger
import torch.nn.functional as F

from sc2bench.analysis import check_if_analyzable
from sc2bench.models.registry import load_classification_model
from sc2bench.models.wrapper import get_wrapped_classification_model

from FI_Weights import DatasetSampling

from torch.utils.data import DataLoader, Subset

logger = def_logger.getChild(__name__)
#torch.multiprocessing.set_sharing_strategy('file_system')
import logging

def get_argparser():
    parser = argparse.ArgumentParser(description='Supervised compression for image classification tasks')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--log', help='log file path')
    parser.add_argument('--seed', type=int, help='seed in random number generator')
    parser.add_argument('--adv_attack', action='store_true', help='Yaml file path fsim config')
    return parser


def load_model(model_config, device, distributed):
    if 'classification_model' not in model_config:
        return load_classification_model(model_config, device, distributed)
    return get_wrapped_classification_model(model_config, device, distributed)



#@torch.inference_mode()
def evaluate(model_wo_ddp, data_loader, device,
             log_freq=1000, title=None, header='Test:'):
    model = model_wo_ddp.to(device)

    if title is not None:
        logger.info(title)

    # model.eval()

    analyzable = check_if_analyzable(model_wo_ddp)

    metric_logger = MetricLogger(delimiter='  ')
    save_dir = os.path.join("hook_encoder_output", "AdversarialAttack")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the output tensor
    
    epsilon = 0.0003
    for img_id, (image, target) in enumerate(metric_logger.log_every(data_loader, log_freq, header)):
        logger.info(image)
        
        if isinstance(image, torch.Tensor):
            image = image.to(device, non_blocking=True)

        adv_image = image.detach().clone().requires_grad_(True)
        n=0
        adv_image = image
        golden = None

        min_adv_iter = 15
        iter_to_save = []
        iter_to_save = random.sample(range(min_adv_iter, 300), 2)
        print(iter_to_save)
        for i in range(300):
            n+=1
            int_feature = model.bottleneck_layer.encoder(adv_image)
            
            aa_path = os.path.join(save_dir, f"{i}_{img_id}.pt")

            if i in iter_to_save:
                torch.save(int_feature, aa_path)
            aa_path_label = os.path.join(save_dir, f"L_{i}_{img_id}.pt")
            output = model(adv_image)
            if i in iter_to_save:
                torch.save(output, aa_path_label)
            if i == 0:
                golden = output
                
            mse=F.mse_loss(torch.zeros_like(int_feature),int_feature)

            model.zero_grad()
            mse.backward(retain_graph=True)
            grad_sign = adv_image.grad.sign()
            adv_image = adv_image - epsilon * grad_sign

            adv_image = torch.clamp(adv_image, 0, 1).detach().requires_grad_(True)
            if i > 0:
                acc1, acc5 = compute_accuracy(output, golden, topk=(1, 5))

        # FIXME need to take into account that the datasets could have been padded in distributed setup
        batch_size = len(image)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    if analyzable and model_wo_ddp.activated_analysis:
        model_wo_ddp.summarize()
    return metric_logger.acc1.global_avg



def main(args):
    global fault_id
    global img_id
    log_file_path = args.log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    logger.info(args)
    cudnn.enabled=True
    cudnn.benchmark = True
    cudnn.deterministic = True

    set_seed(args.seed)

    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))

    device = torch.device(args.device)
    dataset_dict = util.get_all_datasets(config['datasets'])
    models_config = config['models']
    
    student_model_config =\
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    
    student_model = load_model(student_model_config, device, False)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                              test_data_loader_config, False)
    
    log_freq = test_config.get('log_freq', 1000)
    test_batch_size=1
    test_shuffle=config['test']['test_data_loader']['random_sample']
    test_num_workers=config['test']['test_data_loader']['num_workers']

    #QUA DATASET SAMPLING PRENDE A CASO LE IMMAGINI DAL DATASET INIZIALE (50 per classe per 50 classi)
    num_of_classes = 50
    num_of_image_for_class = 50  
    subsampler = DatasetSampling(test_data_loader.dataset, num_of_image_for_class, num_of_classes)

    index_dataset=subsampler.listindex()
    data_subset=Subset(test_data_loader.dataset, index_dataset)
    dataloader = DataLoader(data_subset,batch_size=test_batch_size, shuffle=test_shuffle,pin_memory=True,num_workers=test_num_workers)

    if args.adv_attack:
        evaluate(student_model, dataloader, device,
                log_freq=log_freq, title='[Student: {}]'.format(student_model_config['name']), header='Golden')

if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())