import argparse
import os
import random
# Global counter
img_id = 0
fault_id = 0

import torch
import torchvision
from torch.backends import cudnn
from torchdistill.common import yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.datasets import util
from torchdistill.eval.classification import compute_accuracy
from torchdistill.misc.log import MetricLogger

from sc2bench.analysis import check_if_analyzable
from sc2bench.models.registry import load_classification_model
from sc2bench.models.wrapper import get_wrapped_classification_model

from pytorchfi.FI_Weights import FI_manager 
from FI_Weights import DatasetSampling 

from torch.utils.data import DataLoader, Subset

logger = def_logger.getChild(__name__)

import logging

def get_argparser():
    parser = argparse.ArgumentParser(description='Supervised compression for image classification tasks')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--fsim_config', help='Yaml file path fsim config')
    return parser


def load_model(model_config, device, distributed):
    if 'classification_model' not in model_config:
        return load_classification_model(model_config, device, distributed)
    return get_wrapped_classification_model(model_config, device, distributed)


@torch.inference_mode()
def evaluate(model_wo_ddp, data_loader, device,
             log_freq=1000, title=None, header='Test:', fsim_enabled=False, Fsim_setup:FI_manager = None):
    model = model_wo_ddp.to(device)

    if title is not None:
        logger.info(title)

    model.eval()

    analyzable = check_if_analyzable(model_wo_ddp)
    metric_logger = MetricLogger(delimiter='  ')
    im=0

    num_of_inj_x_fault = 3  #configurable parameter
    list_index_to_inj = random.sample(range(0, len(data_loader)),  num_of_inj_x_fault)

    logger.info("Indici che mi aspetto vada a stampare:")
    logger.info(list_index_to_inj)

    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        # logger.info(image)
        if header == "Golden" or im in list_index_to_inj:
            logger.info("IMG = " + str(im))
            if isinstance(image, torch.Tensor):
                image = image.to(device, non_blocking=True)

            if isinstance(target, torch.Tensor):
                target = target.to(device, non_blocking=True)

            if fsim_enabled==True:
                output = model(image)
                Fsim_setup.FI_report.update_report(im,output,target,topk=(1,5))
            else:
                output = model(image)
            acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
            batch_size = len(image)
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

            metric_logger.synchronize_between_processes()
            top1_accuracy = metric_logger.acc1.global_avg
            top5_accuracy = metric_logger.acc5.global_avg
            # logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
            if analyzable and model_wo_ddp.activated_analysis:
                model_wo_ddp.summarize()
        
        else: pass

        im+=1    

def save_encoder_golden_output(module, input, output):
    global img_id
    cwd=os.getcwd() 
    save_dir = os.path.join(f"{cwd}/hook_encoder_output", "golden")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the output tensor
    golden_path = os.path.join(save_dir, f"{img_id}.pt")
    torch.save(output, golden_path)
    
    img_id += 1  # Increment counter

def save_encoder_faulty_output(module, input, output):
    global img_id
    global fault_id
    cwd=os.getcwd() 
    golden_path = os.path.join(f"{cwd}/hook_encoder_output", "golden", f"{img_id}.pt")
    
    # Ensure faulty directory exists
    faulty_dir = os.path.join(f"{cwd}/hook_encoder_output", "faulty")
    os.makedirs(faulty_dir, exist_ok=True)
    faulty_path = os.path.join(faulty_dir, f"{fault_id}_{img_id}.pt")

    # Check if the golden output exists
    if os.path.exists(golden_path):
        golden_output = torch.load(golden_path)  # Load golden output

        # Compare tensors
        if not torch.equal(output, golden_output):
            torch.save(output, faulty_path)
    else:
        logger.warning(f"Golden output for img_id {img_id} not found! Saving faulty output anyway.")
        torch.save(output, faulty_path)  # Save if no golden reference exists
    
    img_id += 1  # Increment counter


def main(args):
    global fault_id
    global img_id

    cudnn.enabled=True
    # cudnn.benchmark = True
    cudnn.deterministic = True
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))

    device = torch.device(args.device)
    dataset_dict = util.get_all_datasets(config['datasets'])
    models_config = config['models']
    
    student_model_config =\
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    
    student_model = load_model(student_model_config, device)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                              test_data_loader_config, distributed = False)
    log_freq = test_config.get('log_freq', 1000)
        

    test_batch_size=1
    test_shuffle=config['test']['test_data_loader']['random_sample']
    test_num_workers=config['test']['test_data_loader']['num_workers']
    
    
    num_of_classes = 50
    num_of_image_for_class = 50  
    subsampler = DatasetSampling(test_data_loader.dataset, num_of_image_for_class, num_of_classes)
    
    index_dataset=subsampler.listindex()
    data_subset=Subset(test_data_loader.dataset, index_dataset)
    dataloader = DataLoader(data_subset,batch_size=test_batch_size, shuffle=test_shuffle,pin_memory=True,num_workers=test_num_workers)

    # if args.fsim_config:
    fsim_config_descriptor = yaml_util.load_yaml_file(os.path.expanduser(args.fsim_config))
    conf_fault_dict=fsim_config_descriptor['fault_info']['neurons']
    cwd=os.getcwd() 
    full_log_path=cwd

    # 1. create the fault injection setup
    FI_setup=FI_manager(full_log_path,chpt_file_name='ckpt_FI.json',fault_report_name='fsim_report.csv')

    # 2. Run a fault free scenario to generate the golden model
    hook_golden_handle = student_model.bottleneck_layer.encoder.register_forward_hook(save_encoder_golden_output)
    FI_setup.open_golden_results("Golden_results")
    evaluate(student_model, dataloader, device,
            log_freq=log_freq, title='[Student: {}]'.format(student_model_config['name']), header='Golden', fsim_enabled=True, Fsim_setup=FI_setup) 
    FI_setup.close_golden_results()
    hook_golden_handle.remove()
    img_id = 0

    # 3. Prepare the Model for fault injections
    FI_setup.FI_framework.create_fault_injection_model(device,student_model,
                                        batch_size=test_batch_size,
                                        input_shape=[3,224,224],
                                        layer_types=[torch.nn.Conv2d,torch.nn.Linear],Neurons=True)
    
    # 4. generate the fault list
    logging.getLogger('pytorchfi').disabled = True
    
    FI_setup.generate_fault_list(flist_mode='neurons',
                                f_list_file='fault_list.csv',
                                layers=conf_fault_dict['layers'],
                                trials=conf_fault_dict['trials'], 
                                size_tail_y=conf_fault_dict['size_tail_y'], 
                                size_tail_x=conf_fault_dict['size_tail_x'],
                                block_fault_rate_delta=conf_fault_dict['block_fault_rate_delta'],
                                block_fault_rate_steps=conf_fault_dict['block_fault_rate_steps'],
                                neuron_fault_rate_delta=conf_fault_dict['neuron_fault_rate_delta'],
                                neuron_fault_rate_steps=conf_fault_dict['neuron_fault_rate_steps'])     
    
    FI_setup.load_check_point()


    # 5. Execute the fault injection campaign
    for fault,k in FI_setup.iter_fault_list():
        fault_id = k
        img_id = 0
        FI_setup.FI_framework.bit_flip_err_neuron(fault)
        FI_setup.open_faulty_results(f"F_{k}_results")

        hook_faulty_handle = FI_setup.FI_framework.faulty_model.bottleneck_layer.encoder.register_forward_hook(save_encoder_faulty_output)
        try:
            evaluate(FI_setup.FI_framework.faulty_model, dataloader, device,
                log_freq=log_freq, title='[Student: {}]'.format(student_model_config['name']), header='FSIM', fsim_enabled=True,Fsim_setup=FI_setup)        
        
        except OSError as Oserr:
            msg=f"Oserror: {Oserr}"
            logger.info(msg)

        except Exception as Error:
            msg=f"Exception error: {Error}"
            logger.info(msg)
        hook_faulty_handle.remove()
        FI_setup.parse_results()
        # break
    FI_setup.terminate_fsim()


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
