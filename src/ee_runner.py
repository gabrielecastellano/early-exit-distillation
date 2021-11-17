import argparse
import datetime
import json
import math
import time

import numpy as np
import os

import torch
from pathlib import Path

from torch import distributed as dist
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

import early_classifier.base
from early_classifier.ee_dataset import EmbeddingDataset
from early_classifier.ee_utils import iterate_configurations
from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util, module_util
from structure.logger import MetricLogger, SmoothedValue, CtrValue
from utils import main_util, mimic_util, dataset_util

from early_classifier.base import BaseClassifier
from early_classifier import ee_utils


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Learner')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('-bn_train', action='store_true', help='train a bottlenecked model by distilling a teacher')
    argparser.add_argument('-org_eval', action='store_true', help='evaluate the original model')
    argparser.add_argument('-bn_eval', action='store_true', help='evaluate the bottlenecked model')
    argparser.add_argument('-ee_joint_train', action='store_true', help='train an early exit model jointly')
    argparser.add_argument('-ee_solo_train', action='store_true', help='train an early exit model independently')
    # distributed training parameters
    argparser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    argparser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return argparser


def save_ckpt(student_model, epoch, best_valid_value, ckpt_file_path, teacher_model_type):
    print('Saving..')
    module =\
        student_model.module if isinstance(student_model, (DataParallel, DistributedDataParallel)) else student_model
    state = {
        'type': teacher_model_type,
        'model': module.state_dict(),
        'epoch': epoch + 1,
        'best_valid_value': best_valid_value,
        'student': True
    }
    file_util.make_parent_dirs(ckpt_file_path)
    torch.save(state, ckpt_file_path)


@torch.no_grad()
def evaluate(model, data_loader, device, ee_model=None, ee_threshold=0, interval=1000, split_name='Test', title=None):
    """
    Run the model on a test set and compute the top-1 and top-5 accuracy.
    The model is run jointly with an early exit model.
    Args:
        model:
        data_loader:
        device:
        ee_model (BaseClassifier):
        ee_threshold:
        interval:
        split_name:
        title:

    Returns:

    """
    if title is not None:
        print(title)

    model = model.to(model.device)
    if ee_model is not None:
        ee_model = ee_model.to(ee_model.device)

    overall_samples = 50000
    overall_samples_per_class = 500
    used_samples_per_class = 500
    samples = int(overall_samples / overall_samples_per_class) * used_samples_per_class

    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    if ee_model:
        ee_model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_counter('early_predictions', CtrValue())
    header = '{}:'.format(split_name)
    with torch.no_grad():
        img_ctr_wide = -1
        img_ctr = 0
        early_exit_ctr = 0
        for image, target in metric_logger.log_every(data_loader, interval, header):

            img_ctr_wide += data_loader.batch_size
            if img_ctr_wide % overall_samples_per_class >= used_samples_per_class:
                continue
            if img_ctr >= samples:
                    break

            image = image.to(model.device, non_blocking=True)
            target = target.to(model.device, non_blocking=True)

            if not ee_model:
                output = model(image)
                new_early_exits = 0
            else:
                # run model up to bottleneck
                bn_output = model.forward_to_bn(image)
                embeddings = bn_output.to(ee_model.device)
                embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1:].numel())

                # early prediction
                ee_output = ee_model.predict(embeddings)
                ee_conf = ee_model.get_prediction_confidences(ee_output)

                # forward not-confident vectors to the full model
                # c, p = torch.max(torch.nn.functional.softmax(ee_output[i], dim=0), 0)
                # full_predictions = bn_output[ee_output < ee_threshold].gpu()
                full_predictions = bn_output[ee_conf < ee_threshold]
                new_early_exits = data_loader.batch_size - full_predictions.shape[0]
                output = ee_output.to(model.device)
                # early_exit_ctr += new_early_exits

                if full_predictions.shape[0] > 0:
                    full_output = model.forward_from_bn(full_predictions)
                    # merge early and full predictions
                    j = 0
                    for i in range(len(ee_output)):
                        if ee_conf[i] < ee_threshold:
                            output[i] = full_output[j]
                            j += 1

            acc1, acc5 = main_util.compute_accuracy(output, target, topk=(1, 5))
            metric_logger.meters['acc1'].update(acc1.item(), n=data_loader.batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=data_loader.batch_size)
            metric_logger.counters['early_predictions'].update(new_early_exits, data_loader.batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    early_predictions = metric_logger.early_predictions.global_avg
    print(' * Acc@1 {:.4f}\tAcc@5 {:.4f}'.format(top1_accuracy, top5_accuracy))
    print(' * Fraction of early predictions {:.4f}'.format(early_predictions))
    torch.set_num_threads(num_threads)
    return metric_logger.acc1.global_avg


def validate(student_model_without_ddp, data_loader, config, device, distributed, device_ids, ee_model, ee_threshold):
    """
    Evaluate on the validation test after one distillation epoch.
    Args:
        student_model_without_ddp:
        data_loader:
        config:
        device:
        distributed:
        device_ids:
        ee_model (BaseClassifier):
        ee_threshold:

    Returns:

    """
    teacher_model_config = config['teacher_model']
    org_model, teacher_model_type = mimic_util.get_org_model(teacher_model_config, device)
    mimic_model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config,
                                             device, head_model=student_model_without_ddp)
    mimic_model_without_dp = mimic_model.module if isinstance(mimic_model, DataParallel) else mimic_model
    if distributed:
        mimic_model = DistributedDataParallel(mimic_model_without_dp, device_ids=device_ids)
    return evaluate(mimic_model, data_loader, device, ee_model=ee_model, ee_threshold=ee_threshold, split_name='Validation')


def distill_one_epoch(student_model, teacher_model, teacher_input_size, student_input_size, train_loader, optimizer,
                      criterion, epoch, device, interval, ee_model=None, ee_threshold=0):
    student_model.train()
    teacher_model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    # TODO check if this preserves the accuracy of the models
    student_upsampler = torch.nn.Upsample(student_input_size).to(device)
    teacher_upsampler = torch.nn.Upsample(teacher_input_size).to(device)
    for sample_batch, targets in metric_logger.log_every(train_loader, interval, header):
        start_time = time.time()
        sample_batch, targets = sample_batch.to(device), targets.to(device)
        optimizer.zero_grad()
        teacher_outputs = teacher_model(teacher_upsampler(sample_batch))
        if ee_model:
            embedding = student_model.forward_to_bn(student_upsampler(sample_batch))
            student_outputs = student_model.forward_from_bn(embedding)
        else:
            student_outputs = student_model(student_upsampler(sample_batch))
        # TODO use the embedding for joint training (loss should be affected by early classification)
        loss = criterion(student_outputs, teacher_outputs)

        loss.backward()
        optimizer.step()
        batch_size = sample_batch.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


def distill(train_loader, valid_loader, student_input_shape, teacher_input_shape, config, device, distributed,
            device_ids, ee_model=None, ee_threshold='auto'):
    """
    Train the (head of a) student model by knowledge distillation from the teacher model.
    The student is stored in ckpt.
    Args:
        train_loader:
        valid_loader:
        student_input_shape:
        teacher_input_shape:
        config:
        device:
        distributed:
        device_ids:
        ee_model (BaseClassifier):
        ee_threshold:

    Returns:

    """
    teacher_model_config = config['teacher_model']
    teacher_model, teacher_model_type = mimic_util.get_teacher_model(teacher_model_config, student_input_shape, device)
    module_util.freeze_module_params(teacher_model)
    student_model_config = config['student_model']
    student_model = mimic_util.get_student_model(teacher_model_type, student_model_config, config['dataset']['name'])
    student_model = student_model.to(device)
    start_epoch, best_valid_acc = mimic_util.resume_from_ckpt(student_model_config['ckpt'], student_model, device,
                                                              is_student=True)
    if best_valid_acc is None:
        best_valid_acc = 0.0

    train_config = config['train']
    criterion_config = train_config['criterion']
    criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
    optim_config = train_config['optimizer']
    optimizer = func_util.get_optimizer(student_model, optim_config['type'], optim_config['params'])
    scheduler_config = train_config['scheduler']
    scheduler = func_util.get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
    interval = train_config['interval']
    if interval <= 0:
        num_batches = len(train_loader)
        interval = num_batches // 20 if num_batches >= 20 else 1

    student_model_without_ddp = student_model
    if distributed:
        teacher_model = DataParallel(teacher_model, device_ids=device_ids)
        student_model = DistributedDataParallel(student_model, device_ids=device_ids)
        student_model_without_ddp = student_model.module

    ckpt_file_path = student_model_config['ckpt']
    end_epoch = start_epoch + train_config['epoch']
    start_time = time.time()
    for epoch in range(start_epoch, end_epoch):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        distill_one_epoch(student_model, teacher_model, student_input_shape[-1], teacher_input_shape[-1], train_loader, optimizer, criterion,
                          epoch, device, interval)
        ee_threshold = ee_model.get_threshold() if ee_threshold == 'auto' else ee_threshold
        valid_acc = validate(student_model, valid_loader, config, device, distributed, device_ids, ee_model, ee_threshold)
        if valid_acc > best_valid_acc and main_util.is_main_process():
            print('Updating ckpt (Best top1 accuracy: {:.4f} -> {:.4f})'.format(best_valid_acc, valid_acc))
            best_valid_acc = valid_acc
            save_ckpt(student_model_without_ddp, epoch, best_valid_acc, ckpt_file_path, teacher_model_type)
        scheduler.step()

    dist.barrier()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    del teacher_model
    del student_model


def get_embeddings(dataset, model, input_shape, device, fraction_of_samples=1.0, interval=10, split_name='Get Embeddings',
                   load_from_storage=False, store=False, embedding_storage=None):
    """

    Args:
        dataset:
        model (models.mimic.base.BaseMimic):
        input_shape:
        device:
        fraction_of_samples:
        interval:
        split_name:
        load_from_storage:
        store:
        embedding_storage:

    Returns:

    """
    overall_samples = len(dataset)
    overall_classes = max(dataset.targets) + 1
    overall_samples_per_class = overall_samples / overall_classes
    used_samples_per_class = int(fraction_of_samples * overall_samples_per_class)
    used_samples = overall_classes * used_samples_per_class
    bn_shape = model.head.bn_shape(input_shape, device)

    cache_labels = np.zeros([used_samples], dtype=int)
    cache_labels_t = np.zeros([used_samples], dtype=int)
    cache_confidences = np.zeros([used_samples])
    cache_data = np.zeros([used_samples, np.prod(bn_shape)], dtype=np.float32)

    metric_logger = MetricLogger(delimiter='  ')
    header = '{}:'.format(split_name)

    if load_from_storage:
        print(f"Loading previous embeddings tensors from disk...")
        try:
            cache_labels = torch.load(f'{embedding_storage}/labels.sav')[[i for i in range(overall_samples) if i % overall_samples_per_class < used_samples_per_class]]
            cache_labels_t = torch.load(f'{embedding_storage}/labels_t.sav')[[i for i in range(overall_samples) if i % overall_samples_per_class < used_samples_per_class]]
            cache_confidences = torch.load(f'{embedding_storage}/confidences.sav')[[i for i in range(overall_samples) if i % overall_samples_per_class < used_samples_per_class]]
            img_ctr = 0
            for img_ctr_wide in range(overall_samples):
                if img_ctr_wide % overall_samples_per_class < used_samples_per_class:
                    cache_data[img_ctr] = torch.load(f'{embedding_storage}/embedding_{img_ctr_wide}.sav')
                    img_ctr += 1
        except FileNotFoundError as ex:
            load_from_storage = False

    if not load_from_storage:
        # use the bottlenecked model to produce embeddings
        data_loader = dataset_util.get_loader(dataset, shuffle=False, order_labels=True, n_labels=overall_classes)
        model = model.to(model.device)
        num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        model.eval()
        with torch.no_grad():
            img_ctr_wide = -1
            img_ctr = 0
            for image, target in metric_logger.log_every(data_loader, len(data_loader.dataset), header):
                img_ctr_wide += data_loader.batch_size
                if img_ctr_wide%overall_samples_per_class >= used_samples_per_class:
                    continue
                if img_ctr >= used_samples:
                    break

                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                embeddings = model.forward_to_bn(image)
                output = model.forward_from_bn(embeddings)

                # process model output
                confidence = list()
                prediction = list()
                target_labels = list()
                predicted_labels = list()
                for i in range(output.shape[0]):
                    c, p = torch.max(torch.nn.functional.softmax(output[i], dim=0), 0)
                    confidence.append(c)
                    prediction.append(p)
                    target_labels.append(data_loader.dataset.classes[target[i]])
                    predicted_labels.append(data_loader.dataset.classes[p])

                # process embedding vector
                for i, embedding in enumerate(embeddings.cpu().detach().numpy()):
                    cache_labels_t[img_ctr] = target[i]
                    cache_labels[img_ctr] = prediction[i]
                    cache_confidences[img_ctr] = confidence[i]
                    cache_data[img_ctr] = embedding.reshape(np.prod(embedding.shape))
                    img_ctr += 1
        torch.set_num_threads(num_threads)
    if store:
        save_embeddings_on_storage(cache_data, cache_labels, cache_labels_t, cache_confidences, embedding_storage)
    return cache_data, cache_labels, cache_labels_t, cache_confidences


def save_embeddings_on_storage(cache_data, cache_labels, cache_labels_t, cache_confidences, storage_folder):
    if not os.path.exists(storage_folder):
        os.mkdir(storage_folder)
    for img_ctr, embedding in enumerate(cache_data):
        torch.save(cache_data[img_ctr], f'{storage_folder}/embedding_{img_ctr}.sav')
    torch.save(cache_labels, f'{storage_folder}/labels.sav')
    torch.save(cache_labels_t, f'{storage_folder}/labels_t.sav')
    torch.save(cache_confidences, f'{storage_folder}/confidences.sav')


def train_ee_model(mimic_model, ee_config, samples_fraction_per_class, train_dataset, valid_dataset, device):
    """

    Args:
        mimic_model:
        ee_config:
        samples_fraction_per_class: TODO this parameter is ignored
        train_dataset:
        valid_dataset:
        device:

    Returns:

    """
    mimic_model = mimic_model.to(mimic_model.device)
    ee_type = ee_config['type']
    shuffle = ee_config['shuffle_train_set']
    experiment_name = ee_config['experiment']
    results = dict()
    best_ee_model = dict()

    configurations = iterate_configurations(ee_type, ee_config['params'], device, ee_config['threshold'])

    for ee_params in configurations:
        n_labels = ee_params['n_labels']
        # samples_subset = used_samples_per_class * label_subset
        instance_key = f"{n_labels}:{samples_fraction_per_class}"

        # Initialize early exit model
        ee_model = early_classifier.ee_utils.models[ee_type](**ee_params)

        # Get data loaders
        pin_memory = 'cuda' in ee_model.device.type
        train_loader = dataset_util.get_loader(train_dataset, shuffle=shuffle, n_labels=n_labels, pin_memory=pin_memory)
        valid_loader = dataset_util.get_loader(valid_dataset, shuffle=False, n_labels=n_labels, pin_memory=pin_memory)

        results.setdefault(instance_key, dict())
        best_ee_model.setdefault(instance_key, dict())

        print(f"Fitting early exit model of type '{ee_type}' with parameters '{ee_params}'...")
        epochs = ee_params['epochs'] if 'epochs' in ee_params else 1
        for epoch in range(epochs):

            # Train early exit model one epoch
            ee_model.fit(train_loader, epoch=epoch)
            # Evaluate early exit model
            r = evaluate_ee_model(ee_model, mimic_model, valid_loader, device, use_threshold=True)

            results[instance_key].setdefault(ee_model.key_param(), r)
            if r['performance'] >= results[instance_key][ee_model.key_param()]['performance']:
                results[instance_key][ee_model.key_param()] = r
                best_ee_model[instance_key][ee_model.key_param()] = ee_model.to_state_dict()

    '''
    # store best model
    dname = ee_config['ckpt']
    Path(dname).mkdir(parents=True, exist_ok=True)
    for instance_key, ee_model in best_ee_model.items():
        label_subset, used_samples_per_class = parse.parse("{}:{}", instance_key)
        ee_model_file = dname.format(label_subset, used_samples_per_class, ee_model.key_param())
        ee_model.save(ee_model_file)
    '''

    # store best model for last configuration
    ee_params = configurations[-1]
    n_labels = ee_params['n_labels']
    instance_key = f"{n_labels}:{samples_fraction_per_class}"
    ee_model = early_classifier.ee_utils.models[ee_type](**ee_params)
    ee_model.from_state_dict(best_ee_model[instance_key][ee_model.key_param()])
    dname = ee_config['ckpt']
    Path(dname).mkdir(parents=True, exist_ok=True)
    ee_model_file = dname.format(n_labels, samples_fraction_per_class, ee_model.key_param())
    ee_model.save(ee_model_file)

    # store results on disk
    dname = f'ee_stats/{ee_type}/stats'
    Path(dname).mkdir(parents=True, exist_ok=True)
    with open(f"{dname}/{experiment_name}_{time.strftime('%Y%m%d-%H%M%S')}.json", "w") as f:
        json.dump(results, f)

def evaluate_ee_model(ee_model, mimic_model, data_loader, device, interval=100, split_name='Evaluate EE', use_threshold=True):

    mimic_model = mimic_model.to(mimic_model.device)
    ee_model = ee_model.to(ee_model.device)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_counter('early_predictions', CtrValue())
    metric_logger.add_meter('ee_acc1', SmoothedValue())
    metric_logger.add_meter('ee_acc5', SmoothedValue())
    metric_logger.add_meter('ee_acc1_c', SmoothedValue())
    metric_logger.add_meter('ee_acc5_c', SmoothedValue())
    header = '{}:'.format(split_name)
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    ee_model.eval()
    mimic_model.eval()

    for embeddings, target in metric_logger.log_every(data_loader, interval, header):
        embeddings = embeddings.to(mimic_model.device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if any([t > ee_model.n_labels - 1 for t in target]):
            break

        # run full model up to bottleneck
        # embeddings = mimic_model.forward_to_bn(image).to(device)
        # embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1:].numel())

        # ee prediction
        # predictions, confidences = ee_model.predict(np.array(embeddings))
        # confident_predictions = predictions[confidences < ee_model.get_threshold()]
        # confident_targets = target[confidences < ee_model.get_threshold()]
        embeddings = embeddings.to(ee_model.device)
        ee_output = ee_model.predict(embeddings)
        ee_conf = ee_model.get_prediction_confidences(ee_output)
        ee_output = ee_output.to(device)
        ee_conf = ee_conf.to(device)

        if not use_threshold:
            confidence_threshold = 0.2
        else:
            confidence_threshold = ee_model.get_threshold()
        confident_output = ee_output[ee_conf >= confidence_threshold]
        confident_targets = target[ee_conf >= confidence_threshold]

        # compute performance metrics
        early_predictions = confident_output.shape[0]
        acc1, acc5 = main_util.compute_accuracy(ee_output, target, topk=(1, 5))
        acc1_c, acc5_c = main_util.compute_accuracy(confident_output, confident_targets, topk=(1, 5))
        metric_logger.meters['ee_acc1'].update(acc1.item(), n=data_loader.batch_size)
        metric_logger.meters['ee_acc5'].update(acc5.item(), n=data_loader.batch_size)
        metric_logger.meters['ee_acc1_c'].update(acc1_c.item(), n=early_predictions)
        metric_logger.meters['ee_acc5_c'].update(acc5_c.item(), n=early_predictions)
        metric_logger.counters['early_predictions'].update(early_predictions, data_loader.batch_size)

    torch.set_num_threads(num_threads)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.ee_acc1.global_avg
    top5_accuracy = metric_logger.ee_acc5.global_avg
    top1_accuracy_c = metric_logger.ee_acc1_c.global_avg
    top5_accuracy_c = metric_logger.ee_acc5_c.global_avg
    early_predictions = metric_logger.early_predictions.global_avg
    # Compute a performance metric that considers both accuracy and percentage of confident predictions
    def compute_performance_metric(acc, frac, exp):
        """ The higher exp, the higher the importance given to the accuracy """
        return (((min(acc, 65) + math.log(max(1, acc - 65))) * 0.01) + 0.5) ** exp * math.log(frac * 100 + 1) * 0.01
    # performance_metric = 1.5 * top1_accuracy_c + 100*early_predictions
    # performance_metric = ((min(top1_accuracy_c, 85)*0.01)**2 + 0.5) * (early_predictions + 0.5)
    # performance_metric = ((min(top1_accuracy_c, 85)*0.01) + 0.5)**5 * early_predictions
    performance_metric = compute_performance_metric(top1_accuracy_c, early_predictions, 3.5)

    print(' * OVERALL:\t\tAcc@1 {:.4f}\tAcc@5 {:.4f}'.format(top1_accuracy, top5_accuracy))
    print(' * CONFIDENT:\t\tAcc@1 {:.4f}\tAcc@5 {:.4f}\t(fraction of early predictions {:.4f})'.format(top1_accuracy_c, top5_accuracy_c, early_predictions))
    print(' * Performance metric:\t{:.4f}'.format(performance_metric))

    results = ee_model.init_results()
    results['overall_accuracy'] = top1_accuracy
    results['confident_accuracy'] = top1_accuracy_c
    results['predicted'] = early_predictions
    results['performance'] = performance_metric

    return results

def run(args):
    distributed, device_ids = main_util.init_distributed_mode(args.world_size, args.dist_url)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        cudnn.benchmark = True

    print(args)
    config = yaml_util.load_yaml_file(args.config)
    # experiment_name = f"{config['dataset']['name']}_ver{config['student_model']['version']}-{config['student_model']['params']['bottleneck_channel']}ch"
    dataset_config = config['dataset']
    student_input_shape = config['input_shape']
    train_config = config['train']
    test_config = config['test']

    teacher_model_config = config['teacher_model']
    teacher_input_shape = yaml_util.load_yaml_file(teacher_model_config['config'])['input_shape']

    ee_config = config['ee_model']
    ee_threshold = ee_config['threshold']
    fraction_of_samples_per_class = ee_config['samples_fraction']
    load_embeddings = ee_config['load_embeddings']
    store_embeddings = ee_config['store_embeddings']
    embeddings_storage = ee_config['storage']
    ee_device = torch.device(ee_config['device'] if torch.cuda.is_available() else 'cpu')

    # Build datasets and loaders for full model
    input_shape = teacher_input_shape if teacher_input_shape[-1] > student_input_shape[-1] else student_input_shape
    train_dataset, valid_dataset, test_dataset = dataset_util.get_datasets(dataset_config,
                                                                           reshape_size=input_shape[1:3],
                                                                           rough_size=int(256/224*input_shape[-1]))
    pin_memory = 'cuda' in device.type
    train_loader = dataset_util.get_loader(train_dataset, shuffle=True, batch_size=train_config['batch_size'], pin_memory=pin_memory)
    valid_loader = dataset_util.get_loader(valid_dataset, shuffle=False, batch_size=test_config['batch_size'], pin_memory=pin_memory)
    test_loader = dataset_util.get_loader(test_dataset, shuffle=False, batch_size=test_config['batch_size'], pin_memory=pin_memory)

    # Train mimic model through distillation from the original model
    if args.bn_train:
        ee_model = None
        if args.ee_joint_train:
            ee_model = early_classifier.ee_utils.get_ee_model(ee_config, device)
        distill(train_loader, valid_loader, student_input_shape, teacher_input_shape, config, device, distributed,
                device_ids, ee_model=ee_model, ee_threshold=ee_threshold)

    # Generate embedding datasets
    org_model, teacher_model_type = mimic_util.get_org_model(teacher_model_config, device)
    mimic_model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device)
    cache_data, cache_labels, _, cache_confidences = get_embeddings(train_dataset, mimic_model,
                                                                    student_input_shape, device,
                                                                    fraction_of_samples=1.0,
                                                                    load_from_storage=load_embeddings,
                                                                    store=store_embeddings,
                                                                    embedding_storage=embeddings_storage)
    valid_data, _, valid_labels, valid_confidences = get_embeddings(valid_dataset, mimic_model,
                                                                    student_input_shape, device,
                                                                    fraction_of_samples=1.0,
                                                                    load_from_storage=False,
                                                                    store=False)
    ee_train_dataset = EmbeddingDataset(cache_data, cache_labels, cache_confidences)
    ee_valid_dataset = EmbeddingDataset(valid_data, valid_labels, valid_confidences)

    # Train the early exit model starting from an already trained student model
    if args.ee_solo_train:
        train_ee_model(mimic_model, ee_config, fraction_of_samples_per_class, ee_train_dataset, ee_valid_dataset, ee_device)

    # Test original model
    if args.org_eval:
        if distributed:
            org_model = DataParallel(org_model, device_ids=device_ids)
        evaluate(org_model, test_loader, device, title='[Original model]')

    # Test mimic model
    mimic_model_without_dp = mimic_model.module if isinstance(mimic_model, DataParallel) else mimic_model
    if args.bn_eval:
        if distributed:
            mimic_model = DataParallel(mimic_model, device_ids=device_ids)
        evaluate(mimic_model, test_loader, device, title='[BN model]')

    # Test early exit model
    ee_model = ee_utils.get_ee_model(ee_config, device, pre_trained=True)
    ee_valid_loader = dataset_util.get_loader(ee_valid_dataset, shuffle=False, n_labels=ee_model.n_labels)
    evaluate_ee_model(ee_model, mimic_model, ee_valid_loader, ee_device)

    # Joint test mimic model with early exit model
    file_util.save_pickle(mimic_model_without_dp, config['mimic_model']['ckpt'])
    if distributed:
        mimic_model = DistributedDataParallel(mimic_model_without_dp, device_ids=device_ids)
    ee_threshold = ee_model.get_threshold()
    if ee_model.n_labels == mimic_model.out_features:
        evaluate(mimic_model, test_loader, device, ee_model=ee_model, ee_threshold=ee_threshold, title='[BN_EE model]')


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
