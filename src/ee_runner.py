import argparse
import datetime
import json
import time
import copy
import os

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from early_classifier.base import BaseClassifier
from early_classifier import ee_utils
from early_classifier.ee_dataset import EmbeddingDataset

from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util, module_util
from structure.logger import MetricLogger, SmoothedValue, CtrValue
from utils import main_util, mimic_util, dataset_util, metric_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Learner')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('-bn_train', action='store_true', help='train a bottlenecked model by distilling a teacher')
    argparser.add_argument('-finetune', action='store_true', help='fine tune the mimic model after head distillation')
    argparser.add_argument('-org_eval', action='store_true', help='evaluate the original model')
    argparser.add_argument('-bn_eval', action='store_true', help='evaluate the bottlenecked model')
    argparser.add_argument('-metric_learning', action='store_true', help='optimize distance metric on embeddings')
    argparser.add_argument('-ee_joint_train', action='store_true', help='train an early exit model jointly')
    argparser.add_argument('-ee_solo_train', action='store_true', help='train an early exit model independently')
    # distributed training parameters
    argparser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    argparser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return argparser


def train_epoch(model, train_loader, optimizer, criterion, epoch, device, interval):
    model.train()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets, _ in metric_logger.log_every(train_loader, interval, header):
        start_time = time.time()
        sample_batch, targets = sample_batch.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(sample_batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        batch_size = sample_batch.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


def save_ckpt(student_model, epoch, best_valid_value, ckpt_file_path, teacher_model_type, ee_model=None,
              ee_config=None):
    print('Saving..')
    module = student_model.module if isinstance(student_model,
                                                (DataParallel, DistributedDataParallel)) else student_model
    state = {
        'type': teacher_model_type,
        'model': module.state_dict(),
        'epoch': epoch + 1,
        'best_valid_value': best_valid_value,
        'student': True
    }
    file_util.make_parent_dirs(ckpt_file_path)
    torch.save(state, ckpt_file_path)
    if ee_model is not None:
        dname = ee_config['ckpt']
        ee_model_file = dname.format(ee_model.n_labels, ee_config['samples_fraction'], ee_model.key_param())
        file_util.make_parent_dirs(ckpt_file_path)
        ee_model.save(ee_model_file)


@torch.no_grad()
def evaluate(model, data_loader, device, ee_model=None, interval=1000, split_name='Test', title=None):
    """
    Run the model on a test set and compute the top-1 and top-5 accuracy.
    The model is run jointly with an early exit model.
    Args:
        model:
        data_loader:
        device:
        ee_model (BaseClassifier):
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

    # overall_samples = 50000
    overall_samples_per_class = 100
    used_samples_per_class = 100
    # samples = int(overall_samples / overall_samples_per_class) * used_samples_per_class
    samples = len(data_loader.dataset)

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
        for image, target, _ in metric_logger.log_every(data_loader, interval, header, verbose=False):

            img_ctr_wide += data_loader.batch_size
            if img_ctr_wide % overall_samples_per_class >= used_samples_per_class:
                continue
            if img_ctr >= samples:
                break

            image = image.to(model.device, non_blocking=True)
            target = target.to(model.device, non_blocking=True)

            batch_size = image.shape[0]

            if not ee_model:
                output = model(image)
                new_early_exits = 0
            else:
                # run model up to bottleneck
                bn_output, *_ = model.forward_to_bn(image)
                embeddings = bn_output.to(ee_model.device)
                embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1:].numel())

                # early prediction
                ee_output = ee_model.predict(embeddings)
                ee_conf = ee_model.get_prediction_confidences(ee_output)
                if ee_model.n_labels < model.out_features:
                    ee_output = torch_f.pad(ee_output, pad=(0, model.out_features - ee_model.n_labels, 0, 0), value=0)

                # forward not-confident vectors to the full model
                # c, p = torch.max(torch.nn.functional.softmax(ee_output[i], dim=0), 0)
                # full_predictions = bn_output[ee_output < ee_threshold].gpu()
                full_predictions = bn_output[ee_conf < ee_model.get_threshold()]
                new_early_exits = batch_size - full_predictions.shape[0]
                output = ee_output.to(model.device)
                # early_exit_ctr += new_early_exits

                if full_predictions.shape[0] > 0:
                    full_output = model.forward_from_bn(full_predictions)
                    # merge early and full predictions
                    j = 0
                    for i in range(len(ee_output)):
                        if ee_conf[i] < ee_model.get_threshold():
                            output[i] = full_output[j]
                            j += 1

                if ee_model.get_threshold() == 1 and new_early_exits > 0:
                    print(new_early_exits)

            acc1, acc5 = main_util.compute_accuracy(output, target, topk=(1, 5))
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            metric_logger.counters['early_predictions'].update(new_early_exits, batch_size)

    # gather the solo_train-solo_eval from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    early_predictions = metric_logger.early_predictions.global_avg
    print(' * Acc@1 {:.4f}\tAcc@5 {:.4f}'.format(top1_accuracy, top5_accuracy))
    print(' * Fraction of early predictions {:.4f}'.format(early_predictions))
    torch.set_num_threads(num_threads)

    results = ee_model.init_results() if ee_model else dict()
    results['overall_accuracy'] = top1_accuracy
    results['confident_accuracy'] = top1_accuracy
    results['coverage'] = early_predictions

    return results


def validate(student_model_without_ddp, data_loader, config, device, distributed, device_ids, ee_model):
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

    Returns:

    """
    teacher_model_config = config['teacher_model']
    org_model, teacher_model_type = mimic_util.get_org_model(teacher_model_config, device)
    mimic_model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config,
                                             device, head_model=student_model_without_ddp)
    mimic_model_without_dp = mimic_model.module if isinstance(mimic_model, DataParallel) else mimic_model
    if distributed:
        mimic_model = DistributedDataParallel(mimic_model_without_dp, device_ids=device_ids)
    mimic_accuracy = evaluate(mimic_model, data_loader, device, split_name='Mimic Validation')["overall_accuracy"]
    ee_accuracy = evaluate(mimic_model, data_loader, device, ee_model=ee_model, split_name='EE Validation')[
        "overall_accuracy"] if ee_model else mimic_accuracy
    return 0.5 * ee_accuracy + 0.5 * mimic_accuracy


def distill_one_epoch(student_model, teacher_model, teacher_input_size, student_input_size, train_loader, optimizer,
                      criterion, epoch, device, interval, bn_shape, loss_c, ee_model=None):
    student_model.train()
    teacher_model.eval()
    if ee_model:
        ee_model.train()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    student_upsampler = torch.nn.Upsample(student_input_size).to(device)
    teacher_upsampler = torch.nn.Upsample(teacher_input_size).to(device)
    for sample_batch, targets, indexes in metric_logger.log_every(train_loader, interval, header):
        start_time = time.time()
        sample_batch, targets = sample_batch.to(device), targets.to(device)
        batch_size = sample_batch.shape[0]
        optimizer.zero_grad()
        teacher_outputs = teacher_model(teacher_upsampler(sample_batch))
        cls_loss = 0
        reg_loss = 0
        if ee_model:
            ee_model.train()
            # get embedding
            z, mu, logvar = student_model.forward_to_bn(student_upsampler(sample_batch))
            # embedding = z.detach()
            embedding = z.reshape((z.shape[0], np.prod(z.shape[1:])))
            # early prediction
            ee_outputs = ee_model.forward(embedding)
            # full prediction
            student_outputs = student_model.forward_from_bn(z)
            # update ee model
            ee_model.update_and_fit(embedding, indexes, epoch)
            # classification loss
            cls_loss = ee_model.get_cls_loss(ee_outputs, targets)
            # discrepancy loss (regularization)
            if student_model.variational:
                mu = mu.reshape((mu.shape[0], np.prod(mu.shape[1:])))
                logvar = logvar.reshape((mu.shape[0], np.prod(mu.shape[1:])))
                # kld = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                reg_loss = ee_model.get_kl_divergence(mu, logvar, targets)
                if reg_loss.isnan():
                    print(reg_loss)
                    reg_loss = 0
        else:
            student_outputs = student_model(student_upsampler(sample_batch))
        mimic_loss = criterion(student_outputs, teacher_outputs)
        loss = loss_c[0] * mimic_loss + loss_c[2] * cls_loss + loss_c[1] * reg_loss

        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


def distill(train_loader, valid_loader, student_input_shape, teacher_input_shape, config, device, distributed,
            device_ids, bn_shape, loss_c=None, ee_model=None):
    """
    Train the (head of a) student model by knowledge distillation from the teacher model.
    The student is stored in ckpt.
    Args:
        bn_shape:
        train_loader:
        valid_loader:
        student_input_shape:
        teacher_input_shape:
        config:
        device:
        distributed:
        device_ids:
        loss_c (tuple):
        ee_model (BaseClassifier):

    Returns:

    """
    teacher_model_config = config['teacher_model']
    teacher_model, teacher_model_type = mimic_util.get_teacher_model(teacher_model_config, teacher_input_shape, device)
    module_util.freeze_module_params(teacher_model)
    student_model_config = config['student_model']
    student_model = mimic_util.get_student_model(teacher_model_type, student_model_config, config['dataset']['name'],
                                                 student_input_shape[-1])
    student_model = student_model.to(device)
    student_model.device = device
    start_epoch, best_valid_acc = mimic_util.resume_from_ckpt(student_model_config['ckpt'], student_model, device,
                                                              is_student=True)
    if best_valid_acc is None:
        best_valid_acc = 0.0

    train_config = config['train']
    criterion_config = train_config['criterion']
    criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
    optim_config = train_config['optimizer']
    optimizer = func_util.get_optimizer(
        list(student_model.parameters()) + (list(ee_model.get_model_parameters()) if ee_model else list()),
        optim_config['type'], optim_config['params'])
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
    end_epoch = train_config['epoch']
    start_time = time.time()

    # build initial embedding cache for early exit
    if ee_model and ee_model.requires_full_fit:
        print("Initializing cache for early exit joint training...")
        cache_data, cache_labels, _, cache_confidences = get_embeddings(train_loader.dataset, config, device,
                                                                        fraction_of_samples=1.0,
                                                                        load_from_storage=False,
                                                                        store=False)
        ee_train_dataset = EmbeddingDataset(cache_data, cache_labels, cache_confidences)
        ee_model.init_and_fit(ee_train_dataset)
    for epoch in range(1, start_epoch):
        scheduler.step()
    for epoch in range(start_epoch, end_epoch + 1):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        # distill
        distill_one_epoch(student_model, teacher_model, student_input_shape[-1], teacher_input_shape[-1], train_loader,
                          optimizer, criterion, epoch, device, interval, bn_shape, loss_c, ee_model)
        # evaluate
        valid_acc = validate(student_model, valid_loader, config, device, distributed, device_ids, ee_model)
        if valid_acc > best_valid_acc and main_util.is_main_process():
            print('Updating ckpt (Best top1 accuracy: {:.4f} -> {:.4f})'.format(best_valid_acc, valid_acc))
            best_valid_acc = valid_acc
            save_ckpt(student_model_without_ddp, epoch, best_valid_acc, ckpt_file_path, teacher_model_type,
                      ee_model=ee_model, ee_config=config["ee_model"])
        scheduler.step()
        # fit the ee model to the new embeddings
        if ee_model:
            ee_model.init_and_fit()

    # dist.barrier()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    del teacher_model
    del student_model


def finetune(train_loader, valid_loader, student_input_shape, config, device, distributed, device_ids, bn_shape):
    # Models
    teacher_model_config = config['teacher_model']
    org_model, teacher_model_type = mimic_util.get_org_model(teacher_model_config, device)
    student_model_config = config['student_model']
    student_model = mimic_util.get_student_model(teacher_model_type, student_model_config, config['dataset']['name'],
                                                 student_input_shape[-1])
    _, best_valid_acc = mimic_util.resume_from_ckpt(student_model_config['ckpt'], student_model, device,
                                                    is_student=True)
    mimic_model_config = config['mimic_model']
    mimic_model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config,
                                             device, head_model=student_model)
    mimic_model.to(device)
    mimic_model.device = device
    mimic_model.head.device = device

    # Train conf
    train_config = config['train2']
    criterion_config = train_config['criterion']
    criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
    optim_config = train_config['optimizer']
    optimizer = func_util.get_optimizer(
        list(mimic_model.parameters()),
        optim_config['type'], optim_config['params'])
    scheduler_config = train_config['scheduler']
    scheduler = func_util.get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
    interval = train_config['interval']
    if interval <= 0:
        num_batches = len(train_loader)
        interval = num_batches // 20 if num_batches >= 20 else 1

    ckpt_file_path = mimic_model_config['ckpt']
    end_epoch = train_config['epoch']
    start_time = time.time()

    mimic_model.head.freeze_encoder()

    for epoch in range(1, end_epoch):
        train_epoch(mimic_model, train_loader, optimizer, criterion, epoch, device, interval)
        valid_acc = evaluate(mimic_model, valid_loader, device, split_name='Mimic Validation')["overall_accuracy"]
        if valid_acc > best_valid_acc and main_util.is_main_process():
            print('Updating ckpt (Best top1 accuracy: {:.4f} -> {:.4f})'.format(best_valid_acc, valid_acc))
            best_valid_acc = valid_acc
            save_ckpt(mimic_model, epoch, best_valid_acc, ckpt_file_path, teacher_model_type)
        scheduler.step()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    del org_model
    del student_model
    del mimic_model


def get_embeddings(dataset, config, device, fraction_of_samples=1.0, split_name='Get Embeddings', scale=1.0,
                   store_prefix='', load_from_storage=False, store=False, embedding_storage=None, use_ckpt=False):
    """

    Args:
        use_ckpt:
        config:
        scale:
        dataset:
        device:
        fraction_of_samples:
        split_name:
        store_prefix:
        load_from_storage:
        store:
        embedding_storage:

    Returns:

    """

    org_model, teacher_model_type = mimic_util.get_org_model(config["teacher_model"], device)
    model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, config["teacher_model"], device,
                                       use_ckpt=use_ckpt)
    bn_shape = model.head.bn_shape(config["input_shape"], device)

    overall_samples = len(dataset)
    overall_classes = max(dataset.targets) + 1
    overall_samples_per_class = overall_samples / overall_classes
    used_samples_per_class = int(fraction_of_samples * overall_samples_per_class)
    used_samples = overall_classes * used_samples_per_class
    # bn_shape = model.head.bn_shape(input_shape, device)

    cache_labels = np.zeros([used_samples], dtype=int)
    cache_labels_t = np.zeros([used_samples], dtype=int)
    cache_confidences = np.zeros([used_samples])
    cache_data = np.zeros([used_samples, np.prod(bn_shape)], dtype=np.float32)

    metric_logger = MetricLogger(delimiter='  ')
    header = '{}:'.format(split_name)

    if load_from_storage:
        print(f"Loading previous embeddings tensors from disk...")
        try:
            cache_labels = torch.load(f'{embedding_storage}/{store_prefix}labels.sav')[
                [i for i in range(overall_samples) if i % overall_samples_per_class < used_samples_per_class]]
            cache_labels_t = torch.load(f'{embedding_storage}/{store_prefix}labels_t.sav')[
                [i for i in range(overall_samples) if i % overall_samples_per_class < used_samples_per_class]]
            cache_confidences = torch.load(f'{embedding_storage}/{store_prefix}confidences.sav')[
                [i for i in range(overall_samples) if i % overall_samples_per_class < used_samples_per_class]]
            img_ctr = 0
            for img_ctr_wide in range(overall_samples):
                if img_ctr_wide % overall_samples_per_class < used_samples_per_class:
                    embedding = torch.load(f'{embedding_storage}/{store_prefix}embedding_{img_ctr_wide}.sav')
                    cache_data[img_ctr] = embedding
                    # save embeddings as pictures
                    # bn_util.intermediate_output_to_fig(embedding.reshape(bn_shape), img_ctr, dataset.classes[cache_labels_t[img_ctr]], dataset.classes[cache_labels[img_ctr]], cache_confidences[img_ctr])
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
            for image, target, _ in metric_logger.log_every(data_loader, len(data_loader.dataset), header):
                img_ctr_wide += data_loader.batch_size
                if img_ctr_wide % overall_samples_per_class >= used_samples_per_class:
                    continue
                if img_ctr >= used_samples:
                    break

                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                embeddings, *_ = model.forward_to_bn(image)
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
        save_embeddings_on_storage(cache_data, cache_labels, cache_labels_t, cache_confidences, embedding_storage,
                                   store_prefix)
    return cache_data, cache_labels, cache_labels_t, cache_confidences


def save_embeddings_on_storage(cache_data, cache_labels, cache_labels_t, cache_confidences, storage_folder,
                               store_prefix):
    if not os.path.exists(storage_folder):
        os.mkdir(storage_folder)
    for img_ctr, embedding in enumerate(cache_data):
        torch.save(cache_data[img_ctr], f'{storage_folder}/{store_prefix}embedding_{img_ctr}.sav')
    torch.save(cache_labels, f'{storage_folder}/{store_prefix}labels.sav')
    torch.save(cache_labels_t, f'{storage_folder}/{store_prefix}labels_t.sav')
    torch.save(cache_confidences, f'{storage_folder}/{store_prefix}confidences.sav')


def train_ee_model(mimic_model, ee_config, samples_fraction_per_class, train_dataset, valid_dataset, bn_shape, device):
    """

    Args:
        bn_shape:
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

    # Normalize if needed
    # if ee_utils.requires_normalization(ee_type):
    #     train_dataset.normalize_data(bn_shape)
    #     valid_dataset.normalize_data(bn_shape)

    configurations, thresholds = ee_utils.iterate_configurations(ee_type, ee_config['params'], device, bn_shape,
                                                                 ee_config['thresholds'])

    for variant, ee_params in enumerate(configurations):
        n_labels = ee_params['n_labels']
        batch_size = ee_config['params']['batch_size'] if 'batch_size' in ee_config['params'] else 32
        instance_key = f"{n_labels}:{samples_fraction_per_class}"
        results.setdefault(instance_key, dict())
        best_ee_model.setdefault(instance_key, dict())
        # samples_subset = used_samples_per_class * label_subset

        # Load pre-trained early exit model
        ee_model = ee_utils.get_ee_model(ee_config, device, bn_shape, pre_trained=True, conf_idx=variant)
        if ee_model is None:
            # Initialize early exit model
            print(f"Training new ee model from scratch.")
            ee_model = ee_utils.models[ee_type](**ee_params)
            ee_model.init_param_from_dataset(train_dataset)
        else:
            print(f"Restoring pre-trained model from disk (performance metric = {ee_model.performance}).")
            results[instance_key].setdefault(ee_model.key_param(), dict())
            results[instance_key][ee_model.key_param()].setdefault(thresholds[-1], {'performance': ee_model.performance})

        # Get data loaders
        pin_memory = 'cuda' in ee_model.device.type
        train_loader = dataset_util.get_loader(train_dataset, batch_size=batch_size, shuffle=shuffle, n_labels=n_labels,
                                               pin_memory=pin_memory)
        valid_loader = dataset_util.get_loader(valid_dataset, shuffle=False, n_labels=n_labels, pin_memory=pin_memory)

        print(f"Fitting early exit model of type '{ee_type}' with parameters '{ee_params}'...")
        epochs = ee_params['epochs'] if 'epochs' in ee_params else 1
        for epoch in range(0, ee_model.last_epoch + 1):
            ee_model.scheduler.step()
        for epoch in range(ee_model.last_epoch + 1, epochs):
            # Train early exit model one epoch
            ee_model.train()
            ee_model.fit(train_loader, epoch=epoch)
            # Evaluate early exit model
            for threshold in thresholds:
                ee_model.set_threshold(threshold)
                ee_model.eval()
                r = evaluate_ee_model(ee_model, mimic_model, valid_loader, device, use_threshold=True)

                results[instance_key].setdefault(ee_model.key_param(), dict())
                results[instance_key][ee_model.key_param()].setdefault(threshold, r)

                if threshold == thresholds[-1]:
                    if r['performance'] >= results[instance_key][ee_model.key_param()][threshold]['performance']:
                        print('Updating ckpt (Best performance value: {:.4f} -> {:.4f})'.format(
                            results[instance_key][ee_model.key_param()][threshold]['performance'], r['performance']))
                        results[instance_key][ee_model.key_param()][threshold] = r
                        ee_model.performance = r['performance']
                        best_ee_model[instance_key][ee_model.key_param()] = copy.deepcopy(ee_model.to_state_dict())
                '''
                if r['performance'] >= results[instance_key][ee_model.key_param()][threshold]['performance']:
                    print('Updating ckpt (Best performance value: {:.4f} -> {:.4f})'.format(
                        results[instance_key][ee_model.key_param()][threshold]['performance'], r['performance']))
                    results[instance_key][ee_model.key_param()][threshold] = r
                    if threshold == thresholds[0]:
                        best_ee_model[instance_key][ee_model.key_param()] = copy.deepcopy(ee_model.to_state_dict())
                '''
            # store best models
            if epoch % 1 == 0:
                b_ee_model = ee_utils.models[ee_type](**ee_params)
                if b_ee_model.key_param() in best_ee_model[instance_key]:
                    print(f"Saving best model so far for instance {instance_key}, key {b_ee_model.key_param()}")
                    b_ee_model.from_state_dict(best_ee_model[instance_key][b_ee_model.key_param()])
                    dname = ee_config['ckpt']
                    Path(dname).mkdir(parents=True, exist_ok=True)
                    ee_model_file = dname.format(n_labels, samples_fraction_per_class, b_ee_model.key_param())
                    b_ee_model.save(ee_model_file)
        print(f"Training summary: {ee_model.training_history}")

    '''
    # store best model
    dname = ee_config['ckpt']
    Path(dname).mkdir(parents=True, exist_ok=True)
    for instance_key, ee_model in best_ee_model.items():
        label_subset, used_samples_per_class = parse.parse("{}:{}", instance_key)
        ee_model_file = dname.format(label_subset, used_samples_per_class, ee_model.key_param())
        ee_model.save(ee_model_file)

    # store best models
    for ee_params in configurations:
        n_labels = ee_params['n_labels']
        instance_key = f"{n_labels}:{samples_fraction_per_class}"
        ee_model = early_classifier.ee_utils.models[ee_type](**ee_params)
        if ee_model.key_param() in best_ee_model[instance_key]:
            print(f"Saving best model so far for instance {instance_key}, key {ee_model.key_param()}")
            ee_model.from_state_dict(best_ee_model[instance_key][ee_model.key_param()])
            dname = ee_config['ckpt']
            Path(dname).mkdir(parents=True, exist_ok=True)
            ee_model_file = dname.format(n_labels, samples_fraction_per_class, ee_model.key_param())
            ee_model.save(ee_model_file)
    '''

    '''
    # store results on disk
    dname = f'ee_stats/{ee_type}/solo_train'
    Path(dname).mkdir(parents=True, exist_ok=True)
    with open(f"{dname}/{experiment_name}_{time.strftime('%Y%m%d-%H%M%S')}.json", "w") as f:
        json.dump(results, f)
    '''


def evaluate_ee_model(ee_model, mimic_model, data_loader, device, interval=100, use_threshold=True):
    mimic_model = mimic_model.to(mimic_model.device)
    ee_model = ee_model.to(ee_model.device)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_counter('early_predictions', CtrValue())
    metric_logger.add_meter('ee_acc1', SmoothedValue())
    metric_logger.add_meter('ee_acc5', SmoothedValue())
    metric_logger.add_meter('ee_acc1_c', SmoothedValue())
    metric_logger.add_meter('ee_acc5_c', SmoothedValue())
    metric_logger.add_meter('cls_loss', SmoothedValue())

    if not use_threshold:
        confidence_threshold = 0
    else:
        confidence_threshold = ee_model.get_threshold()
    split_name = 'Evaluate EE t={}'.format(ee_model.get_threshold(normalized=False))
    header = '{}:'.format(split_name)

    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    ee_model.eval()
    mimic_model.eval()

    for embeddings, target in metric_logger.log_every(data_loader, interval, header, verbose=False):
        embeddings = embeddings.to(mimic_model.device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = embeddings.shape[0]
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

        confident_output = ee_output[ee_conf >= confidence_threshold]
        confident_targets = target[ee_conf >= confidence_threshold]

        # compute performance metrics
        early_predictions = confident_output.shape[0]
        acc1, acc5 = main_util.compute_accuracy(ee_output, target, topk=(1, 5))
        acc1_c, acc5_c = main_util.compute_accuracy(confident_output, confident_targets, topk=(1, 5))
        metric_logger.meters['cls_loss'].update(ee_model.get_cls_loss(ee_output, target), n=batch_size)
        metric_logger.meters['ee_acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['ee_acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['ee_acc1_c'].update(acc1_c.item(), n=early_predictions)
        metric_logger.meters['ee_acc5_c'].update(acc5_c.item(), n=early_predictions)
        metric_logger.counters['early_predictions'].update(early_predictions, batch_size)

    torch.set_num_threads(num_threads)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.ee_acc1.global_avg
    top5_accuracy = metric_logger.ee_acc5.global_avg
    top1_accuracy_c = metric_logger.ee_acc1_c.global_avg
    top5_accuracy_c = metric_logger.ee_acc5_c.global_avg
    cls_loss = metric_logger.cls_loss.global_avg
    early_predictions = metric_logger.early_predictions.global_avg

    ''' # Compute a performance metric that considers both accuracy and percentage of confident predictions
    def compute_performance_metric(acc, frac, exp):
        """ The higher the exp, the higher the importance given to the accuracy """
        return (((min(acc, 65) + math.log(max(1, acc - 65))) * 0.01) + 0.5) ** exp * math.log(frac * 100 + 1) * 0.01

    performance_metric = compute_performance_metric(top1_accuracy_c, early_predictions, 3.5)
    '''

    performance_metric = (1 / cls_loss.log()).item()

    print(' * OVERALL:\t\tAcc@1 {:.4f}\tAcc@5 {:.4f}'.format(top1_accuracy, top5_accuracy))
    print(' * CONFIDENT:\t\tAcc@1 {:.4f}\tAcc@5 {:.4f}\t(fraction of early predictions {:.4f})'.format(top1_accuracy_c,
                                                                                                       top5_accuracy_c,
                                                                                                       early_predictions))
    # print(' * Performance metric:\t{:.4f}'.format(performance_metric))

    results = ee_model.init_results()
    results['overall_accuracy'] = top1_accuracy
    results['confident_accuracy'] = top1_accuracy_c
    results['coverage'] = early_predictions
    results['performance'] = performance_metric

    return results


def run(args):
    str_time = time.strftime('%Y%m%d-%H%M%S')
    distributed, device_ids = main_util.init_distributed_mode(args.world_size, args.dist_url)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        cudnn.benchmark = True

    print(args)
    print(str_time)
    config = yaml_util.load_yaml_file(args.config)
    dataset_config = config['dataset']
    student_input_shape = config['input_shape']
    train_config = config['train']
    test_config = config['test']

    teacher_model_config = config['teacher_model']
    teacher_input_shape = yaml_util.load_yaml_file(teacher_model_config['config'])['input_shape']

    ee_config = config['ee_model']
    fraction_of_samples_per_class = ee_config['samples_fraction']
    load_embeddings = ee_config['load_embeddings'] if not args.bn_train else False
    store_embeddings = ee_config['store_embeddings']
    embeddings_storage = ee_config['storage']
    ee_device = torch.device(ee_config['device'] if torch.cuda.is_available() else 'cpu')
    ee_config['thresholds'] = ee_config['thresholds'] if type(ee_config['thresholds']) == list else [ee_config['thresholds']]
    n_labels = 100

    org_model, teacher_model_type = mimic_util.get_org_model(teacher_model_config, device)
    mimic_model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device)
    bn_shape = mimic_model.head.bn_shape(student_input_shape, device)

    # Build datasets and loaders for full model
    input_shape = teacher_input_shape if teacher_input_shape[-1] > student_input_shape[-1] else student_input_shape
    train_dataset, valid_dataset, test_dataset = dataset_util.get_datasets(dataset_config,
                                                                           reshape_size=input_shape[1:3],
                                                                           rough_size=int(256 / 224 * input_shape[-1]))
    pin_memory = 'cuda' in device.type
    train_loader = dataset_util.get_loader(train_dataset, n_labels=n_labels, shuffle=True,
                                           batch_size=train_config['batch_size'], pin_memory=pin_memory)
    valid_loader = dataset_util.get_loader(valid_dataset, n_labels=n_labels, shuffle=False,
                                           batch_size=test_config['batch_size'], pin_memory=pin_memory)
    test_loader = dataset_util.get_loader(test_dataset, n_labels=n_labels, shuffle=False,
                                          batch_size=test_config['batch_size'], pin_memory=pin_memory)

    # [bn_train] Train mimic model through distillation from the original model
    if args.bn_train:
        print("[Distilling Mimic Model]")
        ee_model = None
        loss_coefficients = (1, 0, 0)
        if args.ee_joint_train:
            print("- EE Model is trained jointly.")
            ee_model = ee_utils.get_ee_model(ee_config, device, bn_shape, pre_trained=False)
            ee_model.set_threshold(ee_config['thresholds'][0])
            ee_model.jointly_trained = True
            loss_coefficients = tuple(ee_config['params']['joint_loss_coefficients'].values())
        distill(train_loader, valid_loader, student_input_shape, teacher_input_shape, config, device, distributed,
                device_ids, bn_shape, ee_model=ee_model, loss_c=loss_coefficients)
    # [finetune] Keep training mimic model freezing the layers before the bottleneck
    if args.finetune:
        print("[Fine tuning Mimic Model]")
        finetune(train_loader, valid_loader, student_input_shape, config, device, distributed, device_ids, bn_shape)

    # Generate embedding datasets
    cache_data, cache_labels, _, cache_confidences = get_embeddings(train_dataset, config, device,
                                                                    fraction_of_samples=1.0,
                                                                    load_from_storage=load_embeddings,
                                                                    store=store_embeddings,
                                                                    embedding_storage=embeddings_storage,
                                                                    use_ckpt=True)
    valid_data, _, valid_labels, valid_confidences = get_embeddings(valid_dataset, config, device,
                                                                    fraction_of_samples=1.0,
                                                                    load_from_storage=load_embeddings,
                                                                    store=store_embeddings,
                                                                    embedding_storage=embeddings_storage,
                                                                    use_ckpt=True,
                                                                    store_prefix='v_')
    ee_train_dataset = EmbeddingDataset(torch.tensor(cache_data), cache_labels, cache_confidences)
    ee_valid_dataset = EmbeddingDataset(torch.tensor(valid_data), valid_labels, valid_confidences)

    # bn_util.intermediate_output_to_fig(ee_train_dataset.data[0].reshape(bn_shape), 0,
    #                                    train_dataset.classes[cache_labels[0]], train_dataset.classes[cache_labels[0]],
    #                                    cache_confidences[0])

    # [metric_learning] Learn an optimal distance metric on the embedding space
    if args.metric_learning:
        print("[Metric Learning]")
        metric_model = metric_util.learn_dataset(ee_train_dataset, device)
        ee_train_dataset = metric_util.transform_dataset(metric_model, ee_train_dataset, device)
        ee_valid_dataset = metric_util.transform_dataset(metric_model, ee_valid_dataset, device)
        bn_shape = ee_train_dataset.data.shape[1:]

    # [ee_solo_train] Train the early exit model starting from an already trained student model
    if args.ee_solo_train:
        print("[Training EE Model disjointedly]")
        mimic_model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device,
                                                 use_ckpt=True)
        module_util.freeze_module_params(mimic_model)
        train_ee_model(mimic_model, ee_config, fraction_of_samples_per_class, ee_train_dataset, ee_valid_dataset,
                       bn_shape, ee_device)

    # [org_eval] Test original model
    org_model, teacher_model_type = mimic_util.get_org_model(teacher_model_config, device)
    if args.org_eval:
        print("[Test Original Model]")
        if distributed:
            org_model = DataParallel(org_model, device_ids=device_ids)
        evaluate(org_model, test_loader, device, title='[Original model]')

    # [bn_eval] Test mimic model
    mimic_model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device,
                                             use_ckpt=True)
    mimic_model_without_dp = mimic_model.module if isinstance(mimic_model, DataParallel) else mimic_model
    if args.bn_eval:
        print("[Test Mimic Model]")
        if distributed:
            mimic_model = DataParallel(mimic_model, device_ids=device_ids)
        evaluate(mimic_model, test_loader, device, title='[BN model]')

    # Test early exit models
    results = dict()
    for variant in range(ee_utils.num_ee_models_variants(ee_config, device, bn_shape)):
        ee_model = ee_utils.get_ee_model(ee_config, device, bn_shape, pre_trained=True, conf_idx=variant)
        if ee_model:
            print("[Test EE Model]")
            ee_valid_loader = dataset_util.get_loader(ee_valid_dataset, shuffle=False, n_labels=ee_model.n_labels)
            results.setdefault(f"{ee_model.n_labels}:{fraction_of_samples_per_class}", dict())
            results[f"{ee_model.n_labels}:{fraction_of_samples_per_class}"].setdefault(ee_model.key_param(), dict())
            for threshold in ee_config['thresholds']:
                ee_model.set_threshold(threshold)
                r = evaluate_ee_model(ee_model, mimic_model, ee_valid_loader, ee_device, use_threshold=True)
                results[f"{ee_model.n_labels}:{fraction_of_samples_per_class}"][ee_model.key_param()][threshold] = r
        # store results on disk
        if results:
            dname = f'ee_stats/{ee_config["type"]}/{"joint_train" if ee_model.jointly_trained else "solo_train"}-solo_eval'
            Path(dname).mkdir(parents=True, exist_ok=True)
            with open(f"{dname}/{ee_config['experiment']}_{str_time}.json", "w") as f:
                json.dump(results, f)

    # Test jointly mimic model with early exit model
    mimic_model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device,
                                             use_ckpt=True)
    ee_model = ee_utils.get_ee_model(ee_config, device, bn_shape, pre_trained=True)
    if distributed:
        mimic_model = DistributedDataParallel(mimic_model_without_dp, device_ids=device_ids)
    if mimic_model:  # and ee_model.n_labels == mimic_model.out_features:
        joint_results = dict()
        for threshold in ee_config['thresholds']:
            ee_model.set_threshold(threshold)
            r = evaluate(mimic_model, test_loader, device, ee_model=ee_model, title=f'[BN_EE model - t={threshold}]')
            joint_results[threshold] = r
        # store results on disk
        joint_results = {f"{ee_model.n_labels}:{fraction_of_samples_per_class}": {ee_model.key_param(): joint_results}}
        dname = f'ee_stats/{ee_config["type"]}/{"joint_train" if ee_model.jointly_trained else "solo_train"}-joint_eval'
        Path(dname).mkdir(parents=True, exist_ok=True)
        with open(f"{dname}/{ee_config['experiment']}_{str_time}.json", "w") as f:
            json.dump(joint_results, f)


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
