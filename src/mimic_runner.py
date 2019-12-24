import argparse

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util, module_util
from utils import main_util, mimic_util
from utils.dataset import general_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Learner')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('--aux', type=float, default=100.0, help='auxiliary weight')
    argparser.add_argument('-test_only', action='store_true', help='only test model')
    # distributed training parameters
    argparser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    argparser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return argparser


# TODO: add metric logger
def distill_one_epoch(student_model, teacher_model, train_loader, optimizer, criterion, epoch, device, interval, aux_weight):
    print('\nEpoch: {}, LR: {:.3E}'.format(epoch, optimizer.param_groups[0]['lr']))
    student_model.train()
    teacher_model.eval()
    train_size = len(train_loader.sampler)
    train_loss = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        student_outputs = student_model(inputs)
        teacher_outputs = teacher_model(inputs)
        if isinstance(student_outputs, tuple):
            student_outputs, aux = student_outputs[0], student_outputs[1]
            loss = criterion(student_outputs, teacher_outputs) + aux_weight * nn.functional.cross_entropy(aux, targets)
        else:
            loss = criterion(student_outputs, teacher_outputs)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += targets.size(0)
        if batch_idx > 0 and batch_idx % interval == 0:
            print('[{}/{} ({:.0f}%)]\tAvg Loss: {:.6f}'.format(total, train_size, 100.0 * total / train_size,
                                                               loss.item() / targets.size(0)))


# TODO: 1) add metric logger \
#  and 2) temporarily build mimic model to compute validation accuracy (and detach the tail at the end of this function
@torch.no_grad()
def validate(student_model, teacher_model, valid_loader, criterion, device):
    student_model.eval()
    teacher_model.eval()
    valid_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            student_outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs)
            loss = criterion(student_outputs, teacher_outputs)
            valid_loss += loss.item()
            total += targets.size(0)

    avg_valid_loss = valid_loss / total
    print('Validation Loss: {:.6f}\tAvg Loss: {:.6f}'.format(valid_loss, avg_valid_loss))
    return avg_valid_loss


def save_ckpt(student_model, epoch, best_avg_loss, ckpt_file_path, teacher_model_type):
    print('Saving..')
    module =\
        student_model.module if isinstance(student_model, (DataParallel, DistributedDataParallel)) else student_model
    state = {
        'type': teacher_model_type,
        'model': module.state_dict(),
        'epoch': epoch + 1,
        'best_avg_loss': best_avg_loss,
        'student': True
    }
    file_util.make_parent_dirs(ckpt_file_path)
    torch.save(state, ckpt_file_path)


def predict(inputs, targets, model):
    preds = model(inputs)
    loss = nn.functional.cross_entropy(preds, targets)
    _, pred_labels = preds.max(1)
    correct_count = pred_labels.eq(targets).sum().item()
    return correct_count, loss.item()


def distill(train_loader, valid_loader, input_shape, aux_weight, config, device, distributed, device_ids):
    teacher_model_config = config['teacher_model']
    teacher_model, teacher_model_type = mimic_util.get_teacher_model(teacher_model_config, input_shape, device)
    module_util.freeze_module_params(teacher_model)
    student_model_config = config['student_model']
    student_model = mimic_util.get_student_model(teacher_model_type, student_model_config)
    student_model = student_model.to(device)
    start_epoch, best_avg_loss = mimic_util.resume_from_ckpt(student_model_config['ckpt'], student_model,
                                                             is_student=True)
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
        interval = num_batches // 100 if num_batches >= 100 else 1

    if distributed:
        teacher_model = DataParallel(teacher_model, device_ids=device_ids)
        student_model = DistributedDataParallel(student_model, device_ids=device_ids)

    ckpt_file_path = student_model_config['ckpt']
    end_epoch = start_epoch + train_config['epoch']
    for epoch in range(start_epoch, end_epoch):
        distill_one_epoch(student_model, teacher_model, train_loader, optimizer, criterion,
                          epoch, device, interval, aux_weight)
        avg_valid_loss = validate(student_model, teacher_model, valid_loader, criterion, device)
        if avg_valid_loss < best_avg_loss and main_util.is_main_process():
            best_avg_loss = avg_valid_loss
            save_ckpt(student_model, epoch, best_avg_loss, ckpt_file_path, teacher_model_type)
        scheduler.step()

    del teacher_model
    del student_model


def test(mimic_model, org_model, test_loader, device):
    print('Testing..')
    mimic_model.eval()
    org_model.eval()
    mimic_correct_count = 0
    mimic_test_loss = 0
    org_correct_count = 0
    org_test_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            sub_correct_count, sub_test_loss = predict(inputs, targets, mimic_model)
            mimic_correct_count += sub_correct_count
            mimic_test_loss += sub_test_loss
            sub_correct_count, sub_test_loss = predict(inputs, targets, org_model)
            org_correct_count += sub_correct_count
            org_test_loss += sub_test_loss

    mimic_acc = 100.0 * mimic_correct_count / total
    print('[Mimic]\t\tAverage Loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        mimic_test_loss / total, mimic_correct_count, total, mimic_acc))
    org_acc = 100.0 * org_correct_count / total
    print('[Original]\tAverage Loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        org_test_loss / total, org_correct_count, total, org_acc))
    return mimic_acc, org_acc


def run(args):
    distributed, device_ids = main_util.init_distributed_mode(args.world_size, args.dist_url)
    device = torch.device(args.device)
    if torch.cuda.is_available():
        cudnn.benchmark = True

    config = yaml_util.load_yaml_file(args.config)
    dataset_config = config['dataset']
    input_shape = config['input_shape']
    train_config = config['train']
    train_loader, valid_loader, test_loader =\
        general_util.get_data_loaders(dataset_config, batch_size=train_config['batch_size'],
                                      reshape_size=input_shape[1:3], jpeg_quality=-1, distributed=distributed)
    teacher_model_config = config['teacher_model']
    if not args.test_only:
        distill(train_loader, valid_loader, input_shape, args.aux, config, device, distributed, device_ids)

    org_model, teacher_model_type = mimic_util.get_org_model(teacher_model_config, device)
    mimic_model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device)
    test(mimic_model, org_model, test_loader, device)
    file_util.save_pickle(mimic_model, config['mimic_model']['ckpt'])


if __name__ == '__main__':
    parser = get_argparser()
    run(parser.parse_args())
