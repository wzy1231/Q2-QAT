import argparse
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER, TQDM_BAR_FORMAT, check_dataset, check_file, check_git_info,
    check_git_status, check_img_size, check_requirements, check_suffix, check_yaml,
    colorstr, get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
    labels_to_image_weights, methods, one_cycle, print_args, strip_optimizer,
    yaml_save
)
from utils.loggers import Loggers
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.torch_utils import (
    EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
    smart_resume, torch_distributed_zero_first
)


# Project-specific quantizer conversion; provide this module with the repository.
from LTQ.ltq_models1 import convert_to_ltq

QADA_FEATURE_LAYERS = (
    'model.2.cv3.act',
    'model.6.cv3.act',
    'model.13.cv3.act',
    'model.20.cv3.act',
    'model.23.cv3.act',
)


class FeatureStorage:
    """Store intermediate features from the frozen teacher and quantized student."""
    def __init__(self):
        self.pretrained = {}
        self.quant = {}


class FeatureHook:
    """Collect layer outputs while retaining student gradients."""
    def __init__(self, layer_name, storage, teacher=False):
        self.layer_name = layer_name
        self.storage = storage
        self.teacher = teacher

    def __call__(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected tensor output from {self.layer_name}")
        target = self.storage.pretrained if self.teacher else self.storage.quant
        target[self.layer_name] = output.detach() if self.teacher else output


class QADA(nn.Module):
    """Per-channel spatial attention alignment using Jensen-Shannon divergence."""
    def __init__(self, kappa=1e-5, gamma=1.0, eps=1e-12):
        super().__init__()
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def attention(self, x, delta):
        if x.ndim != 4 or x.shape != delta.shape:
            raise ValueError("Expected matching [B, C, H, W] features and distortion")
        mu = x.mean(dim=(-2, -1), keepdim=True)
        sigma = (x - mu).square().mean(dim=(-2, -1), keepdim=True).sqrt()
        z = (x - mu).abs() / (sigma + self.kappa)
        r = delta.abs() / (sigma + self.kappa)
        score = torch.log1p(z.square()) + self.gamma * torch.log1p(r.square())
        return torch.sigmoid(score)

    def forward(self, student, teacher):
        if teacher.shape != student.shape:
            raise ValueError("Teacher and student features must have matching shapes")
        distortion = (teacher.detach() - student).abs()
        with torch.no_grad():
            p = self.attention(teacher.detach(), distortion.detach())
            p = p / p.sum(dim=(-2, -1), keepdim=True)
        q = self.attention(student, distortion)
        q = q / q.sum(dim=(-2, -1), keepdim=True)
        m = 0.5 * (p + q)
        js = 0.5 * (
            (p * (p.clamp_min(self.eps).log() - m.clamp_min(self.eps).log())).sum(dim=(-2, -1))
            + (q * (q.clamp_min(self.eps).log() - m.clamp_min(self.eps).log())).sum(dim=(-2, -1))
        )
        return js.mean()


def get_base_model(model):
    return model.module if hasattr(model, 'module') else model


LOCAL_RANK = int(os.getenv('LOCAL_RANK', '-1'))
RANK = int(os.getenv('RANK', '-1'))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))
GIT_INFO = check_git_info()


def register_feature_hooks(teacher, student, layer_names, storage):
    handles = []
    for name in layer_names:
        for net, is_teacher in ((teacher, True), (student, False)):
            try:
                layer = get_base_model(net).get_submodule(name)
            except AttributeError as exc:
                for handle in handles:
                    handle.remove()
                raise ValueError(f"Feature layer not found: {name}") from exc
            handles.append(layer.register_forward_hook(FeatureHook(name, storage, is_teacher)))
    return handles


def remove_feature_hooks(handles):
    for handle in handles:
        handle.remove()
    handles.clear()


def train(hyp, opt, device, callbacks):
    save_dir = Path(opt.save_dir)
    epochs = opt.epochs
    batch_size = opt.batch_size
    weights = opt.weights
    single_cls = opt.single_cls
    evolve = False
    data = opt.data
    cfg = opt.cfg
    resume = opt.resume
    noval = opt.noval
    nosave = opt.nosave
    workers = opt.workers
    freeze = opt.freeze

    callbacks.run('on_pretrain_routine_start')

    w = save_dir / 'weights'
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    if isinstance(hyp, str):
        import yaml
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)

    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
        data_dict = loggers.remote_dataset
        if resume:
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    plots = not evolve and not opt.noplots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)

    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)

    train_path, val_path = data_dict['train'], data_dict['val']
    nc = int(data_dict["nc"])
    names = data_dict["names"]
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    if not pretrained:
        raise ValueError('Provide --weights with a pretrained full-precision checkpoint')

    # Capture intermediate representations for Q-ADA.
    feature_storage = FeatureStorage()

    # Load the frozen full-precision teacher.
    pretrain_weights = opt.teacher_weights
    if not pretrain_weights:
        raise ValueError('Provide --teacher-weights for Q-ADA supervision')
    pretrained_ckpt = torch.load(pretrain_weights, map_location='cpu', weights_only=False)
    pretrained_model = Model(pretrained_ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    pretrained_model.load_state_dict(pretrained_ckpt['model'].float().state_dict(), strict=False)
    pretrained_model.eval()
    for param in pretrained_model.parameters():
        param.requires_grad = False

    qada = QADA(kappa=opt.qada_kappa, gamma=opt.qada_gamma).to(device)
    # Feature levels used for attention alignment.
    cam_layer_names = list(QADA_FEATURE_LAYERS)

    # Initialize the quantized student.
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location='cpu', weights_only=False)
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        model = convert_to_ltq(deepcopy(model), opt.wbits, opt.abits)

        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        missingkeys, unexpected_keys = model.load_state_dict(csd, strict=False)
        LOGGER.info(f'missingkeys: {missingkeys}')
        LOGGER.info(f'unexpected_keys: {unexpected_keys}')
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')
    else:
        raise ValueError("A pretrained checkpoint is required")

    LOGGER.info(f'model {model}')

    hook_handles = register_feature_hooks(pretrained_model, model, cam_layer_names, feature_storage)
    LOGGER.info(f'Q-ADA distillation layers: {cam_layer_names}')

    amp = False
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    if RANK == -1 and batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({'batch_size': batch_size})

    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    best_fitness, start_epoch = 0.0, 0
    if pretrained and resume:
        best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    if cuda and LOCAL_RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('Using nn.DataParallel because LOCAL_RANK=-1 and multiple GPUs found.')
        model = nn.DataParallel(model)

    if opt.sync_bn and cuda and LOCAL_RANK != -1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == 'val' else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr('train: '),
        shuffle=True,
        seed=opt.seed
    )
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}.'

    if RANK in {-1, 0}:
        val_loader, _ = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            prefix=colorstr('val: ')
        )
        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.float()
        callbacks.run('on_pretrain_routine_end', labels, names)

    if cuda and LOCAL_RANK != -1:
        model = smart_DDP(model)

    net = de_parallel(model)
    if hasattr(net.model[-1], 'nl'):
        nl = net.model[-1].nl
    else:
        nl = 3
    hyp['box'] *= 3 / nl
    hyp['cls'] *= nc / 80 * 3 / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    t0 = time.time()
    nb = len(train_loader)
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    last_opt_step = -1
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    if opt.noval:
        LOGGER.warning("Early stopping requires validation; enabling validation for every epoch.")
        noval = False
    stopper = EarlyStopping(patience=opt.patience)
    stop = False

    compute_loss_fn = ComputeLoss(model)

    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val')
    LOGGER.info(f'Starting training for {epochs} epochs...')

    mloss = torch.zeros(4, device=device)

    for epoch in range(start_epoch, epochs):
        callbacks.run('on_train_epoch_start')
        model.train()

        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)

        if LOCAL_RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        if RANK in {-1, 0}:
            LOGGER.info(('\n' + '%11s' * 8) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss',
                                               'qada_loss', 'Instances', 'Size'))

        pbar = enumerate(train_loader)
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)
                loss, loss_items = compute_loss_fn(pred, targets.to(device))

                qada_loss = torch.zeros((), device=device)
                if epoch < opt.qada_epochs and opt.qada_weight > 0:
                    with torch.no_grad():
                        _ = pretrained_model(imgs)
                    for layer_name in cam_layer_names:
                        qf = feature_storage.quant.get(layer_name)
                        pf = feature_storage.pretrained.get(layer_name)
                        if qf is None or pf is None:
                            raise RuntimeError(f"Missing Q-ADA features from {layer_name}")
                        qada_loss = qada_loss + qada(qf.float(), pf.float())
                    qada_loss = qada_loss / len(cam_layer_names)
                    loss = loss + opt.qada_weight * qada_loss
                feature_storage.pretrained.clear()
                feature_storage.quant.clear()

            if LOCAL_RANK != -1:
                loss *= WORLD_SIZE
            if opt.quad:
                loss *= 4.0

            scaler.scale(loss).backward()
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            if RANK in {-1, 0}:
                loss_items_kl = torch.cat([loss_items, (opt.qada_weight * qada_loss).detach().unsqueeze(0)])
                mloss = (mloss * i + loss_items_kl) / (i + 1)

                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.8g}G'
                pbar.set_description(('%11s' * 2 + '%11.4g' * 6) %
                                     (f'{epoch}/{epochs - 1}',
                                      mem,
                                      mloss[0],
                                      mloss[1],
                                      mloss[2],
                                      mloss[3],
                                      targets.shape[0],
                                      imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return

        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        if RANK in {-1, 0}:
            callbacks.run('on_train_epoch_end', epoch=epoch)
            if ema:
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop

            from val import run as val_run
            validated = (not noval) or final_epoch
            if validated:
                results, maps, _ = val_run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema if ema else model,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss_fn
                )
            improved = False
            if validated:
                fi = float(fitness(np.array(results).reshape(1, -1)))
                stop = stopper(epoch=epoch, fitness=fi)
                if fi > best_fitness:
                    best_fitness = fi
                    improved = True
            else:
                fi = best_fitness

            if (not nosave) or final_epoch:
                # Save the training state.
                remove_feature_hooks(hook_handles)
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)),
                    'ema': deepcopy(ema.ema) if ema else None,
                    'updates': ema.updates if ema else None,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,
                    'date': datetime.now().isoformat(),
                }
                torch.save(ckpt, last)
                if improved:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)
                hook_handles = register_feature_hooks(pretrained_model, get_base_model(model), cam_layer_names, feature_storage)

        if LOCAL_RANK != -1:
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            stop = broadcast_list[0]
        if stop and RANK in {-1, 0}:
            LOGGER.info(f"Early stopping at epoch {epoch + 1}: validation fitness did not improve for {opt.patience} epochs.")
            break

    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)
                if f == best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val_run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device),
                        iou_thres=0.65 if is_coco else 0.60,
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss_fn
                    )
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)
        callbacks.run('on_train_end', last, best, epoch, results)

    remove_feature_hooks(hook_handles)
    torch.cuda.empty_cache()
    return 

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=not known, default='')
    parser.add_argument('--teacher-weights', type=str, default='')
    parser.add_argument('--wbits', type=int, default=4)
    parser.add_argument('--abits', type=int, default=4)
    parser.add_argument('--qada-weight', type=float, default=0.001)
    parser.add_argument('--qada-kappa', type=float, default=1e-5)
    parser.add_argument('--qada-gamma', type=float, default=1.0)
    parser.add_argument('--qada-epochs', type=int, default=100)
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml')
    parser.add_argument('--data', type=str, default='data/VOC.yaml')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.VOC.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640)
    parser.add_argument('--rect', action='store_true')
    parser.add_argument('--resume', nargs='?', const=True, default=False)
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--noval', action='store_true')
    parser.add_argument('--noautoanchor', action='store_true')
    parser.add_argument('--noplots', action='store_true')
    parser.add_argument('--cache', type=str, nargs='?', const='ram')
    parser.add_argument('--image-weights', action='store_true')
    parser.add_argument('--device', default='')
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--single-cls', action='store_true')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD')
    parser.add_argument('--sync-bn', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--project', default=ROOT / 'runs/train')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--quad', action='store_true')
    parser.add_argument('--cos-lr', action='store_true')
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--freeze', nargs='+', type=int, default=[0])
    parser.add_argument('--save-period', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1)

    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt, callbacks=Callbacks()):
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / 'requirements.txt')

    if opt.resume:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'
        opt_data = opt.data
        if opt_yaml.is_file():
            import yaml
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu', weights_only=False)['opt']
        teacher_weights = opt.teacher_weights
        opt = argparse.Namespace(**d)
        opt.teacher_weights = teacher_weights or getattr(opt, 'teacher_weights', '')
        opt.cfg, opt.weights, opt.resume = '', str(last), True
        if is_url(opt_data):
            opt.data = check_file(opt_data)
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    if LOCAL_RANK != -1:
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(
            backend='nccl' if dist.is_nccl_available() else 'gloo',
            timeout=timedelta(seconds=10800)
        )
    else:
        device = select_device(opt.device, batch_size=opt.batch_size)

    train(opt.hyp, opt, device, callbacks)


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt) 