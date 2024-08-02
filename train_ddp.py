import math
import os
import os.path as osp
import torch
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms

from data.multiple_dataset import MultipleDatasets
from glob import glob
# from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data import DataLoader
# from lr_scheduler import CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from utils.options import parse_options, copy_opt_file
from utils.logger import ColorLogger, init_tb_logger
from utils.misc import mkdir_and_rename
from utils.timer import Timer

# from models_newer_gttopk.blurhandnet_newpos import BlurHandNet
# from models_newer_gttopk.blurhandnet import BlurHandNet
# from model_current.blurhandnet import BlurHandNet
# from models_combine.blurhandnet import BlurHandNet
# from models_token1.blurhandnet3 import BlurHandNet
# from models_deformer.blurhandnet import BlurHandNet
from models_two.blurhandnet import BlurHandNet


def main():
    # load opt and args from yaml
    # os.environ["TORCH_CPP_LOG_LEVEL"] = "DEBUG"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    local_rank = int(os.environ["LOCAL_RANK"])
    opt, args = parse_options()
    args.local_rank = local_rank
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    dist.barrier()

    # dynamic dataset import   
    dataset_list = opt['dataset_list']
    for _, opt_data in dataset_list.items():
        dataset_name = opt_data['name']
        exec(f'from data.{dataset_name} import {dataset_name}')
    globals().update(locals())
    
    trainer = Trainer(opt, args)
    trainer._make_batch_generator()
    trainer._prepare_training()

    if dist.get_rank()==0:trainer.logger.info("begin training.")
    # training
    apply_grad_clip = trainer.opt['train']['optim']['apply_grad_clip']
    grad_norm = trainer.opt['train']['optim']['grad_clip_norm']
    end_epoch = trainer.end_epoch
    for epoch in range(trainer.start_epoch, (end_epoch+1)):
        trainer.sampler.set_epoch(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            for optimizer in trainer.optimizers:
                optimizer.zero_grad()
            loss, rm_info = trainer.model(inputs, targets, meta_info, trainer.train_mode)
            loss = {k:loss[k].mean() for k in loss}

            # tensorboard logging
            tot_iter = (epoch-1) * trainer.itr_per_epoch + itr
                
            # backward
            dist.barrier()
            sum(loss[k] for k in loss).backward()
            if apply_grad_clip:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), grad_norm)
            if dist.get_rank()==0:
                trainer.log_grad(tot_iter)

            trainer.step()

            trainer.gpu_timer.toc()
            trainer.print_timer.tic()

            if dist.get_rank()==0:
                for k,v in loss.items():
                    trainer.tb_logger.add_scalar(k, v, tot_iter)
                trainer.tb_logger.add_scalars('lr/', {'lr%d'%i: scheduler.get_last_lr()[0] for i, scheduler in enumerate(trainer.schedulers)}, tot_iter)
                # print("tb log done")
                
                screen_loss_info = ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
                screen_basic_info = [
                    'Epoch %d/%d itr %d/%d: tot %d:' % (epoch, end_epoch, itr, trainer.itr_per_epoch, tot_iter),
                    'lr: %g' % (trainer.schedulers[0].get_last_lr()[0]),
                    'speed: %.2f(c%.2fs p%.2fs r%.2fs)s/itr' % (
                        trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.print_timer.average_time, trainer.read_timer.average_time),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                    ]
                
                screen = screen_basic_info + screen_loss_info
                
                trainer.logger.info('\n\t'.join(screen))
            trainer.print_timer.toc()
            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
        if trainer.local_rank==0 and (epoch%20==0 or epoch==end_epoch):
            state = {'epoch': epoch, 'network': trainer.model.state_dict()}
            state.update({'optimizer%d'%i: opt.state_dict()for i, opt in enumerate(trainer.optimizers)})
            trainer.save_state(state, epoch)
            
    dist.destroy_process_group()


class Trainer():
    def __init__(self, opt, args):
        self.opt = opt
        self.train_mode = 'train-{}'.format(self.opt['train_mode'])
        self.suffix = ('_' + opt['rm_suffix']) if self.train_mode == 'train-rew' else ('')
        self.exp_dir = osp.join('experiments', opt['name']+self.suffix)
        self.tb_dir = osp.join('tb_logger', opt['name']+self.suffix)
        self.cur_epoch = 0
        self.end_epoch = opt['train']['gen_end_epoch'] if self.train_mode == 'train-gen' else opt['train']['rew_end_epoch']
        self.world_size = dist.get_world_size()
        self.local_rank = args.local_rank
        self.device = torch.device("cuda", self.local_rank)
        
        # timers
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()
        self.print_timer = Timer()
        
        if dist.get_rank() == 0:
            # directories
            if not opt.get('continue_train', False):
                mkdir_and_rename(osp.join(self.exp_dir))
                mkdir_and_rename(osp.join(self.tb_dir))
            
            # logger
            self.logger = ColorLogger(self.exp_dir, log_name='train_logs.txt')
            self.tb_logger = init_tb_logger(self.tb_dir)
            
            # copy the yml file to the experiment root
            copy_opt_file(args.opt, self.exp_dir)
            
    def _make_batch_generator(self):
        # data loader and construct batch generator
        trainset3d_loader = []
        trainset2d_loader = []
        
        dataset_list = self.opt['dataset_list']
        for _, opt_data in dataset_list.items():
            dataset_name = opt_data['name']
            if self.local_rank == 0: self.logger.info(f"Creating dataset ... [{dataset_name}]")
            if opt_data.get('is_3d', False):
                trainset3d_loader.append(eval(dataset_name)(self.opt, opt_data, transforms.ToTensor(), "train"))
            else:
                trainset2d_loader.append(eval(dataset_name)(self.opt, opt_data, transforms.ToTensor(), "train"))
        
        # dataloader for validation
        valid_loader_num = 0
        if len(trainset3d_loader) > 0:
            trainset3d_loader = [MultipleDatasets(trainset3d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset3d_loader = []
        if len(trainset2d_loader) > 0:
            trainset2d_loader = [MultipleDatasets(trainset2d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset2d_loader = []
        if valid_loader_num > 1:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=True)
        else:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=False)

        num_gpus = self.opt['num_gpus']
        num_threads = self.opt['num_threads']
        train_batch_size = self.opt['train']['batch_size']
        self.itr_per_epoch = math.ceil(len(trainset_loader) / train_batch_size / self.world_size) 

        self.sampler = torch.utils.data.DistributedSampler(trainset_loader)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=train_batch_size,
                                          num_workers=num_threads, sampler=self.sampler, 
                                          pin_memory=True, drop_last=True)

    def _prepare_training(self):
        # prepare network and optimizer
        if self.local_rank == 0:self.logger.info("Creating network and optimizer ... [seed {}]".format(self.opt['manual_seed']))
        model = BlurHandNet(self.opt)
        model = model.to(self.device)
        model = DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.gen_model = model

        optimizers, schedulers = self.get_optimizer(model)
        
        # if continue training, load the most recent training state
        if self.local_rank == 0 and self.opt.get('continue_train', False):
            start_epoch, model, optimizers = self.continue_train(model, optimizers)
        else:
            start_epoch = 1
        model.train()

        self.start_epoch = start_epoch
        self.end_epoch = self.opt['train']['gen_end_epoch']
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers

    def save_state(self, state, epoch):
        os.makedirs(osp.join(self.exp_dir, 'training_states'), exist_ok=True)
        file_path = osp.join(self.exp_dir, 'training_states', 'epoch_{:02d}.pth.tar'.format(epoch))

        # do not save human model layer weights
        dump_key = []
        for k in state['network'].keys():
            if 'smpl_layer' in k or 'mano_layer' in k or 'flame_layer' in k:
                dump_key.append(k)
        for k in dump_key:
            state['network'].pop(k, None)

        torch.save(state, file_path)
        
        if self.local_rank == 0:self.logger.info("Saving training states into {}".format(file_path))

    def continue_train(self, model, optimizers):
        states_list = glob(osp.join(self.exp_dir, 'training_states', '*.pth.tar'))
        
        # find the most recent training state
        cur_epoch = max([int(file_name[file_name.find('epoch_') + 6:file_name.find('.pth.tar')])
                         for file_name in states_list])
        ckpt_path = osp.join(self.exp_dir, 'training_states', 'epoch_' + '{:02d}'.format(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path) 
        
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'], strict=False, map_location=torch.device("cuda", self.local_rank))
        for i, optimizer in enumerate(optimizers):
            optimizer.load_state_dict(ckpt['optimizer%d'%i], map_location=torch.device("cuda", self.local_rank))
        
        if self.local_rank == 0:self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        
        return start_epoch, model, optimizer

    def get_optimizer(self, model: DistributedDataParallel):
        g_params = []
        for module in model.module.gen_modules:
            g_params += list(module.parameters())
        r_params = []
        for module in model.module.rew_modules:
            r_params += list(module.parameters())
        a_params = []
        for module in model.module.agg_modules:
            a_params += list(module.parameters())
        if self.train_mode == 'train-gen':
            for p in r_params + a_params:
                p.requires_grad = False
            optimizers = [torch.optim.Adam(g_params, lr=self.opt['train']['optim']['lr'], weight_decay=self.opt['train']['optim']['weight_decay'])]
        elif self.train_mode == 'train-rew':
            for p in g_params:
                p.requires_grad = False
            optimizers = [torch.optim.Adam(r_params, lr=self.opt['train']['optim']['lr_rm']), ]
                        #   torch.optim.Adam(a_params, lr=self.opt['train']['optim']['lr'])]
        
        if self.train_mode == 'train-gen':
            schedulers = [MultiStepLR(optimizer,
                                    milestones=[e*self.itr_per_epoch for e in self.opt['train']['optim']['lr_dec_epoch']],
                                    gamma= 1/self.opt['train']['optim']['lr_dec_factor']) for optimizer in optimizers]
            # [CosineAnnealingWarmupRestarts(optimizer, 
            #         first_cycle_steps = self.itr_per_epoch*self.opt['train']['gen_end_epoch'],
            #         cycle_mult = 1.,
            #         max_lr = self.opt['train']['optim']['lr'],
            #         min_lr = self.opt['train']['optim']['lr_min'],
            #         warmup_steps = self.itr_per_epoch//2,
            #         gamma = 1.,
            #         last_epoch = -1) for optimizer in optimizers]
        elif self.train_mode == 'train-rew':
            schedulers = [CosineAnnealingWarmupRestarts(optimizers[0], # FlippedCosineAnnealing
                            first_cycle_steps = self.itr_per_epoch*self.opt['train']['rew_end_epoch'],
                            cycle_mult = 1.,
                            max_lr = self.opt['train']['optim']['lr_rm'],
                            min_lr = self.opt['train']['optim']['lr_min'],
                            warmup_steps = self.itr_per_epoch*2,
                            gamma = 1.,
                            last_epoch = -1), ]
                    #     CosineAnnealingWarmupRestarts(optimizers[1], 
                    #         first_cycle_steps = self.itr_per_epoch*self.opt['train']['rew_end_epoch'],
                    #         cycle_mult = 1.,
                    #         max_lr = self.opt['train']['optim']['lr'],
                    #         min_lr = self.opt['train']['optim']['lr_min'],
                    #         warmup_steps = self.itr_per_epoch//2,
                    #         gamma = 1.,
                    #         last_epoch = -1), 
                    # ]
        return optimizers, schedulers
    
    def step(self):
        for optimizer, scheduler in zip(self.optimizers, self.schedulers):
                    optimizer.step()
                    scheduler.step()
    
    def load_model_rm(self, model:DistributedDataParallel):
        exp_dir_origin = self.exp_dir[:self.exp_dir.find(self.suffix)]
        states_list = glob(osp.join(exp_dir_origin, 'training_states', '*.pth.tar'))
        
        # find the most recent training state
        cur_epoch = max([int(file_name[file_name.find('epoch_') + 6:file_name.find('.pth.tar')])
                         for file_name in states_list])
        ckpt_path = osp.join(exp_dir_origin, 'training_states', 'epoch_' + '{:02d}'.format(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path) 
        if self.local_rank == 0:self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        
        temp_info = model.load_state_dict(ckpt['network'], strict=False)
        
        if self.local_rank == 0:
            self.logger.info('Load checkpoint from {}'.format(ckpt_path))
            self.logger.info('weights of {} are missing, it would be randomly initialized.'.format(temp_info.missing_keys))
            self.logger.info('get unexpcted weights {}'.format(temp_info.unexpected_keys))
        # model.module.copy_regressor()
        # model.module.transformer.eval()
        return model
    
    def log_grad(self, tot_iter):
        # trainer.tb_logger.add_histogram('all_error', rm_info['all_error'][:, 0], tot_iter)   # only store one batch
            grad_vit = {}
            grad_nce = {}
            grad_agg = {}
            grad_rm = {}
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    if 'qkv' in name and 'transformer' in name:   # vit
                        grad_vit[name] = torch.norm(p.grad)
                    if 'qkv' in name and 'aggregator' in name:   # vit
                        grad_agg[name] = torch.norm(p.grad)
                    elif 'info_nce' in name and 'weight' in name:    # info nce
                        grad_nce[name] = torch.norm(p.grad)
                    elif 'ca.in_proj_weight' in name or 'sa.in_proj_weight' in name:    # rm
                        grad_rm[name] = torch.norm(p.grad)

            self.tb_logger.add_scalars(f"grad/vit", grad_vit, tot_iter)
            self.tb_logger.add_scalars(f"grad/nce", grad_nce, tot_iter)
            self.tb_logger.add_scalars(f"grad/rm", grad_rm, tot_iter)
            self.tb_logger.add_scalars(f"grad/agg", grad_agg, tot_iter)


if __name__ == "__main__":
    main()
