import torch
import numpy as np
from train_test_seq2seq import train, test
from tqdm import tqdm
import wandb
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import random
import continuous_model
from continuous_model import torch_init_LSUV
from config_continuous_attention import return_args
from utils import del_previous_plots, transfer_weights

device='cuda:0'
subject=0
lr=1e-3
bs=64
datAug=64
wd=0
ws=2000 
ks=7
stride=5
padding=int((ks-stride)/2)
ntokens=int((ws-ks+2*padding)/stride+1)
stored_vector_size=150
name = f'transformer_sub_{subject:02d}_continuous_transfo_conv_embed_lr={lr:.0e}_datAug={datAug}_bs={bs}_wd={wd:.0e}_ws={ws}_np={ntokens}_svs={stored_vector_size}_ks={ks}_s={stride}_p={padding} (0)'
group = "OST"

def main(**kwargs): 
    # try:
    torch.autograd.set_detect_anomaly(True)
    # update parameters from config file and import other parameters 
    args = return_args() if not('outer_parameters' in kwargs) else return_args(**kwargs['outer_parameters'])    
    # create a wandb session 
    wandb_run = wandb.init(project=args["project"],
                            name=args["name"],
                            group=args["group"],
                            notes="",
                            config=args,
                            mode="disabled") if not('wandb_run' in kwargs) else kwargs['wandb_run']
    # change args dictonary to a wandb config object and allow wandb to track it
    args = wandb_run.config
    
    # manually change the seed
    random.seed(args.seed) if isinstance(args.seed, int) else None
    torch.manual_seed(args.seed) if isinstance(args.seed, int) else None

    # load dataset on RAM and make appropriate transformations 
    from data_loader_generator_ninapro8 import _load_ninapro8_emg_windows as _load_data
    from data_loader_generator_ninapro8 import make_loader
    data_train, start_idx_train = _load_data(args, times=[0,1])
    data_test, start_idx_test = _load_data(args, times=[2])
    
    # define model, optimizer, scheduler and loss function
    model_test = getattr(continuous_model, args.Net)(args, args.dim_in, args.dim_out).to(args.device)
    model_train = getattr(continuous_model, args.trainingNet)(args, args.dim_in, args.dim_out).to(args.device)
    model_to_load = torch.load("./saved_models/"+args.pre_trained_model_name+".pt", map_location=args.device) if args.pre_trained_model_name is not None else None
    model_train.load_state_dict(model_to_load, strict=False) if args.pre_trained_model_name is not None else None
    transfer_weights(model_train, model_test)
    
    # Quantization
    # model_test= torch.quantization.quantize_dynamic(model_test.to(torch.device('cpu')), dtype=torch.qint8)
    # model_train = torch.quantization.quantize_dynamic(model_train.to(torch.device('cpu')), dtype=torch.qint8)    
    ##############
    
    optimizer = getattr(optim, args.optimizer)(model_train.parameters(), lr=args.lr, weight_decay=args.weight_decay) if not('optimizer' in kwargs) else kwargs['optimizer']
    scheduler = getattr(lr_scheduler, args.scheduler)(optimizer, **args.scheduler_args) if not('scheduler' in kwargs) else kwargs['scheduler'] #(optimizer, **args.scheduler_args) 
    # resuming the scheduler to a specific step is useful when resuming a previous job
    resume_scheduler(scheduler, args.previous_job_epoch)
    criterion = getattr(nn, args.loss_fn)()    
    
    # make wandb watch the model
    wandb_run.watch(model_test, criterion=criterion, log="all", log_freq=1000//args.batch_size)

    # training and testing model
    log_count = 0    
    for epoch in tqdm(range(args.epochs-args.previous_job_epoch)): 
        torch.cuda.empty_cache() 
        epoch += args.previous_job_epoch
        print(f'\n____________________________________\nEpoch: {epoch:.0f}\n')
        
        ## Init testing ###        
        test_loader = make_loader(args, data_test, start_idx_test, train=False, specific_subject=None)
        # Initiate model           
        model_test.eval()         
        x, y = next(iter(test_loader))
        if epoch==0 and args.pre_trained_model_name is None: 
            torch_init_LSUV(model_test, x, y=y, tgt_mu=-0.75) 
        model_test.init_epoch(x)
        metrics_test, fig_test, sparsity = test(args, model_test, test_loader, criterion, epoch, wandb_run)
        
        ### init training ###                       
        train_loader = make_loader(args, data_train, start_idx_train, train=True, specific_subject=None)   
        # Initiate model                 
        if epoch==0 and args.pre_trained_model_name is None: 
            model_train.train()
            x, y = next(iter(train_loader))
            torch_init_LSUV(model_train, x, y=y, tgt_mu=-0.75)        

            
        metrics_train, log_count, scheduler, fig_train, sparsity_train = train(args, model_train, scheduler, train_loader, criterion, optimizer, epoch, wandb_run, log_count)   
        transfer_weights(model_train, model_test)
    
        scheduler.step() 

        wandb_run.log({"epoch": epoch,
                    "MAE (degrees) train": metrics_train[0],
                        "MSE (degrees²) train": metrics_train[1],
                        "10° - accuracy train": metrics_train[2],
                        "15° - accuracy train": metrics_train[3],
                        "R² score train":       metrics_train[4],
                        "Train results":        fig_train,
                        "MAE (degrees) test":   metrics_test[0],
                        "MSE (degrees²) test":  metrics_test[1],
                        "10° - accuracy test":  metrics_test[2],
                        "15° - accuracy test":  metrics_test[3],
                        "R² score test":        metrics_test[4],
                        "Test results":         fig_test,
                        "QKV sparsity train":   sparsity_train[0],
                        "Attention sparsity train (before sum on Dembedding)":sparsity_train[1],
                        "Attention sparsity train":   sparsity_train[2],
                        "Mean attention train": sparsity_train[3],
                        "Weighted average sparsity train (before sum on stored tokens)":   sparsity_train[4],
                        "Weighted average sparsity train":sparsity_train[5],
                        "Mean weighted average train": sparsity_train[6],
                        "QKV sparsity":         sparsity[0],
                        "Attention sparsity (before sum on Dembedding)":sparsity[1],
                        "Attention sparsity":   sparsity[2],
                        "Mean attention": sparsity[3],
                        "Weighted average sparsity (before sum on stored tokens)":   sparsity[4],
                        "Weighted average sparsity":sparsity[5],
                        "Mean weighted average":sparsity[6],
                        })        
        del_previous_plots(epoch, wandb_run)
        if args.save_model:
            torch.save(model_train.state_dict(), "./saved_models/"+args.save_model_name+".pt")
    wandb_run.finish()
    return True

def resume_scheduler(scheduler, previous_job_epoch):
    [scheduler.step() for _ in range(previous_job_epoch)]
    return

if __name__ == '__main__':   
    kwargs = dict(outer_parameters=dict(
                                        project="sEMG_DOA",
                                        name=name,
                                        group=group,
                                        dataset='ninapro8',
                                        device=device,   
                                        subjects=subject,           
                                        Net='ContinuousTransformer',  
                                        log_interval=40,
                                        epochs=1,                                 
                                        
                                        loss_fn="L1Loss",
                                        window_size=ws,
                                        sliding_size=ws,
                                        stored_vector_size=stored_vector_size,
                                        test_batch_size=1,
                                        batch_size=bs,
                                        shuffle_dataset=True,
                                        lr=lr,                    
                                        weight_decay=wd,                    
                                        conv_kernel_size=ks,
                                        conv_stride=stride,
                                        data_augmentation_factor=datAug,
                                        
                                        save_model=True,
                                        save_model_name=name,
                                        )
                )
                
    main(**kwargs)
    

