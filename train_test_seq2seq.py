import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def train(args, model, scheduler, train_loader, criterion, optimizer, epoch, wandb_run, log_count):
    model.train()  
    total_loss = 0     
    metrics = [] 
    sparsity_tensor = torch.zeros(len(train_loader), 9).to(args.device)
    for batch_idx, (x, y) in enumerate(train_loader):         
        if (batch_idx % 10 == 0) and (args.device != 'cpu'):
            with torch.cuda.device(args.device): 
                torch.cuda.empty_cache()        
        optimizer.zero_grad() 
        
        output, embedding, sparsity, qkv = model(x, y) 
        
        sparsity_tensor[batch_idx] = sparsity
        n_samples, dim_out, time_steps = output.shape
        # Matching the size of the network output and target
        x = x[:,:,0:time_steps] if not(args.spiking_embedding) else x 
        y = y[:,:,0:time_steps] if not(args.spiking_embedding) else y       
        metrics = cat_metrics(output, y, metrics)   
        # The loss is the sum of a L1 difference between the output and the target, and L2 of the emebedding and of QKV tensor (to increase sparsity) 
        loss = criterion(output, y) 
        embeeding_sparse_loss = sparsity_loss(args)(embedding) if args.binarize_embedding else 0
        qkv_sparse_loss = sparsity_loss(args)(qkv) if args.binarize_qkv else 0        
        backward_loss = sum([loss, embeeding_sparse_loss, qkv_sparse_loss])
        
        backward_loss.backward() 
        optimizer.step()
        total_loss += loss.item()                 
         
        if batch_idx % args.log_interval == 0:
            print(f'\nEpoch {epoch:.0f}\tTrain \t|\t[{batch_idx:.0f}/{len(train_loader):.0f} ({100.*batch_idx/len(train_loader):.0f}%)]\t|\tLoss: {loss.item():.3f}')
            wandb_run.log({"epoch": epoch, "batch_train_loss": loss.item()})
        log_count += len(x)

    mean_loss = total_loss/(batch_idx+1)                      ### only divide by the number of batches: cross entopy loss already averages on the number of elements inside a batch  
    metrics = [np.mean(met) for met in metrics]
    print(f'\nTrain set\t|\tLoss={mean_loss:.4f}')
    print(f'MAE={metrics[0]:.4f}\tMSE={metrics[1]:.4f}\t10°accuracy={metrics[2]:.4f}\t15°accuracy={metrics[3]:.4f}\n')
    return metrics, log_count, scheduler, torch.mean(sparsity_tensor, dim=0)

class sparsity_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sparsity_loss_threshold = args.sparsity_loss_threshold
        self.sparsity_loss_coeff = args.sparsity_loss_coeff
    def forward(self, x):
        mean_number_of_elements = nn.MSELoss()(x, torch.zeros_like(x).to(x.device))
        if mean_number_of_elements.item()>self.sparsity_loss_threshold:
            loss = 0
        else:
            loss = 1/2 * self.sparsity_loss_coeff * mean_number_of_elements              
        return loss

def test(args, model, test_loader, criterion, epoch, wandb_run = None, log_count = 0):
    model.eval()   
    total_loss = 0       
    # Tensor created to store the time trace input, target, results
    total_input = torch.empty(16, 1).to(args.device) 
    total_output = torch.empty(5, 1).to(args.device) 
    total_y = torch.empty(5, 1).to(args.device) 
    metrics = [] 
    sparsity_tensor = torch.zeros(len(test_loader), 9).to(args.device)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            output, sparsity = model(x)
            sparsity_tensor[batch_idx] = sparsity
            n_samples, dim_out, time_steps = output.shape
            x = x[:,:,0:time_steps] if not(args.spiking_embedding) else x       
            y = y[:,:,0:time_steps] if not(args.spiking_embedding) else y       
            metrics = cat_metrics(output, y, metrics) 
            loss = criterion(output, y)
            total_loss += loss.item()
            total_input = torch.cat((total_input, x[0]), dim=1)
            total_output = torch.cat((total_output, output[0]), dim=1)
            total_y = torch.cat((total_y, y[0]), dim=1)
            if batch_idx % args.log_interval == 0:
                print(f'\nEpoch {epoch:.0f}\tTest \t|\t[{batch_idx:.0f}/{len(test_loader):.0f} ({100.*batch_idx/len(test_loader):.0f}%)]\t|\tLoss: {loss.item():.3f}')
                wandb_run.log({"epoch": epoch, "batch_test_loss": loss})   
        mean_loss = total_loss/(batch_idx+1)                      ### only divide by the number of batches: cross entopy loss already averages on the number of elements inside a batch
        metrics = [np.mean(met) for met in metrics]
        print(f'\nTest set\t|\tLoss={mean_loss:.4f}')
        print(f'MAE={metrics[0]:.4f}\tMSE={metrics[1]:.4f}\t10°accuracy={metrics[2]:.4f}\t15°accuracy={metrics[3]:.4f}\n')
        fig = plot_DOA_and_results(args.group, args.dataset, total_input, total_y, total_output) 
        
    return metrics, fig, torch.mean(sparsity_tensor, dim=0)

# from torchmetrics import R2Score
def cat_metrics(output, y, metrics):    
    with torch.no_grad():
        if len(metrics)==0:
            mae = torch.mean(torch.abs(output-y), dim=1)                                 # mae
            metrics += [np.array(torch.mean(mae).item())]                # mae   
            metrics += [np.array(nn.MSELoss()(output, y).item())] #.detach().cpu().numpy()        # mse
            metrics += [np.array(torch.mean((mae<10).float()).item())]   # accuracy 10°
            metrics += [np.array(torch.mean((mae<15).float()).item())]  # accuracy 15°            
            samples, dim_out, time = y.shape
            r2 = np.zeros((samples, dim_out))
            for i in range(samples):
                for j in range(dim_out):
                    r2[i,j] = r2_score(y[i,j].detach().cpu().numpy(), output[i,j].detach().cpu().numpy())
            metrics += [np.array(np.mean(np.mean(r2)))]                 # R²
        else:
            mae = torch.mean(torch.abs(output-y), dim=1)                                 # mae
            metrics[0] = np.hstack((metrics[0], np.array(torch.mean(mae).item())))                # mae   
            metrics[1] = np.hstack((metrics[1], np.array(nn.MSELoss()(output, y).item()))) #.detach().cpu().numpy()        # mse
            metrics[2] = np.hstack((metrics[2], np.array(torch.mean((mae<10).float()).item())))   # accuracy 10°
            metrics[3] = np.hstack((metrics[3], np.array(torch.mean((mae<15).float()).item())))  # accuracy 15°            
            samples, dim_out, time = y.shape
            r2 = np.zeros((samples, dim_out))
            for i in range(samples):
                for j in range(dim_out):
                    r2[i,j] = r2_score(y[i,j].detach().cpu().numpy(), output[i,j].detach().cpu().numpy())
            metrics[4] = np.hstack((metrics[4], np.array(np.mean(np.mean(r2)))))                 # R²
    return metrics

def plot_DOA_and_results(group, dataset, x, y, output, last_time_step=2000000):
    plt.close('all')
    if dataset=='ninapro8':
        sampling_rate = 2000
    elif dataset=='ninapro5':
        sampling_rate = 200
    moving_average_window_size = int(sampling_rate/20)
    # we only choose one sample in the batch
    x = moving_average(x[:,1:last_time_step], moving_average_window_size)
    y = moving_average(y[:,1:last_time_step], moving_average_window_size)
    output = moving_average(output[:,1:last_time_step], moving_average_window_size)    
    x = x.transpose(0,1).cpu().numpy()
    y = y.cpu().numpy()
    output = output.detach().cpu().numpy()
    
    num_DOA, n_time_steps = y.shape
    time = np.arange(0, n_time_steps)/sampling_rate*moving_average_window_size
    fig, ax = plt.subplots(6,1)
    cm_to_inch = 1/2.54
    fig.set_figheight(22*cm_to_inch)
    fig.set_figwidth(12*cm_to_inch)
    for i in range(num_DOA):
        ax[i].plot(time, y[i], 'k')
        ax[i].plot(time, output[i], 'r')
        ax[i].set_ylabel('DOA (°)')
        ax[i].set_xlabel('Time (s)')
        ax[i].legend(["Ground truth", "Estimate"], loc="upper right")
        ## save to txt file ###
        np.savetxt(f'{group}_target_angle_{i}.txt', y[i])
        np.savetxt(f'{group}_pred_angle_{i}.txt', output[i])
    ax[-1].plot(time, x)
    ax[-1].set_ylabel('Normalized EMG')
    ax[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    return fig

def moving_average(x, window_size):
    channels, time = x.shape
    new_time = time//window_size
    x = x[:,:new_time*window_size]
    x = x.reshape(channels, new_time, window_size)
    return torch.mean(x, dim=-1)