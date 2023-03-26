# plots 
from matplotlib import rc, rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import torch
import numpy as np
from config_continuous_attention import return_args
import wandb
from data_loader_generator_ninapro8 import _load_ninapro8_emg_windows
import matplotlib.cm as cm
from pytorch_forecasting.utils import autocorrelation as torch_autocorrelate
from numpy import correlate as numpy_autocorrelate
from scipy.signal import correlate as scipy_autocorrelate
from scipy.signal import correlation_lags as scipy_correlation_lags

entity = "nleroux"
project = "sEMG_DOA_regression_start_05_01_23"
out_file_root = "./plots/"

metrics_list = ["MAE (degrees) test", "10° - accuracy test", "15° - accuracy test"]
ylabel_list = ["Mean Absolute Error (°)", "10° - accuracy", "15° - accuracy"]

font = {'size': 10}
rc('font', **font)
# change font
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
# rcParams['text.usetex'] = True
rcParams['mathtext.default'] = 'regular'  # Math subscripts and greek letters non-italic
linewidth = 2
marker_size = 2
centimeters = 1 / 2.54  # centimeters in inches   

def DOA_to_DOF(DOA):
    linear_transformation_matrix = torch.FloatTensor([[0.639, 0.000, 0.000, 0.000, 0.000],
                                                    [0.383, 0.000, 0.000, 0.000, 0.000],
                                                    [0.000, 1.000, 0.000, 0.000, 0.000],
                                                    [-0.639,0.000, 0.000, 0.000, 0.000],
                                                    [0.000, 0.000, 0.400, 0.000, 0.000],
                                                    [0.000, 0.000, 0.600, 0.000, 0.000],
                                                    [0.000, 0.000, 0.000, 0.400, 0.000],
                                                    [0.000, 0.000, 0.000, 0.600, 0.000],
                                                    [0.000, 0.000, 0.000, 0.000, 0.000],
                                                    [0.000, 0.000, 0.000, 0.000,0.1667],
                                                    [0.000, 0.000, 0.000, 0.000,0.3333],
                                                    [0.000, 0.000, 0.000, 0.000, 0.000],
                                                    [0.000, 0.000, 0.000, 0.000,0.1667],
                                                    [0.000, 0.000, 0.000, 0.000,0.3333],
                                                    [0.000, 0.000, 0.000, 0.000, 0.000],
                                                    [0.000, 0.000, 0.000, 0.000, 0.000],
                                                    [-0.19, 0.000, 0.000, 0.000, 0.000],
                                                    [0.000, 0.000, 0.000, 0.000, 0.000],
                                                    [0.000, 0.000, 0.000, 0.000, 0.000]
                                                    ])
    linear_transformation_matrix = linear_transformation_matrix[0:-1].transpose(0,1)
    pinv_linear_transformation_matrix = torch.pinverse(linear_transformation_matrix)
    pinv_linear_transformation_matrix = pinv_linear_transformation_matrix.unsqueeze(0)
    DOA = DOA.unsqueeze(2)
    DOF = pinv_linear_transformation_matrix @ DOA
    return DOF.squeeze().numpy()

def plot_emg(apply=False, start_time=0, end_time=-1, save_plot=False, file_out = "emg_plots/test"):
    if apply:
        plt.close('all')
        _, data_test, _ = _load_ninapro8_emg_windows(args, times=[2])
        signal = data_test[0][0][0]
        SR = 2000
        start_time = int(SR*start_time)
        end_time = int(SR*end_time) if end_time != -1 else -1
        channels, time_steps = signal.shape
        time = np.arange(0,time_steps)/2000
        fig, ax = plt.subplots(1)          
        fig.set_figwidth(6 * centimeters)
        fig.set_figheight(4 * centimeters)
        # fig.set_figwidth(25 * centimeters)
        # fig.set_figheight(7 * centimeters)
        ax.plot(time[start_time:end_time], signal[0][start_time:end_time]*1e+3, '-', c='darkblue', linewidth=0.5) 

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EMG signal (mV)")
        # ax.legend()
        ax.tick_params(axis="x", direction='in')
        ax.tick_params(which="minor", direction='in')    
        ax.tick_params(axis="x", direction='in')
        ax.tick_params(which="minor", direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))  
        fig.tight_layout()
        ax.xaxis.labelpad = 5
        ax.yaxis.labelpad = 5
        if save_plot:    
            for fmt in ['png', 'svg', 'pdf']:
                plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)    
        plt.show()
    return

def plot_autocorrelation(apply=False, start_time=0, end_time=-1, save_plot=False, file_out = "emg_plots/test"):
    if apply:
        plt.close('all')
        _, data_test, _ = _load_ninapro8_emg_windows(args, times=[2])
        signal = data_test[0][0][0].float().numpy()
        # autocorr = autocorrelation(signal[0].float(), dim=0)
        # autocorr = np.correlate(signal[0], signal[0], mode='same')
        autocorr, lags = scipy_autocorrelate(signal[0], signal[0], mode='full'), scipy_correlation_lags(len(signal[0]), len(signal[0]), mode='full')
        SR = 2000
        start_time = int(SR*start_time)
        end_time = int(SR*end_time) if end_time != -1 else -1
        channels, time_steps = signal.shape
        time = np.arange(0,time_steps)/SR
        fig, ax = plt.subplots(1)          
        # fig.set_figwidth(6 * centimeters)
        # fig.set_figheight(4 * centimeters)
        fig.set_figwidth(12 * centimeters)
        fig.set_figheight(8 * centimeters)
        # ax.plot(time[start_time:end_time], autocorr[start_time:end_time], '-', c='darkblue', linewidth=0.5) 
        ax.plot(lags[start_time:end_time]/SR, autocorr[start_time:end_time], '-', c='darkblue', linewidth=0.2) 

        ax.set_xlabel("Lags (s)")
        ax.set_ylabel("Autocorrelation")
        ax.tick_params(axis="x", direction='in')
        ax.tick_params(which="minor", direction='in')    
        ax.tick_params(axis="x", direction='in')
        ax.tick_params(which="minor", direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(0.00025))  
        fig.tight_layout()
        ax.xaxis.labelpad = 5
        ax.yaxis.labelpad = 5
        if save_plot:    
            for fmt in ['png', 'svg', 'pdf']:
                plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)    
        plt.show()
    return

def plot_DOA(apply=False, start_time=0, end_time=-1, save_plot=False, file_out = "emg_plots/test"):
    if apply:
        _, data_test, _ = _load_ninapro8_emg_windows(args, times=[2])
        signal = data_test[0][0][0]
        DOA = data_test[1][0][0]
        SR = 2000
        start_time = int(SR*start_time)
        end_time = int(SR*end_time) if end_time != -1 else -1
        channels, time_steps = signal.shape
        time = np.arange(0,time_steps)/SR
        fig, ax = plt.subplots(5)          
        fig.set_figwidth(10 * centimeters)
        fig.set_figheight(14 * centimeters)
        for i in range(5):
            ax[i].plot(time[start_time:end_time], DOA[i][start_time:end_time], '-', c='black', linewidth=1)         
            ax[i].set_ylabel("DOA (°)")
            ax[i].tick_params(axis="x", direction='in')
            ax[i].tick_params(which="minor", direction='in')    
            ax[i].tick_params(axis="x", direction='in')
            ax[i].tick_params(which="minor", direction='in')
            ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
            ax[i].yaxis.set_minor_locator(AutoMinorLocator(1))
            ax[i].xaxis.set_major_locator(MultipleLocator(50))
            ax[i].yaxis.set_major_locator(MultipleLocator(30)) 
            ax[i].xaxis.labelpad = 5
            ax[i].yaxis.labelpad = 5
            if i<4:
                # ax[i].axes.xaxis.set_ticklabels([])
                pass
            else:
                ax[i].set_xlabel("Time (s)")
        fig.tight_layout()

        if save_plot:    
            for fmt in ['png', 'svg', 'pdf']:
                plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)    
        plt.show()

def import_run(run_id):
    api = wandb.Api(overrides={
                           "project": project,       
                           "entity": entity,
                            })
    run = api.run(entity+'/'+project+'/'+run_id)
    return run
    
def plot_metrics(apply=False, save_plot=False, metrics="MAE (degrees) test", end=-1, ylabel="Mean Absolute Error (°)", run_id="", file_out=""):
    if apply:
        run = import_run(run_id)
        file_out +=  run.name +'_'+ metrics
        run_history = run.history(keys=[metrics])[metrics]
        epochs = np.arange(0, len(run_history))
        metric = np.zeros(len(run_history))
        for i, row in run.history(keys=[metrics]).iterrows():
            print(f"epoch {i}\t{metrics} = {row[metrics]:.4f}")
            metric[i] = row[metrics]
        fig, ax = plt.subplots(1)          
        fig.set_figwidth(10 * centimeters)
        fig.set_figheight(7 * centimeters)
        ax.plot(epochs[:end], metric[:end], '-o', c='darkblue', linewidth=2, ms=4) 
        # ax.set_xlim([0,21])
        ax.set_xlabel("Epochs")
        ax.set_ylabel(ylabel)
        ax.set_title(f"end value: {metric[end]:.4f}")
        ax.tick_params(axis="x", direction='in')
        ax.tick_params(which="minor", direction='in')    
        ax.tick_params(axis="x", direction='in')
        ax.tick_params(which="minor", direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        # ax.xaxis.set_major_locator(MultipleLocator(5))
        # ax.yaxis.set_major_locator(MultipleLocator(ML))  
        fig.tight_layout()
        ax.xaxis.labelpad = 5
        ax.yaxis.labelpad = 5
        if save_plot:    
            for fmt in ['png', 'svg', 'pdf']:
                plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)    
        plt.show()
            
def plot_metrics_multi_run(apply=False, save_plot=False, metrics="MAE (degrees) test", end=-1, ylabel="Mean Absolute Error (°)", label_list=[], colors=[], run_ids=[], file_out=""):
    if apply:
        file_out += metrics
        fig, ax = plt.subplots(1)          
        fig.set_figwidth(10 * centimeters)
        fig.set_figheight(7 * centimeters)
        metric_end = np.zeros(len(run_ids))
        for r, (run_id, label, color) in enumerate(zip(run_ids, label_list, colors)):
            run = import_run(run_id)        
            run_history = run.history(keys=[metrics])[metrics]
            epochs = np.arange(0, len(run_history))
            metric = np.zeros(len(run_history))
            for i, row in run.history(keys=[metrics]).iterrows():
                print(f"epoch {i}\t{metrics} = {row[metrics]:.4f}")
                metric[i] = row[metrics]
            ax.plot(epochs[:end], metric[:end], '-o', c=color, linewidth=2, ms=4, label=label) 
            metric_end[r] = metric[end]
        # ax.set_xlim([0,21])
        ax.legend()
        ax.set_xlabel("Epochs")
        ax.set_ylabel(ylabel)
        ax.set_title(''.join(["end value: "]+[f"{metric_end[r]:.4f}, " for r in range(len(run_ids))]))
        ax.tick_params(axis="x", direction='in')
        ax.tick_params(which="minor", direction='in')    
        ax.tick_params(axis="x", direction='in')
        ax.tick_params(which="minor", direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        # ax.xaxis.set_major_locator(MultipleLocator(5))
        # ax.yaxis.set_major_locator(MultipleLocator(ML))  
        fig.tight_layout()
        ax.xaxis.labelpad = 5
        ax.yaxis.labelpad = 5
        if save_plot:    
            for fmt in ['png', 'svg', 'pdf']:
                plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)    
        plt.show()

def plot_regression_results(apply=False, save_plot=False, file_out="", target_list=[], pred_list=[], start_time=0, end_time=-1):
    if apply:
        SR = 2000
        average_window=100
        start_time = int(SR*start_time)
        end_time = int(SR*end_time) if end_time != -1 else -1
        
        fig, ax = plt.subplots(5)          
        fig.set_figwidth(8.255 * centimeters)
        fig.set_figheight(14 * centimeters)
        for i, (target_file, pred_file) in enumerate(zip(target_list, pred_list)):
            target_file =  target_file + '.txt'
            pred_file = pred_file + '.txt'
            DOA = np.loadtxt(target_file)
            output = np.loadtxt(pred_file)
            time_steps = len(DOA)
            time = np.arange(0,time_steps)/SR*average_window
            ax[i].plot(time[start_time:end_time], DOA[start_time:end_time], '-', c='black', linewidth=1) 
            ax[i].plot(time[start_time:end_time], output[start_time:end_time], '-', c='red', linewidth=1) 
            ax[i].set_ylabel("DOA (°)")
            ax[i].tick_params(axis="x", direction='in')
            ax[i].tick_params(which="minor", direction='in')    
            ax[i].tick_params(axis="x", direction='in')
            ax[i].tick_params(which="minor", direction='in')
            ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
            ax[i].yaxis.set_minor_locator(AutoMinorLocator(1))
            # ax[i].xaxis.set_major_locator(MultipleLocator(50))
            ax[i].yaxis.set_major_locator(MultipleLocator(30)) 
            ax[i].xaxis.labelpad = 5
            ax[i].yaxis.labelpad = 5
            if i<4:
                # ax[i].axes.xaxis.set_ticklabels([])
                pass
            else:
                ax[i].set_xlabel("Time (s)")
        fig.tight_layout()

        if save_plot:    
            for fmt in ['png', 'svg', 'pdf']:
                plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)
        # plt.show()      
        return
    
def plot_regression_results_DOF(apply=False, save_plot=False, file_out="", target_list=[], pred_list=[], start_time=0, end_time=-1):
    if apply:
        SR = 2000
        average_window=100
        start_time = int(SR*start_time)
        end_time = int(SR*end_time) if end_time != -1 else -1
        
        fig, ax = plt.subplots(3,6)          
        fig.set_figwidth(30 * centimeters)
        fig.set_figheight(20 * centimeters)
        DOA_tensor = []
        output_tensor = []
        for i, (target_file, pred_file) in enumerate(zip(target_list, pred_list)):
            target_file =  target_file + '.txt'
            pred_file = pred_file + '.txt'
            DOA = np.loadtxt(target_file)
            output = np.loadtxt(pred_file)

            DOA_tensor += [torch.FloatTensor(DOA).unsqueeze(1)]
            output_tensor += [torch.FloatTensor(output).unsqueeze(1)]

        DOA_tensor = torch.cat(tuple(DOA_tensor), dim=1)
        output_tensor = torch.cat(tuple(output_tensor), dim=1)

        DOF_target = DOA_to_DOF(DOA_tensor)
        DOF_output = DOA_to_DOF(output_tensor)

        time_steps = len(DOF_target)
        time = np.arange(0,time_steps)/SR*average_window
        subplot_idx = np.reshape(np.arange(0,18), (3,6))
        for i in range(3):
            for j in range(6):
                ax[i,j].plot(time[start_time:end_time], DOF_target[start_time:end_time, subplot_idx[i,j]], '-', c='black', linewidth=1) 
                ax[i,j].plot(time[start_time:end_time], DOF_output[start_time:end_time, subplot_idx[i,j]], '-', c='red', linewidth=1) 
                ax[i,j].set_ylabel(f"Angle {subplot_idx[i,j]+1:.0f} (°)")
                ax[i,j].tick_params(axis="x", direction='in')
                ax[i,j].tick_params(which="minor", direction='in')    
                ax[i,j].tick_params(axis="x", direction='in')
                ax[i,j].tick_params(which="minor", direction='in')
                ax[i,j].xaxis.set_minor_locator(AutoMinorLocator(5))
                ax[i,j].yaxis.set_minor_locator(AutoMinorLocator(1))
                # ax[i,j].xaxis.set_major_locator(MultipleLocator(50))
                ax[i,j].yaxis.set_major_locator(MultipleLocator(30)) 
                ax[i,j].xaxis.labelpad = 5
                ax[i,j].yaxis.labelpad = 5
                if i<4:
                    # ax[i,j].axes.xaxis.set_ticklabels([])
                    pass
                else:
                    ax[i,j].set_xlabel("Time (s)")
        fig.tight_layout()

        if save_plot:    
            for fmt in ['png', 'svg', 'pdf']:
                plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)
        # plt.show()      
        return

def plot_average_subjects_metrics(apply=False, save_plot=False, metrics="MAE (degrees) test", end=-1, ylabel="Mean Absolute Error (°)", run_ids_list=[], colors=[], legend_list=[], file_out=""):
    if apply:
        file_out += '_'+metrics
        fig, ax = plt.subplots(1)          
        fig.set_figwidth(8.255 * centimeters)
        fig.set_figheight(6 * centimeters)
        title_string = ""
        legend_string = []
        for exp, run_ids in enumerate(run_ids_list):
            metric_end = np.zeros(len(run_ids))
            metric = np.zeros((len(run_ids), end+1))
            epochs = np.arange(0, end+1)
            for r, run_id in enumerate(run_ids):
                run = import_run(run_id)             
                i = 0    
                rows = run.history(keys=[metrics]).iterrows()
                while i < end:
                    print(i)
                    i, row = next(rows)      
                    print(f"epoch {i}\t{metrics} = {row[metrics]:.4f}")
                    metric[r, i] = row[metrics]         
            mean_metric, std_metric = np.mean(metric, axis=0), np.std(metric, axis=0)        
            ax.plot(epochs, mean_metric, '-', c=colors[exp], linewidth=3) 
            plt.fill_between(epochs, mean_metric-std_metric, mean_metric+std_metric, color=colors[exp], alpha=0.1)
            mean_end, std_end = mean_metric[-1], std_metric[-1]     
            title_string += f"{mean_end:.3f} $\pm$ {std_end:.3f}\t"
        # ax.legend(legend_list)
        # ax.set_xlim([1,10])
        ax.set_xlabel("Epochs")
        ax.set_ylabel(ylabel)
        ax.set_title(title_string)
        ax.tick_params(axis="x", direction='in')
        ax.tick_params(which="minor", direction='in')    
        ax.tick_params(axis="x", direction='in')
        ax.tick_params(which="minor", direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        fig.tight_layout()
        ax.xaxis.labelpad = 5
        ax.yaxis.labelpad = 5
        if save_plot:    
            for fmt in ['png', 'svg', 'pdf']:
                plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)    
        plt.show()

def plot_average_subjects_metrics_bar(apply=False, save_plot=False, metrics_list="MAE (degrees) test", end=-1, ylabel_list="Mean Absolute Error (°)", run_ids_list=[], colors=[], legend_list=[], file_out=""):
    if apply:
        file_out += '_'+'merged'
        fig, ax = plt.subplots(nrows=1, ncols=3)          
        fig.set_figwidth(8.255 * centimeters)
        fig.set_figheight(8 * centimeters)
        for m, (metrics, ylabel) in enumerate(zip(metrics_list, ylabel_list)):
            title_string = ""
            legend_string = []
            mean_metric = []
            std_metric = []
            for exp, run_ids in enumerate(run_ids_list):
                metric_end = np.zeros(len(run_ids))
                metric = np.zeros((len(run_ids), end+1))
                epochs = np.arange(0, end+1)
                for r, run_id in enumerate(run_ids):
                    run = import_run(run_id)             
                    i = 0    
                    rows = run.history(keys=[metrics]).iterrows()
                    while i < end:
                        try:
                            i, row = next(rows)      
                            print(f"epoch {i}\t{metrics} = {row[metrics]:.4f}")
                        except:
                            i = end
                        metric[r, i] = row[metrics]         
                mean_metric += [np.mean(metric, axis=0)[-1]]
                std_metric += [np.std(metric, axis=0)[-1]]                       
                ax[m].bar(np.array([exp]), np.array(np.mean(metric, axis=0)[-1]), yerr=np.array(np.std(metric, axis=0)[-1]), align='center', width=0.7, alpha=0.7, color=colors[exp], ecolor=colors[exp], capsize=10)  
                title_string += f"{mean_metric[-1]:.3f}$\pm${std_metric[-1]:.3f}\n"
            if m > 0:
                ax[m].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                # lif mlp and int8
                # ax[m].set_ylim([0.7,1])
                # ablation study
                ax[m].set_ylim([0.5,1])
            else:
                # ax[m].legend(legend_list,loc='upper right')
                # lif mlp and int8
                # ax[m].set_ylim([4,8.4])
                # ablation study
                ax[m].set_ylim([4,11])
            ax[m].set_ylabel(ylabel)
            ax[m].set_title(title_string)
            ax[m].tick_params(axis="x", direction='in')
            ax[m].tick_params(which="minor", direction='in')    
            ax[m].tick_params(axis="x", direction='in')
            # ax[m].tick_params(which="minor", direction='in')
            ax[m].xaxis.set_minor_locator(AutoMinorLocator(1))
            ax[m].yaxis.set_minor_locator(AutoMinorLocator(5))
            ax[m].set_xticks([i for i,_ in enumerate(run_ids_list)])
            # labels = [item.get_text() for item in ax[m].get_xticklabels()]
            # for i, label in enumerate(legend_list): 
            #     labels[i] = label        
            # ax[m].set_xticklabels(labels)
            # plt.xticks(rotation=45)
            ax[m].xaxis.labelpad = 1
            ax[m].yaxis.labelpad = 1
        plt.subplots_adjust(wspace=0.7,
                            top=0.6)
        fig.tight_layout(pad=0.2, w_pad=0.8, h_pad=0.)
        if save_plot:    
            for fmt in ['png', 'svg', 'pdf']:
                plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)    
        plt.show()
        
def plot_sweep_results_ks(apply=False, save_plot=False, metrics="MAE (degrees) test", end=-1, ylabel="Mean Absolute Error (°)", run_ids=[], file_out=""):
    if apply:
        file_out += '_'+metrics
        fig, ax = plt.subplots(1)          
        fig.set_figwidth(10 * centimeters)
        fig.set_figheight(7 * centimeters)
        metric_end = np.zeros(len(run_ids))
        metric = np.zeros((len(run_ids), end+1))
        epochs = np.arange(0, end+1)
        ks = np.zeros(len(run_ids))
        for r, run_id in enumerate(run_ids):
            run = import_run(run_id)        
            ks[r] = run.config['conv_kernel_size']     
            i = 0    
            rows = run.history(keys=[metrics]).iterrows()
            if len(run.history(keys=[metrics])) >= end:
                while i < end:                    
                    i, row = next(rows)      
                    print(f"epoch {i}\t{metrics} = {row[metrics]:.4f}")
                    metric[r, i] = row[metrics]           
        end_metrics = metric[:, -1]   
        idx = np.argsort(ks)
        ks, end_metrics = ks[idx], end_metrics[idx]     
        ax.plot(ks, end_metrics, 'o-', c="darkblue", linewidth=2)   
        ax.set_xlabel("Kernel size")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", direction='in')
        ax.tick_params(which="minor", direction='in')    
        ax.tick_params(axis="x", direction='in')
        ax.tick_params(which="minor", direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        fig.tight_layout()
        ax.xaxis.labelpad = 5
        ax.yaxis.labelpad = 5
        if save_plot:    
            for fmt in ['png', 'svg', 'pdf']:
                plt.savefig(file_out + '.%s' % fmt, format=fmt, dpi=1200)    
        plt.show()
        
def plot_sweep_results_svs(apply=False, save_plot=False, metrics_list=[], end=-1, ylabel_list=[], run_ids=[], subjects_list=[], svs_list=[], run_ids_list=[], colors=[], symbol=[],file_out=""):
    if apply:        
        end_metrics = np.zeros((len(metrics_list), len(run_ids_list), len(subjects_list), len(svs_list)))
        for exp, run_ids in enumerate(run_ids_list):
            for r, run_id in enumerate(run_ids):
                run = import_run(run_id)    
                svs = run.config['stored_vector_size']
                sub = run.config['subjects'][0]   
                if (len(run.history(keys=[metrics_list[0]]))>=end) and (svs in svs_list) and (sub in subjects_list):
                    for m, (metrics, ylabel) in enumerate(zip(metrics_list, ylabel_list)):         
                        rows = run.history(keys=[metrics]).iterrows()
                        i = 0  
                        for i in range(end):                    
                            i, row = next(rows)                   
                        print(f"subjects {sub}\tstored_vector_size {svs}\t{metrics} = {row[metrics]:.4f}")
                        end_metrics[m, exp, subjects_list==sub, svs_list==svs] = row[metrics]         
        for m, (metrics, ylabel) in enumerate(zip(metrics_list, ylabel_list)): 
            fig, ax = plt.subplots(1)          
            fig.set_figwidth(8.255 * centimeters)
            fig.set_figheight(5.5 * centimeters)  
            for exp, run_ids in enumerate(run_ids_list):   
                ax.plot(svs_list, np.mean(end_metrics[m, exp], axis=0), '-'+symbol[exp], c=colors[exp], ms=3, linewidth=1.5)   
                # ax.errorbar(svs_list, np.mean(end_metrics[m, exp], axis=0), yerr=np.std(end_metrics[m, exp], axis=0), fmt='-'+symbol[exp], color=colors[exp], ms=3, linewidth=1.5, capsize=4, ecolor=colors[exp], elinewidth=1) 
                # ax.bar(svs_list, np.mean(end_metrics[m], axis=0), yerr=np.std(end_metrics[m], axis=0), align='center', width=2,alpha=1, ecolor='darkblue', capsize=10)
            ax.legend(legend_list)
            ax.set_xlabel("M")
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", direction='in')
            ax.tick_params(which="minor", direction='in')    
            ax.tick_params(axis="x", direction='in')
            ax.tick_params(which="minor", direction='in')
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.set_xticks([svs for svs in svs_list])
            fig.tight_layout()
            ax.xaxis.labelpad = 5
            ax.yaxis.labelpad = 5
            if save_plot:    
                for fmt in ['png', 'svg', 'pdf']:
                    plt.savefig(file_out + '_' + metrics + '.%s' % fmt, format=fmt, dpi=1200)    
            # plt.show()
            fig, ax = plt.subplots(1)      
            fig.set_figwidth(15)  
            fig.set_figheight(12)   
            plt.table(cellText=[[f'{np.mean(end_metrics[m, exp], axis=0)[i]:.3f}$\pm${np.std(end_metrics[m, exp], axis=0)[i]:.3f}' for i in range(len(svs_list))] for exp,_ in enumerate(run_ids_list)],
                    colLabels=[f'{svs:.0f}' for svs in svs_list],
                    rowLabels=legend_list,
                    cellLoc='center',
                    )            
            ax.set_xticks([])
            ax.set_yticks([])
            if save_plot:    
                for fmt in ['png', 'svg', 'pdf']:
                    plt.savefig(file_out + '_table_' + metrics + '.%s' % fmt, format=fmt, dpi=1200)   
            # plt.show()
    return
   
## create a wandb session with config args ###
outer_parameters=dict(subjects=[0])
args = return_args(**outer_parameters) 
args["normalize_data"] = False
wandb_run = wandb.init(project="", name='', config=args, mode="disabled")
args = wandb_run.config

###############################################

### Plot single channel emg signal ###
plot_emg(apply=False, start_time=0, end_time=-1, save_plot=True, file_out = out_file_root+"emg_test_full_fitterd_to_column_width")
######################################

### Plot single channel autocorrelation signal ###
plot_autocorrelation(apply=False, start_time=0, end_time=-1, save_plot=True, file_out = out_file_root+"emg_autocorrelation")
######################################

### Plot 5 degrees Degree Of Actuation ###
plot_DOA(apply=False, start_time=0, end_time=-1, save_plot=True, file_out = out_file_root+"emg_plots/DOA_test_full")
##########################################

### Plot results for a single run run_id ###
for met, ylabel in zip(metrics_list, ylabel_list):
    plot_metrics(apply=False, 
                save_plot=False,
                metrics=met,
                end=-1,
                ylabel=ylabel,
                run_id="2o08fzwg",
                file_out=out_file_root+"results/",
                )
############################################

### Plot results for multiple runs for one plot ###
# run_list = ["31xis6lh","2o08fzwg","1xgoafya"]
# label_list = ["Transformer","ContinuousTransformer","Pre-trained ContinuousTransformer"]
# colors = ["black", "darkred", "darkblue"]
run_list = ["1qgkgzcr","4rg8gjtu","f6muurgs"]
label_list = ["Transformer","ContinuousTransformer", "ContinuousTransformer LIF MLP block"]
colors = ["black", "darkblue", "darkred"]
for met, ylabel in zip(metrics_list, ylabel_list):
    plot_metrics_multi_run(apply=False, 
                save_plot=False,
                metrics=met,
                end=19,
                ylabel=ylabel,
                colors=colors,
                run_ids=run_list,
                label_list=label_list,
                file_out=out_file_root+"results/pretraining+no-pre_training_3MLP_ks=24 (2)",
                )
#################################################

### Plot predictions versus ground truth already saved in txt files. These data can be saved during testing ###
root = out_file_root + 'results/angle_plots/'
target_list = [root + f'1_LIF_online_transformer_target_angle_{i}' for i in range(5)]
pred_list = [root + f'1_LIF_online_transformer_pred_angle_{i}' for i in range(5)]
plot_regression_results(
                        apply=False, 
                        save_plot=False,
                        file_out=root+'target_vs_pred_1MLP_LIF',
                        target_list=target_list,
                        pred_list=pred_list,
                        )
plot_regression_results_DOF(
                        apply=False, 
                        save_plot=False,
                        file_out=root+'DOF_target_vs_pred_1MLP_LIF',
                        target_list=target_list,
                        pred_list=pred_list,
                        )
###############################################################################################################

### Plot average results on multiple subjects ###
def get_run_ids(group):
    api = wandb.Api(overrides={
                "project": project,       
                "entity": entity,
                    })
    runs = api.runs(path=entity+'/'+project,
             filters={"$and": [
                                {'group': group},
                                {"config.stored_vector_size": int(150)}
                                ]
                     }
                    )  
    run_list = [run.id for run in runs]
    return run_list    

# MLP LIF  and quantized results   
# groups = ["self-attention ws=2000 ks=7 s=5", "sweep svs ws=2000 ks=7 s=5", "1 LIF MLP ws=2000 ks=7 s=5 svs=150", "quantized 1 LIF MLP head_dim=8"]
# Ablation study
groups = ["self-attention ws=2000 ks=7 s=5", "sweep svs ws=2000 ks=7 s=5", "ablation_MLP_ws=2000_ks=7_s=5", "ablation_attention_ws=2000_ks=7_s=5"]
run_list = [get_run_ids(group) for group in groups] 
# colors = ['black', 'darkblue', 'darkred', 'darkgreen']
colors = ['black', 'darkblue', 'brown', 'darkmagenta']
# legend_list = ['Transformer', 'Online Transformer', 'Online Transformer with SNN', 'Int8 Online Transformer with SNN']
legend_list = ['Transformer', 'Online Transformer', 'Online Transformer with attention ablation', 'Online Transformer with MLP ablation']
for met, ylabel in zip(metrics_list, ylabel_list):
    plot_average_subjects_metrics(
                                apply=False,
                                save_plot=False,
                                metrics=met,
                                end=10,
                                ylabel=ylabel,
                                run_ids_list=run_list,
                                colors=colors,
                                legend_list=legend_list,
                                file_out=out_file_root+"3_networks_ws=2000_k=7_s=5_svs=150",
                                )

# ylim_list=[]
plot_average_subjects_metrics_bar(
                            apply=False,
                            save_plot=False,
                            metrics_list=metrics_list,
                            end=10,
                            ylabel_list=ylabel_list,
                            run_ids_list=run_list,
                            colors=colors,
                            legend_list=legend_list,
                            # ylim_list=ylim_list,
                            # file_out=out_file_root+"4_networks_ws=2000_k=7_s=5_svs=150_bar",
                            file_out=out_file_root+"ablation_study_ws=2000_k=7_s=5_svs=150_bar",
                            )
###############################################################

###############################################################################################################

### Plot results on different kernel size and different stored vector size ###
def get_run_ids(group="", subjects_list=None, svs_list=None):
    api = wandb.Api(overrides={
                "project": project,       
                "entity": entity,
                    })
    subjects_list = [{"config.subjects": [int(sub)]} for sub in subjects_list] if subjects_list is not None else {'group': group}
    svs_condition = [{"config.stored_vector_size": int(svs)} for svs in svs_list] if svs_list is not None else {'group': group}
    runs = api.runs(path=entity+'/'+project,
                    filters={"$and": [
                                    {'group': group},
                                    {"$or": subjects_list
                                    },
                                    {"$or": svs_condition
                                    }
                                    ]
                            }
                    )
    run_list = [run.id for run in runs]
    return run_list    
run_list = get_run_ids('test ks 3 MLP')
for met, ylabel in zip(metrics_list, ylabel_list):
    plot_sweep_results_ks(
                        apply=False,
                        save_plot=False,
                        metrics=met,
                        end=20,
                        ylabel=ylabel,
                        run_ids=run_list,
                        file_out=out_file_root+"results_ks_sweep_3MLP",
                        )

subjects_list = np.arange(0,12)
svs_list = np.arange(10,160,20)
groups = ['sweep svs ws=2000 ks=7 s=5', 'sweep svs ws=2000 ks=15 s=13', 'sweep svs ws=2000 ks=20 s=18', 'sweep svs ws=2000 ks=25 s=23', 'sweep svs ws=2000 ks=30 s=28']
# run_list = get_run_ids('sweep svs ws=2000 ks=7 s=5', subjects_list=subjects_list, svs_list=svs_list)
run_list = [get_run_ids(group, subjects_list=subjects_list, svs_list=svs_list) for group in groups] 
colors = [cm.copper((k-7)/(30-7)) for k in [7,15,20,25,30]]

legend_list = ['kernel size = 7', 'kernel size = 15', 'kernel size = 20', 'kernel size = 25', 'kernel size = 30']
symbol = ['o', 'd', 's', '^', 'x']
plot_sweep_results_svs(
                    apply=False,
                    save_plot=False,
                    metrics_list=metrics_list,
                    end=10,
                    ylabel_list=ylabel_list,
                    run_ids_list=run_list,
                    subjects_list=subjects_list,
                    svs_list=svs_list,
                    colors=colors,
                    symbol=symbol,
                    file_out=out_file_root+"results_svs_sweep_merged",
                    )
###############################################################


