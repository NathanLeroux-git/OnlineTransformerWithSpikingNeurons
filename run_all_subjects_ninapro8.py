from main_2models import main
import numpy as np

wandb_status="disabled"
project="sEMG_DOA_regression"
# window size
ws=2000
# size of kernel and stride of the convolution embedding
ks=7
stride=5
padding=int((ks-stride)/2)
# number of time windows
n_tokens=int((ws-ks+2*padding)/stride+1)
# stored vector size (called memory length M in the article)
svs_list = np.array([150])

group='TESTS'
device='cuda:0'
subjects = [0,1,2,3,4,5,6,7,8,9,10,11]
for i in range(len(svs_list)): 
    subject=0
    for subject in subjects:
        stored_vector_size=svs_list[i] 
        name = f'sub_{subject:02d}_continuous_transfo_conv_embed_lr=1e-03_datAug=64_bs=64_wd=0e+00_ws={ws}_np={n_tokens}_svs={stored_vector_size}_ks={ks}_s={stride}_p={padding} (0)'
        # experiment parameters
        kwargs = dict(outer_parameters=dict(
                                        wandb_status=wandb_status,
                                        pre_trained_model_name=group+'_'+name,

                                        binarize_embedding=True,
                                        spiking_transformer_mlp=True,
                                        spiking_qkv_proj=True,                                        

                                        project=project,
                                        name=name,
                                        group=group,
                                        device=device,   
                                        subjects=subject,                                                   
                                        window_size=ws,
                                        sliding_size=ws,
                                        n_tokens=n_tokens,
                                        stored_vector_size=stored_vector_size,                 
                                        conv_kernel_size=ks,
                                        conv_stride=stride,                                    
                                        save_model=True,
                                        save_model_name=group+'_'+name,
                                        )
                )
        success = main(**kwargs)    
        if not(success):
            break