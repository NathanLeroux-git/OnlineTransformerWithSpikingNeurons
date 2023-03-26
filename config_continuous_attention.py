from torch import embedding
import argparse

def return_args(**outer_params):
    args = dict(
        project="sEMG_DOA_regression",
        name="",
        group="",
        dataset='ninapro8',

        training=True,
        testing=True,

        # Model
        Net='ContinuousTransformer', #'ContinuousTransformer' is the transformer with online attention, 'transformer' is the transformer with self-attention
        trainingNet='ParallelTrainingTransformer', #'ParallelTrainingTransformer' is equivalent to 'ContinuousTransformer', but not online and parallel.
        pre_trained_model_name=None, 
        post_quantization=False,

        # machine learning hyperparameters
        optimizer='Adam', 
        loss_fn='L1Loss',     
        weight_decay=0,
        test_batch_size=1, #45900                       

        epochs=11,   
        previous_job_epoch=0,
        batch_size=64,
        lr=1e-3,

        ### scheduler config###
        scheduler='StepLR',
        scheduler_args = dict(step_size=1000,
                            gamma=1,
                            ),

        # sparsity loss config
        sparsity_loss_threshold = 0.99,
        sparsity_loss_coeff = 0,        
        
        # dataset parameters
        data_augmentation_factor=64,
        normalize_data=True,
        window_size=2000,
        sliding_size=30,
        shuffle_dataset=True,     
        use_accelerometer=False,
        convert_raw_data_to_spikes=False,
        delta_spike_threshold=0.2, 
        off_spike=True,
        spectrogram=False, # if True, better set the at delta_spike_threshold to 0.01. Note that with a stride of 5, the total signal length is 5 times shorter in number of time steps: need to adapt the window and overlap dataset size (divide by 5)
        spectrogram_config=[200,1], # window size, stride. 

        # SNN modules
        spiking_embedding=False,
        spiking_qkv_proj=False,
        spiking_transformer_mlp=False,
        spiking_classifier=False,
        reccurent_LIF=False,

        # Binarized modules
        binarize_embedding=False,
        binarize_qkv=False,

        # transformer embedding parameters
        n_tokens=400,     # the number of elements in the sequence used to build a single token. for the convolutional embedding, it is equal to the filter size (length).
        embed_dim=64,       # the dimension of each token. for the convolutional embedding, it is equal to the number of convolution filters.
        conv_kernel_size=7,
        conv_stride=5,

        # encoder block parameters
        transfo_depth=1,
        mlp_depth=1,
        n_heads=8,          # the number of attention heads
        head_dim=32,        # the dimension of each attention head fo each token
        pos_drop_p=.0,      # dropout probability after adding the positional embedding vector
        pos_drop_decoder_p=.0,
        atten_drop_p=.0,    # dropout probability of the self-attention weights (on softmax(k.q) dot product)
        proj_drop_p=.2,     # dropout probability after linear projections of the concatened multi-head self-attention                                      
        mlp_drop_p=.2,      # dropout probability after each of the two linear units of the MLP   
        mlp_hidden_size=128,
        qkv_bias=False,
        mlp_NL="GELU",

        # online attention memory depth
        stored_vector_size=150,

        # additional network parameters
        end_NL='GELU', 
        classifier_hidden=None,

        # LIF neurons parameters
        alpha=.95,  # membrane potential decay parameter
        beta=.90,   # input current constant
        betar=.85,  # recurrent current constant

        # choose the datasets that are used
        subjects=0,

        # miscellaneous
        device='cuda',
        seed=1,
        log_interval=40,
        save_model=False,
        save_model_name='',
        wandb_status="disabled" # "online", "offline"
        )   
    
    # Replacing the values of the dictonary with the one that were passed in outer_params
    for key, value in outer_params.items():
        assert key in args
        args[key] = value     

    return args     