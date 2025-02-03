import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
import continuous_model
        
class ContinuousTransformer(nn.Module):
    '''
    This is the Online Transformer module. It computes tokens one at the time. It is thus not recommanded to use it for training.
    '''
    def __init__(self, args, dim_in, dim_out):
        super().__init__()
        self.device = args.device
        self.in_channels = dim_in
        self.n_tokens = args.n_tokens
        self.embed_dim = args.embed_dim
        self.dim_out = dim_out
        self.window_size = args.window_size
        if args.spiking_embedding:
            self.input_reshape = no_overlap_reshape(args, dim_in)
            self.pos_embed = nn.Identity()   
            self.pos_drop = nn.Identity()    
            self.PatchEmbed = ContinuousLIFPatchEmbed(args, in_chans=dim_in)
        else:
            self.input_reshape = overlap_reshape(dim_in, self.window_size, self.n_tokens, args.conv_kernel_size, args.conv_stride, self.device)            
            self.pos_embed = nn.Identity()   
            self.pos_drop = nn.Dropout(p=args.pos_drop_p) if not(args.binarize_embedding) else nn.Identity()  
            self.patch_size = 1
            self.upsampling_factor = self.input_reshape.stride
            self.PatchEmbed = ContinuousConvPatchEmbed(args, in_chans=dim_in)    

        self.binarize_embedding = SmoothStep().apply if args.binarize_embedding else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                transformer_block_encoder(args)
                for _ in range(args.transfo_depth)
            ]
        )                
        self.norm = nn.LayerNorm(args.embed_dim) if not(args.spiking_classifier) else nn.Identity()
        self.NL =  getattr(nn, args.end_NL)() if not(args.spiking_classifier) else nn.Identity()
        self.classifier = MLP(args, args.embed_dim, args.classifier_hidden, dim_out) if not(args.spiking_classifier) else LIFNetwork(args, args.embed_dim, args.classifier_hidden, dim_out, readout_fn='U_readout')
        self.reset_values_to_zero = self.init_epoch if args.shuffle_dataset else nn.Identity()
        
    def forward(self, input, target=0):
        if self.training:
            # Init kv values to zeros and initialize LIF neuron states
            self.reset_values_to_zero(input)
        n_samples = input.shape[0]
        # Detach previous kv values and LIG neuron states from autograd graph. Only useful for training.
        self.init_batch()   
        upsampled_output = torch.zeros(n_samples, self.dim_out, self.n_tokens, self.input_reshape.upsampling_factor).to(self.device)             
        input = self.input_reshape(input) # Reshaping the input is useful to prepare it for embedding: e.g., for convolutional embedding, the input is converted in sliding window, and the embedding module only applies linear on each of them.        
        sparsity_tensor = torch.zeros(self.n_tokens, 9).to(self.device)
        for token in range(self.n_tokens):
            x = input[:,:,token] # Select a single time window 
            x = self.PatchEmbed(x) # Project the input time window into a single token of size embedding_dim
            x = self.pos_embed(x) # Positional embedding
            x = self.pos_drop(x) # Dropout
            x = self.binarize_embedding(x) # Convert each embedding value into either 1 or 0
            x, sparsity, qkv = self.transformer_step(self.blocks, x) # Encoder block step
            sparsity_tensor[token] = sparsity # Store the sparsity measurements
            x = self.norm(x) # Layer Norm
            x = self.NL(x) # Activation function (GeLU, or Identity if spiking classifier)
            x = self.classifier(x) # Linear layer (embedding dim -> DoA)
            x = x.unsqueeze(-1).expand(-1,-1,self.input_reshape.upsampling_factor) # Up sampling: we want the size of the output match the size of the target
            upsampled_output[:,:,token] = x
        upsampled_output = upsampled_output.flatten(2)
        return upsampled_output, torch.mean(sparsity_tensor, dim=0), qkv # We return the output of the network and measurements of the sparsity
    
    def transformer_step(self, blocks, x):   
        sparsity_tensor = torch.zeros(len(blocks), 9).to(x.device)
        for b, block in enumerate(blocks): 
            x, sparsity_attention, qkv, sparsity_mlp = block(x)
            sparsity_tensor[b] = torch.cat([sparsity_attention, sparsity_mlp.unsqueeze(0)])
        return x, torch.mean(sparsity_tensor, dim=0), qkv
    
    def init_epoch(self, x):
        n_samples = x.shape[0]
        # Embedding
        self.PatchEmbed.init_state() 
        # Transformer       
        for block in self.blocks:
            block.atten.zeros_KV(n_samples)
            block.atten.qkv.init_state() if isinstance(block.atten.qkv, LIFNetwork) else None
            block.MLP_layers.init_state()   
        # Classifier          
        self.classifier.init_state()
        
    def init_batch(self):
        # Embedding 
        self.PatchEmbed.detach_LIF_states()
        # Transformer
        for block in self.blocks:
            block.atten.detach_KV()
            block.atten.qkv.detach_LIF_states() if isinstance(block.atten.qkv, LIFNetwork) else None
            block.MLP_layers.detach_LIF_states()
        # Classifier 
        self.classifier.detach_LIF_states()      
        
class ParallelTrainingTransformer(nn.Module):
    '''
    This module is used to train the Online Transformer faster by making it parallel. It computes tokens multiple tokens simultaneaously.
    '''
    def __init__(self, args, dim_in, dim_out):
        super().__init__()       
        self.device = args.device 
        self.n_tokens = args.n_tokens        
        self.in_channels = dim_in        
        self.window_size = args.window_size         
        if args.spiking_embedding:
            self.input_reshape = no_overlap_reshape(args, dim_in)
            self.pos_embed = nn.Identity()   
            self.pos_drop = nn.Identity()    
            self.PatchEmbed = getattr(continuous_model, "LIFPatchEmbed")(args, in_chans=dim_in)       
        else:               
            self.input_reshape = overlap_reshape(dim_in, self.window_size, self.n_tokens, args.conv_kernel_size, args.conv_stride, self.device)
            self.patch_size = 1            
            self.pos_embed = nn.Identity()   
            self.pos_drop = nn.Dropout(p=args.pos_drop_p)  
            self.PatchEmbed = getattr(continuous_model, "ConvPatchEmbed")(args, in_chans=dim_in)
                    
        self.binarize_embedding = SmoothStep().apply if args.binarize_embedding else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                transformer_block_encoder(args, continuous=False, KV_reshape=True)
                for _ in range(args.transfo_depth)
            ]
        )  
        self.norm = nn.LayerNorm(args.embed_dim) 
        self.NL =  getattr(nn, args.end_NL)() if (args.classifier_hidden and not(args.spiking_classifier)) is None else nn.Identity()    
        self.classifier = MLP(args, args.embed_dim, args.classifier_hidden, dim_out) if not(args.spiking_classifier) else LIFNetwork_multi_steps(args, args.embed_dim, args.classifier_hidden, dim_out, readout_fn='U_readout')     
        self.head_dim = args.head_dim
        self.n_heads = args.n_heads
        
    def forward(self, x, target=0):
        self.init_batch() # Initialize LIF neuron states
        x = self.input_reshape(x) # Reshaping the input is useful to prepare it for embedding: e.g., for convolutional embedding, the input is converted in sliding window, and the embedding module only applies linear on each of them.        
        x = self.PatchEmbed(x) # Embedding module   
        x = self.pos_embed(x) # Positionel embedding
        x = self.pos_drop(x) # Dropout
        x = self.binarize_embedding(x) # Convert embedded tokens into 1 or 0.
        embedding = x # Sparsity measurement used in training
        x, sparsity, qkv = self.transformer_step(self.blocks, x) # Encoder block step
        x = self.norm(x) # Layer Norm   
        x = self.NL(x) # Activation function    
        x = self.classifier(x) # Linear layer
        x = x.transpose(1,2)
        x = x.unsqueeze(-1).expand(-1,-1,-1,self.input_reshape.upsampling_factor) # Up sampling layer
        x = x.flatten(2)
        return x, embedding, sparsity, qkv # We return the network output "x", a sparsity metric "sparsity", and two tensors "embedding" and "qkv" used for the loss function
    
    def transformer_step(self, blocks, x):   
        sparsity_tensor = torch.zeros(len(blocks), 9).to(x.device)
        for b, block in enumerate(blocks): 
            x, sparsity_attention, qkv, sparsity_mlp = block(x)
            sparsity_tensor[b] = torch.cat([sparsity_attention, sparsity_mlp.unsqueeze(0)])
        return x, torch.mean(sparsity_tensor, dim=0), qkv
    
    def init_epoch(self, x):
        return    
    
    def init_batch(self):
        # Embedding 
        self.PatchEmbed.init_state()
        # Transformer
        for block in self.blocks:
            block.atten.qkv.init_state() if isinstance(block.atten.qkv, LIFNetwork_multi_steps) else None
            block.MLP_layers.init_state()
        # Classifier 
        self.classifier.init_state() 
    
class overlap_reshape(nn.Module):
    """
    Reshape the input into sliding windows to prepare for convolution
    """ 
    def __init__(self, in_channels, window_size, n_tokens, kernel_size, stride, device):
        super().__init__()
        self.n_tokens = n_tokens
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = int((kernel_size-stride)/2)
        self.unfold = nn.Unfold((1,kernel_size), stride=(1,self.stride), padding=(0,self.padding))
        self.upsampling_factor = stride
    def forward(self, x):
        n_samples, in_channels, times = x.shape
        x = x.unsqueeze(2)
        x = self.unfold(x)
        x = x.reshape(n_samples, in_channels, self.kernel_size, self.n_tokens)
        x = x.transpose(2,3)
        return x
    
class ContinuousConvPatchEmbed(nn.Module):
    """
    Multiply a kernel to a single window of a convolution
    """
    def __init__(self, args, in_chans=14):
        super().__init__()
        self.n_tokens = args.n_tokens
        self.Proj = nn.Linear(in_chans*args.conv_kernel_size, args.embed_dim)
    def forward(self,x):
        n_sample, in_channels, time = x.shape
        x = x.reshape(n_sample, in_channels * time) # returns x with shape (n_samples, in_channels*patch_size)
        x = self.Proj(x)  # returns x with shape (n_samples, embeded_dim)
        return x
    def init_state(self):
        return    
    def detach_LIF_states(self):
        return
    
class ConvPatchEmbed(nn.Module):
    """
    Multiply a kernel to multiple sliding windows of a convolution
    """
    def __init__(self, args, in_chans=14):
        super().__init__()
        self.n_tokens = args.n_tokens
        self.Proj = nn.Linear(in_chans*args.conv_kernel_size, args.embed_dim)
    def forward(self,x):
        n_sample, in_channels, n_tokens, time = x.shape
        x = x.transpose(1,2)
        x = x.reshape(n_sample, n_tokens, in_channels * time) # returns x with shape (n_samples, in_channels*patch_size)
        x = self.Proj(x)  # returns x with shape (n_samples, embeded_dim)
        return x
    def init_state(self):
        return    
    def detach_LIF_states(self):
        return
    
class transformer_block_encoder(nn.Module):
    def __init__(self, args, continuous=True, KV_reshape=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(args.embed_dim) if not(args.binarize_embedding) and not(args.spiking_embedding) else nn.Identity()
        atten_fn = getattr(continuous_model, "Attention") if continuous else getattr(continuous_model, "ParallelAttention") if not(KV_reshape) else getattr(continuous_model, "ReshapedParallelAttention") 
        self.atten = atten_fn(args)
        self.norm2 = nn.LayerNorm(args.embed_dim) 
        if not(args.spiking_transformer_mlp):
            self.MLP_layers = MLP(args, args.embed_dim, [args.mlp_hidden_size for _ in range(args.mlp_depth)], args.embed_dim)
        else:
            if continuous: # This is used with the model "ContinuousTransformer"
                self.MLP_layers = LIFNetwork(args, args.embed_dim, [args.mlp_hidden_size for _ in range(args.mlp_depth)], args.embed_dim, readout_fn='U_readout')
            else:          # This is used with the model "ParallelTrainingTransformer"
                self.MLP_layers = LIFNetwork_multi_steps(args, args.embed_dim, [args.mlp_hidden_size for _ in range(args.mlp_depth)], args.embed_dim, readout_fn='U_readout')

    def forward(self, x):
        y, sparsity_attention, qkv = self.atten(self.norm1(x)) # y is the output of the attention module, sparsity_attention is a metric, qkv is used in the loss function
        x = x + y # Residual Connection
        y = self.MLP_layers(self.norm2(x)) # Feedforward Neural Network
        # Sparsity measure of the first layer of the FNN (if spiking)
        if isinstance(self.MLP_layers, LIFNetwork) or isinstance(self.MLP_layers, LIFNetwork_multi_steps):
            sparsity_mlp = sparsity_measure(self.MLP_layers.layers[0].state.S)
        else:
            sparsity_mlp = torch.tensor(0).to(x.device)
        x = x + y # Residual Connection
        return x, sparsity_attention, qkv, sparsity_mlp

class Attention(nn.Module):
    """
    Continuous Attention module
    """
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.inner_dim = args.head_dim * args.n_heads
        self.scale = self.head_dim**-0.5 
        self.atten_drop = nn.Dropout(args.atten_drop_p) if (not(args.spiking_qkv_proj) and not(args.binarize_qkv)) else nn.Identity()
        self.proj_drop = nn.Dropout(args.proj_drop_p) if (not(args.spiking_qkv_proj) and not(args.binarize_qkv)) else nn.Identity()
        self.qkv = nn.Linear(args.embed_dim, self.inner_dim * 3, bias=args.qkv_bias) if not(args.spiking_qkv_proj) else LIFNetwork(args, args.embed_dim, None,  self.inner_dim * 3, readout_fn='S_readout')
        self.proj = nn.Linear(self.inner_dim, args.embed_dim)        
        self.softmax_procedure = normal_sequential_softmax_procedure(args) if (not(args.spiking_qkv_proj) and not(args.binarize_qkv) and not(args.spiking_embedding)) else sparse_softmax_procedure(args)
        self.binarize_qkv = SmoothStep().apply if args.binarize_qkv else nn.Identity()
        self.device = args.device
        self.stored_vector_size = args.stored_vector_size
        self.K = torch.zeros(1, self.n_heads, args.stored_vector_size, self.head_dim).to(self.device) 
        self.V = torch.zeros(1, self.n_heads, args.stored_vector_size, self.head_dim).to(self.device)         

    def forward(self, x):
        n_samples, embed_dim = x.shape    
        embedding_sparsity = sparsity_measure(x) 
        self.K = torch.roll(self.K, -1, dims=2) # Shift the adress of the element of K we will replace
        self.V = torch.roll(self.V, -1, dims=2) # Shift the adress of the element of V we will replace
        qkv = self.qkv(x) # (n_samples, 3*embed_dim) -> this is equivalent to make 3 different fully connected to tensors q,k and v
        qkv = self.binarize_qkv(qkv) # Change to 1 or 0              
        qkv = qkv.reshape(n_samples, 3, self.n_heads, self.head_dim)  
        qkv = qkv.permute(1, 0, 2, 3) # ((q, k and v), n_samples, n_heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # for each q, k and v tensors : (n_samples, n_heads, head_dim)
        v_sparsity = sparsity_measure(v) 
        self.K[:,:,-1] = k # Store the value of K for the present token
        self.V[:,:,-1] = v # Store the value of V for the present token
        q = q.unsqueeze(2)        
        attention = q*self.K.clone()
        attention_before_sum_sparsity = sparsity_measure(attention)
        attention = torch.sum(attention, dim=-1)
        attention_sparsity = sparsity_measure(attention)
        mean_attention = no_grad_mean(attention) 
        attention *= self.scale       
        attention = self.softmax_procedure(attention) # Softmax with zero values replaced by -Inf 
        attention = attention.unsqueeze(-1)
        weighted_average = attention*self.V.clone()
        weighted_average_before_sum_sparsity = sparsity_measure(weighted_average)
        weighted_average = torch.sum(weighted_average, dim=2)
        weighted_average_sparsity = sparsity_measure(weighted_average)
        mean_weighted_average = no_grad_mean(weighted_average)
        weighted_average = weighted_average.flatten(1) # (n_samples, n_heads*head_dim=embed_dim) flattens only above dimension 1 #
        weighted_average = self.proj(weighted_average) # (n_samples, head_embed_dim)
        weighted_average = self.proj_drop(weighted_average)
        return weighted_average, torch.stack([embedding_sparsity, v_sparsity, attention_before_sum_sparsity, attention_sparsity, mean_attention, weighted_average_before_sum_sparsity, weighted_average_sparsity, mean_weighted_average]), None
    
    def zeros_KV(self, n_samples):
        self.K = 0*self.K[0].detach()
        self.K = self.K.unsqueeze(0).expand(n_samples, -1,-1,-1)
        self.V = 0*self.V[0].detach()
        self.V = self.V.unsqueeze(0).expand(n_samples, -1,-1,-1)
        
    def detach_KV(self):
        self.K = self.K.detach()
        self.V = self.V.detach()
        
def no_grad_mean(x):
    with torch.no_grad():
        x = torch.mean(x.detach())
    return x        

def sparsity_measure(x): 
    with torch.no_grad():
        n_sample = x.shape[0]    
        count_nonzero = torch.tensor(0.).to(x.device)
        # The for loop is necessary here because torch.count_nonzero is not handled well for very large tensors -> we need to compute nonzero count in multiple steps
        for i in range(n_sample):
            count_nonzero += torch.count_nonzero(x[i].detach()).float()
        x = 1 - (count_nonzero/torch.numel(x.detach()))
    return x
    
class normal_sequential_softmax_procedure(nn.Module):
    """
    To prevent zero values to be included in softmax computation, this procedure replace zeros values with -Inf.
    Since this module is used in "non sparse" model, the zero value here correspond to future tokens: it is then similar to masking.
    """
    def __init__(self, args):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):       
        x = seek_zeros_replace_minus_inf(x)
        x = self.softmax(x)
        return x
    
def seek_zeros_replace_minus_inf(x):
    x[x==0] += float("-inf")
    return x

class ReshapedParallelAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.inner_dim = args.head_dim * args.n_heads
        self.scale = self.head_dim**-0.5
        self.atten_drop = nn.Dropout(args.atten_drop_p) if (not(args.spiking_qkv_proj) and not(args.binarize_qkv)) else nn.Identity()
        self.proj_drop = nn.Dropout(args.proj_drop_p) if (not(args.spiking_qkv_proj) and not(args.binarize_qkv)) else nn.Identity()
        self.qkv = nn.Linear(args.embed_dim, self.inner_dim * 3, bias=args.qkv_bias) if not(args.spiking_qkv_proj) else LIFNetwork_multi_steps(args, args.embed_dim, None, self.inner_dim * 3, readout_fn='S_readout')
        self.binarize_qkv = SmoothStep().apply if args.binarize_qkv else nn.Identity()
        self.proj = nn.Linear(self.inner_dim, args.embed_dim)      
        self.device = args.device
        self.softmax_procedure = normal_parallel_softmax_procedure(args) if (not(args.spiking_qkv_proj) and not(args.binarize_qkv) and not(args.spiking_embedding)) else sparse_softmax_procedure(args)
        self.unfold_kv = nn.Unfold((1, args.n_tokens), stride=(1,1), padding=(0,0))
        self.stored_vector_size = args.stored_vector_size

    def forward(self,x):
        n_samples, n_tokens, embed_dim = x.shape
        embedding_sparsity = sparsity_measure(x) 
        qkv = self.qkv(x) # (n_samples, n_tokens, 3*embed_dim) -> this is equivalent to make 3 different fully connected to tensors q,k and v
        qkv = self.binarize_qkv(qkv)         
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.inner_dim) 
        qkv = qkv.permute(2, 0, 3, 1) # qkv, nsamples, inner_dim, ntokens
        q, k, v = qkv[0], qkv[1], qkv[2] # q, k and v tensors
        v_sparsity = sparsity_measure(v)
        # Q reshape
        q = q.reshape(n_samples, self.n_heads, self.head_dim, n_tokens).transpose(2,3)  
        q = q.unsqueeze(3).expand(-1,-1,-1,self.stored_vector_size,-1)        
        # K and V reshape           
        k = torch.cat((torch.zeros(n_samples, self.inner_dim, self.stored_vector_size-1).to(self.device), k), dim=-1)
        v = torch.cat((torch.zeros(n_samples, self.inner_dim, self.stored_vector_size-1).to(self.device), v), dim=-1)        
        k = self.unfold_kv(k.unsqueeze(2)) # This unfold is for preparing K in shape of sliding windows for a sliding window attention
        v = self.unfold_kv(v.unsqueeze(2)) # This unfold is for preparing V in shape of sliding windows for a sliding window attention        
        k = k.reshape(n_samples, self.n_heads, self.head_dim, n_tokens, self.stored_vector_size)        
        v = v.reshape(n_samples, self.n_heads, self.head_dim, n_tokens, self.stored_vector_size)
        k = k.permute(0,1,3,4,2)
        v = v.permute(0,1,3,4,2)
        x = q * k
        del q, k
        attention_before_sum_sparsity = sparsity_measure(x)
        x = torch.sum(x, dim=-1)
        attention_sparsity = sparsity_measure(x)
        mean_attention = no_grad_mean(x)
        x *= self.scale 
        x = self.softmax_procedure(x)     
        x = x.unsqueeze(-1) * v
        del v
        weighted_average_before_sum_sparsity = sparsity_measure(x)    
        x = torch.sum(x, dim=-2)
        weighted_average_sparsity = sparsity_measure(x)
        mean_weighted_average = no_grad_mean(x)
        # concatenate attention heads
        x = x.transpose(1,2) # (n_samples, n_tokens, n_heads, head_dim)
        x = x.flatten(2) # (n_samples, n_tokens, n_heads*head_dim=embed_dim) flattens only above dimension 2 #
        x = self.proj(x) # (n_samples, n_tokens, head_embed_dim)
        x = self.proj_drop(x)
        return x, torch.stack([embedding_sparsity, v_sparsity, attention_before_sum_sparsity, attention_sparsity, mean_attention, weighted_average_before_sum_sparsity, weighted_average_sparsity, mean_weighted_average]), qkv
    
class sparse_softmax_procedure(nn.Module):
    """
    This softmax procedure, made for "sparse models", prevent the vectors where all elements are zero to be implemented by the softmax (to prevent NaN values).
    It also masks future tokens. 
    """
    def __init__(self, args):
        super().__init__()
        self.stored_vector_size = args.stored_vector_size
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        shape = x.shape
        non_zero_row_indexes = torch.any(x, -1).unsqueeze(-1).expand(shape)
        x[non_zero_row_indexes] = seek_zeros_replace_minus_inf(x[non_zero_row_indexes])
        softmax_attention = self.softmax(x.clone()) 
        x[non_zero_row_indexes] = softmax_attention[non_zero_row_indexes]
        return x
    
class normal_parallel_softmax_procedure(nn.Module):
    """
    To prevent zero values to be included in softmax computation, this procedure replace zeros values with -Inf.
    Since this module is used in "non sparse" model, the zero value here correspond to future tokens: it is then similar to masking.
    """
    def __init__(self, args):
        super().__init__()
        # mask_left_triangle masks future tokens. 
        self.mask_left_triangle = mask_left_triangle(args.n_heads, args.n_tokens, args.stored_vector_size, args.device)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):       
        x = self.mask_left_triangle(x)
        x = self.softmax(x)
        return x
    
class mask_left_triangle(nn.Module):
    def __init__(self, heads, n_tokens, stored_vector_size, device) -> None:
        super().__init__()
        self.mask = torch.zeros(1, 1, n_tokens, stored_vector_size).to(device)
        for i in range(n_tokens):
            for j in range(stored_vector_size):
                if i < (stored_vector_size-j)-1:
                    self.mask[:,:,i,j]=1
        self.mask[self.mask==1] = float('-inf')
    def forward(self, x):
        x += self.mask
        return x
    
class LinearizedAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.inner_dim = args.head_dim * args.n_heads
        self.scale = self.head_dim**-0.5 
        self.atten_drop = nn.Dropout(args.atten_drop_p) if (not(args.spiking_qkv_proj) and not(args.binarize_qkv)) else nn.Identity()
        self.proj_drop = nn.Dropout(args.proj_drop_p) if (not(args.spiking_qkv_proj) and not(args.binarize_qkv)) else nn.Identity()
        self.qkv = nn.Linear(args.embed_dim, self.inner_dim * 3, bias=args.qkv_bias) if not(args.spiking_qkv_proj) else LIFNetwork(args, args.embed_dim, None,  self.inner_dim * 3, readout_fn='S_readout')
        self.proj = nn.Linear(self.inner_dim, args.embed_dim)        
        self.softmax_procedure = normal_sequential_softmax_procedure(args) if (not(args.spiking_qkv_proj) and not(args.binarize_qkv) and not(args.spiking_embedding)) else sparse_softmax_procedure(args)
        self.binarize_qkv = SmoothStep().apply if args.binarize_qkv else nn.Identity()
        self.device = args.device
        self.stored_vector_size = args.stored_vector_size
        self.KV = torch.zeros(1, self.n_heads, args.stored_vector_size).to(self.device) 
        self.TopSum = torch.zeros(1, self.n_heads).to(self.device) 
        self.K = torch.zeros(1, self.n_heads, self.head_dim, args.stored_vector_size).to(self.device)
        self.V = torch.zeros(1, self.n_heads, self.head_dim, args.stored_vector_size).to(self.device)
        self.BottomSum = torch.zeros(1, self.n_heads, self.head_dim).to(self.device) 

    def forward(self, x):
        n_samples, embed_dim = x.shape    
        
        qkv = self.qkv(x) # (n_samples, 3*embed_dim) -> this is equivalent to make 3 different fully connected to tensors q,k and v
        qkv = self.binarize_qkv(qkv)
        qkv = qkv.expand(n_samples, -1).reshape(n_samples, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3) # ((q, k and v), n_samples, n_heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # for each q, k and v tensors : (n_samples, n_heads, head_dim)
        
        KV_product = torch.sum(k.clone()*v.clone(), dim=-1) 
        self.TopSum = self.TopSum - self.KV[:,:,0]
        self.BottomSum = self.BottomSum - self.K[:,:,:,0]
        self.KV = torch.roll(self.KV, -1, dims=-1)
        self.KV[:,:,-1] = KV_product  
        self.K = torch.roll(self.K, -1, dims=-1)
        self.K[:,:,:,-1] = k        
        self.TopSum += KV_product    
        self.BottomSum += k 
        attention = q * self.TopSum.unsqueeze(-1)
        attention /= (torch.sum(q * self.BottomSum, dim=-1).unsqueeze(-1) + 1e-10)

        attention = attention.flatten(1) # (n_samples, n_heads*head_dim=embed_dim) flattens only above dimension 1 #
        attention = self.proj(attention) # (n_samples, head_embed_dim)
        attention = self.proj_drop(attention)
        return attention, torch.tensor([0,0,0,0,0,0,0,0]).to(self.device), None
    
    def zeros_KV(self, n_samples):
        self.KV = 0*self.KV[0].detach()
        self.KV = self.KV.unsqueeze(0).expand(n_samples, -1,-1)
        self.K = 0*self.K[0].detach()
        self.K = self.K.unsqueeze(0).expand(n_samples, -1,-1,-1)
        self.TopSum = 0*self.TopSum[0].detach()
        self.TopSum = self.TopSum.unsqueeze(0).expand(n_samples, -1)
        self.BottomSum = 0*self.BottomSum[0].detach()
        self.BottomSum = self.BottomSum.unsqueeze(0).expand(n_samples, -1,-1)
        
    def detach_KV(self):
        self.KV = self.KV.detach()
        self.K = self.K.detach()
        self.TopSum = self.TopSum.detach()
        self.BottomSum = self.BottomSum.detach()

class LinearizedReshapedParallelAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.inner_dim = args.head_dim * args.n_heads
        self.scale = self.head_dim**-0.5
        self.atten_drop = nn.Dropout(args.atten_drop_p) if (not(args.spiking_qkv_proj) and not(args.binarize_qkv)) else nn.Identity()
        self.proj_drop = nn.Dropout(args.proj_drop_p) if (not(args.spiking_qkv_proj) and not(args.binarize_qkv)) else nn.Identity()
        self.qkv = nn.Linear(args.embed_dim, self.inner_dim * 3, bias=args.qkv_bias) if not(args.spiking_qkv_proj) else LIFNetwork_multi_steps(args, args.embed_dim, None, self.inner_dim * 3, readout_fn='S_readout')
        self.binarize_qkv = SmoothStep().apply if args.binarize_qkv else nn.Identity()
        self.proj = nn.Linear(self.inner_dim, args.embed_dim)      
        self.device = args.device
        self.softmax_procedure = normal_parallel_softmax_procedure(args) if (not(args.spiking_qkv_proj) and not(args.binarize_qkv) and not(args.spiking_embedding)) else sparse_softmax_procedure(args)
        self.unfold_kv = nn.Unfold((1, args.n_tokens), stride=(1,1), padding=(0,0))
        self.stored_vector_size = args.stored_vector_size

    def forward(self,x):
        n_samples, n_tokens, embed_dim = x.shape
        embedding_sparsity = sparsity_measure(x) 
        qkv = self.qkv(x) # (n_samples, n_tokens, 3*embed_dim) -> this is equivalent to make 3 different fully connected to tensors q,k and v
        qkv = self.binarize_qkv(qkv)
        qkv_sparsity = sparsity_measure(qkv) 
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.inner_dim) 
        qkv = qkv.permute(2, 0, 3, 1) # qkv, nsamples, inner_dim, ntokens
        q, k, v = qkv[0], qkv[1], qkv[2] # q, k and v tensors
        # Q reshape
        q = q.reshape(n_samples, self.n_heads, self.head_dim, n_tokens).transpose(2,3)  
        q = q.unsqueeze(3).expand(-1,-1,-1,self.stored_vector_size,-1)        
        # K and V reshape           
        k = torch.cat((torch.zeros(n_samples, self.inner_dim, self.stored_vector_size-1).to(self.device), k), dim=-1)
        v = torch.cat((torch.zeros(n_samples, self.inner_dim, self.stored_vector_size-1).to(self.device), v), dim=-1)        
        k = self.unfold_kv(k.unsqueeze(2))
        v = self.unfold_kv(v.unsqueeze(2))        
        k = k.reshape(n_samples, self.n_heads, self.head_dim, n_tokens, self.stored_vector_size)        
        v = v.reshape(n_samples, self.n_heads, self.head_dim, n_tokens, self.stored_vector_size)
        k = k.permute(0,1,3,4,2)
        v = v.permute(0,1,3,4,2)
        x = self.softmax_procedure(q.clone()) * k * v * self.scale
        x = torch.sum(x, dim=3)

        x = x.transpose(1,2) 
        x = x.flatten(2) 
        x = self.proj(x) 
        x = self.proj_drop(x)
        return x, torch.stack([embedding_sparsity, qkv_sparsity, torch.tensor(0).to(self.device), torch.tensor(0).to(self.device), torch.tensor(0).to(self.device), torch.tensor(0).to(self.device), torch.tensor(0).to(self.device), torch.tensor(0).to(self.device)]), qkv
    
    
class MLP(nn.Module):
    def __init__(self, args, dim_in, hidden, dim_out, mlp_drop_p=.0, NL="GELU"):
        super().__init__()
        N = [dim_in]+[h for h in hidden]+[dim_out] if isinstance(hidden, list) else [dim_in, hidden, dim_out] if hidden is not None else [dim_in, dim_out]
        self.N=N
        layers = []
        for i in range(len(N)-2):
            layers.append(nn.ModuleList(
                [
                    nn.Linear(N[i],N[i+1]),
                    getattr(nn, NL)(),
                    nn.Dropout(mlp_drop_p),
                ]
            ))
        layers.append(nn.ModuleList(
                [
                    nn.Linear(N[-2],N[-1]),
                ]
            ))
        self.layers = nn.ModuleList(layers)
    
    def forward(self,x):
        for layer in self.layers:
            for layer_item in layer:
                x = layer_item(x) 
        return x

    def init_state(self):
        return
    
    def detach_LIF_states(self):
        return

class SmoothStep(torch.autograd.Function):
    """
    Here, we define a surrogate gradient for the Heaviside step function.
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """
    @staticmethod
    def forward(aux, x):
        aux.save_for_backward(x)
        return (x >= 0).float() # Behavior similar to Heaviside step function

    def backward(aux, grad_output): # Define the behavior for the backward pass 
        beta = 10.0
        input, = aux.saved_tensors
        surrogate = 1.0/(beta*torch.abs(input) + 1.0)**2
        grad_input = grad_output.clone() * surrogate        
        return grad_input

class LIFNetwork(nn.Module):    
    def __init__(self, args, dim_in, hidden, dim_out, alpha=.95, beta=.9, betar=.85, theta=1.0, readout_fn='S_readout'):
        """
        Function to initialize a spiking neural network consisting of multiple 
        layers of spiking neural networks.
        """
        self.device=args.device
        alpha=args.alpha
        beta=args.beta
        betar=args.betar
        super().__init__()        
        N = [dim_in]+[h for h in hidden]+[dim_out] if isinstance(hidden, list) else [dim_in, hidden, dim_out] if hidden is not None else [dim_in, dim_out]
        self.N=N
        layers = []
        for i in range(len(N)-1):
            layers.append(LIFDensePopulation(
                                             in_channels=N[i],
                                             out_channels=N[i+1],
                                             alpha=alpha,
                                             beta=beta,
                                             betar=betar,
                                             theta=theta,
                                             recurrent_layer=args.reccurent_LIF,
                                             )
                        )
        self.layers = nn.ModuleList(layers)
        self.step = LIF_step
        self.N = N
        self.readout_fn = getattr(LIFNetwork, readout_fn)

    def forward(self, Sin):
        state = self.step(self.layers, Sin)         
        return self.readout_fn(state)

    def S_readout(state):
        return state.S

    def U_readout(state):
        return state.U

    def init_state(self):
        for layer in self.layers:
            layer.init_state()  
            
    def detach_LIF_states(self):
        for layer in self.layers:
            for key in layer.state.state_names:
                state = getattr(layer.state, key)
                setattr(layer.state, key, state.detach())
                
class LIFNetwork_multi_steps(nn.Module):    
    def __init__(self, args, dim_in, hidden, dim_out, alpha=.95, beta=.9, betar=.85, theta=1.0, readout_fn='S_readout'):
        """
        Function to initialize a spiking neural network consisting of multiple 
        layers of spiking neural networks.
        """
        self.device=args.device
        alpha=args.alpha
        beta=args.beta
        betar=args.betar
        super().__init__()        
        N = [dim_in]+[h for h in hidden]+[dim_out] if isinstance(hidden, list) else [dim_in, hidden, dim_out] if hidden is not None else [dim_in, dim_out]
        self.N=N
        layers = []
        for i in range(len(N)-1):
            layers.append(LIFDensePopulation(
                                            in_channels=N[i],
                                            out_channels=N[i+1],
                                            alpha=alpha,
                                            beta=beta,
                                            betar=betar,
                                            theta=theta,
                                            recurrent_layer=args.reccurent_LIF,
                                            )
                        )
        self.layers = nn.ModuleList(layers)
        self.step = LIF_step
        self.N = N
        self.readout_fn = getattr(LIFNetwork, readout_fn)

    def forward(self, Sin):
        n_samples, time, _ = Sin.shape
        output = torch.zeros(n_samples, time, self.N[-1]).to(self.device)
        for i in range(time):
            state = self.step(self.layers, Sin[:,i])  
            output[:,i] = self.readout_fn(state)       
        return output

    def S_readout(state):
        return state.S

    def U_readout(state):
        return state.U

    def init_state(self):
        for layer in self.layers:
            layer.init_state()  
            
    def detach_LIF_states(self):
        for layer in self.layers:
            for key in layer.state.state_names:
                state = getattr(layer.state, key)
                setattr(layer.state, key, state.detach())

class LIFDenseNeuronState(nn.Module):
    """
    Generic module for storing the state of an RNN/SNN.
    We use the buffer function of torch nn.Module to register our
    different states such that PyTorch can manage them.
    """
    def __init__(self, in_channels, out_channels):
        """Simple initialization of the internal states of a LIF population."""
        super(LIFDenseNeuronState, self).__init__()
        self.state_names = ['U', 'I', 'Ir', 'S']
        self.register_buffer('U', torch.zeros(1, out_channels), persistent=True)
        self.register_buffer('I', torch.zeros(1, out_channels), persistent=True)
        self.register_buffer('Ir', torch.zeros(1, out_channels), persistent=True)
        self.register_buffer('S', torch.zeros(1, out_channels), persistent=True)
                                                    
    def update(self, **values):
        """Function to update the internal states."""
        for k, v in values.items():
            setattr(self, k, v) 
    
    def init(self, v=0): 
        """Function that detaches the state/graph across trials."""
        for k in self.state_names:
            state = getattr(self, k)
            state_dims = list(state.shape)
            setattr(self, k, torch.zeros_like(state)[0].reshape(tuple([1]+state_dims[1:]))+v)
            state = getattr(self, k).detach()
            setattr(self, k, state)

class LIFDensePopulation(nn.Module):    
    def __init__(self, in_channels, out_channels, alpha = .95, beta=.9, betar=.85, theta=1.0, recurrent_layer=True):
        super(LIFDensePopulation, self).__init__()       
        """
        Function to initialize a layer of leaky integrate-and-fire neurons.
        """
        self.fwd_layer = nn.Linear(in_channels, out_channels) # Used to store feed-forward weights
        self.rec_layer = nn.Linear(out_channels, out_channels, bias=False) if recurrent_layer else nn.Identity() # Used to store recurrent weights 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha # Controls decay of membrane potential
        self.beta = beta # Controls decay of feed-forward input current
        self.betar = betar # Controls decay of recurrent input current
        self.theta = theta
        self.state = LIFDenseNeuronState(in_channels, out_channels)
        self.recurrent_layer = recurrent_layer
        self.fwd_layer.weight.data.uniform_(-.3, .3) # Initialization of feed-forward layer
        self.rec_layer.weight.data.uniform_(-.0, .0) if recurrent_layer else None # Recurrent layer is initialized with zero weights, i.e. ignored
        self.fwd_layer.bias.data.uniform_(-.01, .01) # Initialization of a slight bias current for the fc layer

        self.smooth_step = SmoothStep().apply

    def forward(self, Sin_t):
        """Forward pass of a batch through the data."""
        state = self.state     
        U = self.alpha*(1-state.S.detach())*state.U + (1-self.alpha)*(state.I+state.Ir) # I is weighted by a factor of 20
        I = self.beta*state.I + (1-self.beta)*self.fwd_layer(Sin_t)
        Ir = self.betar*state.Ir + (1-self.betar)*self.rec_layer(state.S)
        S = self.smooth_step(state.U-self.theta)
        # Update the neuronal state
        self.state.update(U=U, I=I, Ir=Ir, S=S)
        return self.state
    
    def init_state(self, value=0):
        "Initialize the state variables of this layer."
        self.state.init(value)

def LIF_step(layers, Sin_t):    
    """
    Perform a single batched forward timestep through the network.
    """
    for layer in layers:
        Sin_t = layer(Sin_t).S
    return layer.state # Returns final state of last layer  

### Data-driven initialization of the SNN
def get_LIF_layers_list(model):
    layer_list = []    
    # Embedding part
    if hasattr(model, 'PatchEmbed'):
        if hasattr(model.PatchEmbed.Proj, 'layers'):
            for layer in model.PatchEmbed.Proj.layers:
                layer_list.append(layer)
    # Transformer blocks QKV projection part
    if hasattr(model, 'blocks'):            
        for block in model.blocks:
            if hasattr(block.atten.qkv, 'layers'):
                for layer in block.atten.qkv.layers:
                    layer_list.append(layer)
    # Transformer blocks MLP part
    if hasattr(model, 'blocks'):
        if isinstance(model.blocks[0].MLP_layers.layers[0], LIFDensePopulation):
            for block in model.blocks:
                for layer in block.MLP_layers.layers:
                    layer_list.append(layer)
    # Classifier part
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier.layers[0], LIFDensePopulation):
            for layer in model.classifier.layers:
                layer_list.append(layer)
    layer_ids = list(range(len(layer_list)))
    return layer_ids, layer_list

from torch.nn import init
def forward_layer(model, layer_list, x, y=0):
    layer_state = []
    _ = model(x, y)
    for layer in layer_list:
        layer_state.append(layer.state.cpu())
    return layer_state # Returns final state of last layer

def torch_init_LSUV(model, data_batch, y=0, tgt_mu=-.85, tgt_var=1.0):
    '''
    Initialization inspired from Mishkin D and Matas J. All you need is a good init. arXiv:1511.06422 [cs],
February 2016.
    '''
    device = data_batch.device
    ##Initialize
    layer_ids, layer_list = get_LIF_layers_list(model)

    with torch.no_grad():
        for layer in layer_list:
                if layer.fwd_layer.bias is not None:
                    layer.fwd_layer.bias.data *= 0
                init.orthogonal_(layer.fwd_layer.weight)
                if hasattr(layer,'rec_layer'):
                    if layer.recurrent_layer:
                        if layer.rec_layer.bias is not None:
                            layer.rec_layer.bias.data *= 0
                        init.orthogonal_(layer.rec_layer.weight)
        alldone = False
        while not alldone:
            alldone = True
            layer_states = forward_layer(model, layer_list, data_batch, y=y)          
            for layer_num, layer, layer_state in zip(layer_ids, layer_list, layer_states):
                v = np.var(layer_state.U.flatten().cpu().numpy())
                m = np.mean(layer_state.U.flatten().cpu().numpy())
                mus = np.mean(layer_state.S.flatten().cpu().numpy())                
                print("Layer: {0}, Variance: {1:.3}, Mean U: {2:.3}, Mean S: {3:.3}".format(layer_num,v,m,mus))
                if np.isnan(v) or np.isnan(m):
                    print('Nan encountered during init')
                    done = False
                    raise
                if np.abs(v-tgt_var)>.1:
                    layer.fwd_layer.weight.data /= np.sqrt(np.maximum(v,1e-2))                  
                    layer.fwd_layer.weight.data *= np.sqrt(tgt_var)
                    done=False
                else:
                    done=True
                alldone*=done
                    
                if np.abs(m-tgt_mu)>.1:
                    if layer.fwd_layer.bias is not None:
                        layer.fwd_layer.bias.data -= .2*(m-tgt_mu) 
                    done=False
                else:
                    done=True
                alldone*=done
                model = model.to(device)
            if alldone and len(layer_list)>0:
                print("Initialization finalized:")
                print("Layer: {0}, Variance: {1:.3}, Mean U: {2:.3}, Mean S: {3:.3}".format(layer_num,v,m,mus))
    
class transformer(nn.Module):
    def __init__(self, args, dim_in, dim_out):
        super().__init__()       
        self.device = args.device 
        self.n_tokens = args.n_tokens        
        self.in_channels = dim_in         
        self.window_size = args.window_size
        if args.spiking_embedding:
            self.input_reshape = no_overlap_reshape(args, dim_in)
            self.pos_embed = nn.Identity()   
            self.pos_drop = nn.Dropout(p=0)  
            self.upsampling_factor = self.patch_size 
            self.PatchEmbed = getattr(continuous_model, "LIFPatchEmbed")(args, in_chans=dim_in)
        else:
            self.input_reshape = overlap_reshape(dim_in, self.window_size, self.n_tokens, args.conv_kernel_size, args.conv_stride, self.device)         
            self.pos_embed = nn.Identity()   
            self.pos_drop = nn.Dropout(p=args.pos_drop_p)  
            self.upsampling_factor = self.input_reshape.stride
            self.PatchEmbed = getattr(continuous_model, "ConvPatchEmbed")(args, in_chans=dim_in)
            
        self.blocks = nn.ModuleList(
            [
                transformer_block_encoder(args, continuous=False)
                for _ in range(args.transfo_depth)
            ]
        )  
        self.norm = nn.LayerNorm(args.embed_dim) 
        self.NL =  getattr(nn, args.end_NL)()                      
        self.classifier = MLP(args, args.embed_dim, args.classifier_hidden, dim_out) if not(args.spiking_classifier) else LIFNetwork(args, args.embed_dim, args.classifier_hidden, dim_out, readout_fn='U_readout')     

    def forward(self, x, target=0):
        n_samples = x.shape[0]
        x = self.input_reshape(x)
        x = self.PatchEmbed(x)
        x = self.pos_embed(x)
        x = self.pos_drop(x)
        x, sparsity, qkv = self.transformer_step(self.blocks, x)
        x = self.norm(x)
        x = self.NL(x)
        x = self.classifier(x)
        x = x.transpose(1,2)
        x = x.unsqueeze(-1).expand(-1,-1,-1,self.upsampling_factor)
        x = x.flatten(2)
        return x, sparsity, qkv
    
    def transformer_step(self, blocks, x):   
        sparsity_tensor = torch.zeros(len(blocks), 9).to(x.device)
        for b, block in enumerate(blocks): 
            x, sparsity_attention, qkv, sparsity_mlp = block(x)
            sparsity_tensor[b] = torch.cat([sparsity_attention, sparsity_mlp.unsqueeze(0)])
        return x, torch.mean(sparsity_tensor, dim=0), qkv
    
    def init_epoch(self, x):
        return    
    def init_batch(self):
        return
    
class LinearPatchEmbed(nn.Module):
    def __init__(self, args, window_size, patch_size, in_chans=14, embed_dim=64):
        super().__init__()
        self.n_tokens = args.n_tokens
        self.window_size = window_size
        self.patch_size = patch_size
        self.Proj = nn.Linear(in_chans*patch_size, embed_dim)

    def forward(self,x):
        n_sample, in_channels, n_tokens, time = x.shape
        x = x.transpose(1,2)
        x = x.reshape(n_sample, n_tokens, in_channels * time) # returns x with shape (n_samples, in_channels*patch_size)
        x = self.Proj(x)  # returns x with shape (n_samples, n_tokens, embeded_dim)
        return x
    
class ParallelAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.inner_dim = args.head_dim * args.n_heads
        self.scale = self.head_dim**-0.5
        self.atten_drop, self.proj_drop = nn.Dropout(args.atten_drop_p), nn.Dropout(args.proj_drop_p)
        self.qkv = nn.Linear(args.embed_dim, self.inner_dim * 3, bias=args.qkv_bias)
        self.proj = nn.Linear(self.inner_dim, args.embed_dim)      
        self.softmax = nn.Softmax(dim=-1)
        self.device = args.device

    def forward(self,x):
        n_samples, n_tokens, embed_dim = x.shape
        qkv = self.qkv(x) # (n_samples, n_tokens, 3*embed_dim) -> this is equivalent to make 3 different fully connected to tensors q,k and v
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # ((q, k and v), n_samples, n_heads, n_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # for each q, k and v tensors : (n_samples, n_heads, n_tokens, head_dim)
        k_t = k.transpose(2,3) # (n_samples, n_heads, head_dim, n_tokens)
        dot_prod = (q @ k_t)# (n_samples, n_heads, n_tokens, n_tokens)
        dot_prod *= self.scale # (n_samples, n_heads, n_tokens, n_tokens) 
        dot_prod = self.softmax(dot_prod)    
        attention = self.atten_drop(dot_prod)  
        weighted_average = attention @ v # (n_samples, n_heads, n_tokens, head_dim)
        # concatenate attention heads
        weighted_average = weighted_average.transpose(1,2) # (n_samples, n_tokens, n_heads, head_dim)
        weighted_average = weighted_average.flatten(2) # (n_samples, n_tokens, n_heads*head_dim=embed_dim) flattens only above dimension 2 #
        x = self.proj(weighted_average) # (n_samples, n_tokens, head_embed_dim)
        x = self.proj_drop(x)
        return x, torch.zeros(8).to(x.device), None
    
class ParallelCrossAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.inner_dim = args.head_dim * args.n_heads
        self.scale = self.head_dim**-0.5
        self.atten_drop, self.proj_drop = nn.Dropout(args.atten_drop_p), nn.Dropout(args.proj_drop_p)
        self.q_proj = nn.Linear(args.embed_dim, self.inner_dim, bias=args.qkv_bias)
        self.k_proj = nn.Linear(args.embed_dim, self.inner_dim, bias=args.qkv_bias)
        self.v_proj = nn.Linear(args.embed_dim, self.inner_dim, bias=args.qkv_bias)
        self.proj = nn.Linear(self.inner_dim, args.embed_dim)      
        self.softmax = nn.Softmax(dim=-1)        

    def forward(self, x, encoder_out):
        n_samples, n_tokens, embed_dim = x.shape
        encoder_out_n_tokens = encoder_out.shape[1]
        
        q, k, v = self.q_proj(x), self.k_proj(encoder_out), self.v_proj(encoder_out) # (n_samples, embed_dim) 
        q, k, v = q.reshape(n_samples, n_tokens, self.n_heads, self.head_dim), k.reshape(n_samples, encoder_out_n_tokens, self.n_heads, self.head_dim), v.reshape(n_samples, encoder_out_n_tokens, self.n_heads, self.head_dim)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2) # Transpose head dimension and token dimension
        k_t = k.transpose(2,3) # (n_samples, n_heads, head_dim, n_tokens)
        dot_prod = (q @ k_t)# (n_samples, n_heads, n_tokens, n_tokens)
        dot_prod *= self.scale # (n_samples, n_heads, n_tokens, n_tokens)
        # dot_prod[dot_prod==0] += float("-inf") 
        dot_prod = self.softmax(dot_prod)    
        attention = self.atten_drop(dot_prod)  
        weighted_average = attention @ v # (n_samples, n_heads, n_tokens, head_dim)
        # concatenate attention heads
        weighted_average = weighted_average.transpose(1,2) # (n_samples, n_tokens, n_heads, head_dim)
        weighted_average = weighted_average.flatten(2) # (n_samples, n_tokens, n_heads*head_dim=embed_dim) flattens only above dimension 2 #
        x = self.proj(weighted_average) # (n_samples, n_tokens, head_embed_dim)
        x = self.proj_drop(x)
        return x

class ContinuousLinearPatchEmbed(nn.Module):
    def __init__(self, args, window_size, patch_size, in_chans=14, embed_dim=64):
        super().__init__()
        self.n_tokens = args.n_tokens
        self.window_size = window_size
        self.patch_size = patch_size
        self.Proj = nn.Linear(in_chans*patch_size, embed_dim)

    def forward(self,x):
        n_sample, in_channels, time = x.shape
        x = x.reshape(n_sample, in_channels * time) # returns x with shape (n_samples, in_channels*patch_size)
        x = self.Proj(x)  # returns x with shape (n_samples, embeded_dim)
        return x

    def init_state(self):
        return
    
    def detach_LIF_states(self):
        return
    
class no_overlap_reshape(nn.Module):
    """
    Reshape the input into non-overlaping time windows to prepare for Linear embedding of Spiking embedding
    """ 
    def __init__(self, args, in_channels):
        super().__init__()
        assert args.window_size % args.n_tokens == 0
        self.patch_size = args.window_size // args.n_tokens
        self.in_channels = in_channels
        self.n_tokens = args.n_tokens
        self.upsampling_factor = self.patch_size
    def forward(self, x):
        n_samples = x.shape[0]
        return x.reshape(n_samples, self.in_channels, self.n_tokens, self.patch_size)

class ContinuousLIFPatchEmbed(nn.Module):
    def __init__(self, args, in_chans=16, embed_dim=64):
        super().__init__()
        self.Proj = LIFNetwork_multi_steps(args, in_chans, None, embed_dim, readout_fn='S_readout')

    def forward(self,x):
        return self.Proj(x.transpose(1,2))[:,-1]

    def init_state(self):
        self.Proj.init_state()
        return
    
    def detach_LIF_states(self):
        self.Proj.detach_LIF_states()   
        
class LIFPatchEmbed(nn.Module):
    ### This function gives the output in spikes rate ###
    def __init__(self, args, in_chans=14):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.Proj = LIFNetwork_multi_steps(args, in_chans, None, args.embed_dim, readout_fn='S_readout')
        self.device = args.device

    def forward(self,x):
        n_samples, in_channels, n_tokens, patch_size = x.shape
        output = torch.zeros(n_samples, n_tokens, self.embed_dim).to(self.device)
        for t in range(n_tokens):
            output[:,t] = self.Proj(x[:,:,t].transpose(1,2))[:,-1]
        return output

    def init_state(self):
        self.Proj.init_state()
        return
    
    def detach_LIF_states(self):
        self.Proj.detach_LIF_states()   
        
