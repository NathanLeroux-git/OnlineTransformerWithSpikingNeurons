import numpy as np
# import time
# for i in range(1000):
#     print(i)
#     time.sleep(0.3)   
kB = 32/1024

kernel_size=7
embedding_size=64
inner_size=32
n_heads=8
stored_vector_size=150
hidden_units_size=128

# embedding
total_embedding_size = (16*kernel_size+1)*embedding_size
# attention
QKV_linear = n_heads*embedding_size*inner_size*3/16
total_stored_vector_size = stored_vector_size*2*inner_size*n_heads
concat_linear = n_heads*inner_size*embedding_size + embedding_size
total_MHA_size = QKV_linear+total_stored_vector_size+concat_linear
# MLP
MLP1_size = (embedding_size+1)*hidden_units_size#+ 2*hidden_units_size
MLP2_size = (hidden_units_size+1)*embedding_size# + 2*embedding_size
# Linear out
Linear_out = 5*(embedding_size+1)

total_size = total_embedding_size+total_MHA_size+MLP1_size+MLP2_size+Linear_out
print('\nTensors:\t\tSize in kB\n________________________________________')
print(f"Embedding size:\t\t{total_embedding_size*kB}\nQKV proj size:\t\t{QKV_linear*kB}\nKV size:\t\t{total_stored_vector_size*kB}\nconcat size:\t\t{concat_linear*kB}\nTotal MHA size:\t\t{total_MHA_size*kB}\nMLP1 size\t\t{concat_linear*kB}\nTotal MHA size:\t\t{MLP1_size*kB}\nMLP1 size\t\t{MLP1_size*kB}\nMLP2 size\t\t{MLP2_size*kB}\nClassifier size:\t{Linear_out*kB}\nTotal size:\t\t{total_size*kB}")
print(f"Total {total_size*32/(1024**2)} MB\n")
# Note that the stored model has only 512 kB, 10x less than what is computed

# Number of MACs per inference
# embedding
embedding_mac = (16*kernel_size+1)*embedding_size
# attention
qkv_proj_mac = n_heads*embedding_size*inner_size*3
qk_product_mac = n_heads*inner_size*stored_vector_size
v_product_mac = n_heads*stored_vector_size*inner_size
concat_linear_mac = n_heads*inner_size*embedding_size
total_attention_mac = qkv_proj_mac+qk_product_mac+v_product_mac+concat_linear_mac
# MLP
MLP1_mac = (embedding_size+1)*hidden_units_size# + 2*hidden_units_size
MLP2_mac = (hidden_units_size+1)*embedding_size# + 2*embedding_size
# Linear out
Linear_out_mac = 5*(embedding_size+1)

total_mac = embedding_mac+total_attention_mac+MLP1_mac+MLP2_mac+Linear_out_mac
MMACs = 32/1000/1000
print(f"Embedding MAC is  {embedding_mac*MMACs} MMACs")
print(f"qkv_proj_mac MAC is  {qkv_proj_mac*MMACs} MMACs")
print(f"qk_product_mac MAC is  {qk_product_mac*MMACs} MMACs")
print(f"v_product_mac MAC is  {v_product_mac*MMACs} MMACs")
print(f"concat_linear_mac MAC is  {concat_linear_mac*MMACs} MMACs")
print(f"Attention MAC is  {total_attention_mac*MMACs} MMACs")
print(f"MLP1_mac MAC is  {MLP1_mac*MMACs} MMACs")
print(f"MLP2_mac MAC is  {MLP2_mac*MMACs} MMACs")
print(f"Linear_out_mac MAC is  {Linear_out_mac*MMACs} MMACs")
print(f"Total number of MACs per inference is {total_mac*MMACs} MMACs\n")
# print(f"Total number of MACs per second is {total_mac*MMACs*200} MMACs/seconds\n")

# # qkv sparse, sparsity losss 10
# sparsity_embdding = 0
# mean_qk_product = 0.17
# sparsity_v = 0.96
# sparsity_v_product = 0.6673

# embedding + qkv sparse, sparsity losss 1
sparsity_embdding =  0.910  # 0
mean_qk_product = 3 # 0.17
sparsity_v = 0.73 # 0.96
sparsity_v_product = 0.112 # 0.6673

# # embedding + qkv sparse, sparsity losss 10
# sparsity_embdding = 0.889
# mean_qk_product = 1.5
# sparsity_v = 0.84
# sparsity_v_product = 0.496

sparse_qkv_proj_mac = (1-sparsity_embdding)*n_heads*embedding_size*inner_size*3
sparse_qk_product_mac = mean_qk_product*n_heads*stored_vector_size
sparse_v_product_mac = (1-sparsity_v) * n_heads*stored_vector_size*inner_size
sparse_concat_mac = (1-sparsity_v_product) * n_heads*inner_size*embedding_size
sparse_total_attention_mac = sparse_qkv_proj_mac+sparse_qk_product_mac+sparse_v_product_mac+sparse_concat_mac
sparse_total_mac = embedding_mac+sparse_total_attention_mac+MLP1_mac+MLP2_mac+Linear_out_mac

print(f"sparse qkv_proj_mac MAC is  {sparse_qkv_proj_mac*MMACs} MMACs")
print(f"sparse qk_product_mac MAC is  {sparse_qk_product_mac*MMACs} MMACs")
print(f"sparse v_product_mac MAC is  {sparse_v_product_mac*MMACs} MMACs")
print(f"sparse concat_linear_mac MAC is  {sparse_concat_mac*MMACs} MMACs")
print(f"sparse Attention MAC is  {sparse_total_attention_mac*MMACs} MMACs")
print(f"sparse Total number of MACs per inference is {sparse_total_mac*MMACs} MMACs\n")

# # test "Total size is {param_size} kB")TempoConv model, which is supposed to be only 70 kB
# import torch 
# from continuous_model import conv1d
# TempoNet = conv1d(None, 16, 5).to('cuda')
# param_size = 0
# print('\nTempo Net number of parameters per layer and size in kB:\n')
# for param in TempoNet.parameters():
#     param_size += param.nelement() # I don't understand how param.element_size() works* param.element_size()
#     print(f"{param.shape}\t\t\t{param.nelement()*8/1024}")

# print(f"Total number of parameters is {param_size}")
# param_size *= 8/1024 # they quantized to int8

# print(f