import wandb
# import requests# 
import wget
import os

def del_previous_plots(epoch, run):
    if epoch>1:        
        api = wandb.Api(overrides={
                        "project": run.project,       
                        "entity": run.entity,
                            })
        run_adress = f'{run.entity}/{run.project}/{run.id}'
        run_api = api.run(run_adress)        
        list_plot_media = []
        for i, file in enumerate(run_api.files()):
            if file.name[0:12]=='media/plotly':             
                list_plot_media += [[i, file.name, file.updatedAt]]
                
        list_plot_media.sort(key=lambda x: x[2], reverse=False)
        if len(list_plot_media)>2:
            run_api.files()[list_plot_media[0][0]].delete()   
            run_api.files()[list_plot_media[1][0]].delete() 
        
def transfer_weights(train_model, test_model):
    train_model_state_dict = train_model.state_dict()
    # state of LIF neurons must be removed of dictonary because we don't want to transfer them
    for key in train_model.state_dict():
        if key[-7:]=="state.S" or key[-7:]=="state.U" or key[-7:]=="state.I" or key[-8:]=="state.Ir":
            del train_model_state_dict[key]
    test_model.load_state_dict(train_model_state_dict, strict=False)

def fix_LIF_states(model_to_load):
    if model_to_load is not None:
        for k, v in model_to_load.items():
            if k[-7:]=='state.U' or k[-7:]=='state.S' or k[-7:]=='state.I' or k[-8:]=='state.Ir':
                LIF_neurons_num = v.shape[-1]
                model_to_load[k] = v[0,:].reshape((1, LIF_neurons_num))
    return model_to_load

def download_dataset(root, name):
    url = 'http://ninapro.hevs.ch/files/DB8/'+name
    print(f'Downloading request on {url}, can take several minutes...')
    # r = requests.get(url, allow_redirects=True)
    os.makedirs(root, exist_ok=True)
    wget.download(url, out=root)
    # open(root+'/'+name, 'wb').write(r.content)
    