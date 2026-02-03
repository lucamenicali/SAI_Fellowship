from functions_plot import *
import pickle

line = int(os.getenv('SGE_TASK_ID'))

with open(f'./results/preds_{line:02d}.pkl', 'rb') as f:
  results = pickle.load(f)

preds_vit = results['preds_vit']
preds_cae = results['preds_cae']
time_points = results['time_points']

const_dict = load_constants_long()
Uf, P, T_h, T_0, Pr, Ra = get_model_constants(const_dict)
_, data, x, z, _ = load_data(4000, 1000, Uf, P, T_h, T_0)

dx, dz, _ = get_grads(x, z, const_dict, Uf) 

data = data[-500:][time_points]
X, Z = np.meshgrid(x, z, indexing='ij')

plot_variable(name='T', time_points=time_points, data=data, preds_vit=preds_vit, preds_cae=preds_cae, X=X, Z=Z, line=line)

plot_res(name='T', time_points=time_points, data=data, preds_vit=preds_vit, preds_cae=preds_cae, X=X, Z=Z, line=line)

plot_mc(time_points=time_points, preds_vit=preds_vit, preds_cae=preds_cae, X=X, Z=Z, dx=dx, dz=dz, line=line)
