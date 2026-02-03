from functions import *
import numpy as np 
import time
import scipy
import h5py
import pickle

tf.keras.utils.set_random_seed(1)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'\nNumber of GPUs: {len(gpus)}')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

line = int(os.getenv('SGE_TASK_ID'))
suffix = f'{line:02d}'

n = 500
time_points = np.array([0, n//3, 2*n//3, n-1])
train_size, val_size = 4000, 1000

with tf.device('/GPU:0'):
  vit_cae = tf.keras.saving.load_model(f'./models/vit_cae_{suffix}.keras',
                                         custom_objects={'PatchExtract': PatchExtract, 'PatchEmbedding':PatchEmbedding})
                                         
  cae = tf.keras.saving.load_model(f'./models/ae_15.keras')

const_dict = load_constants_long()
Uf, P, T_h, T_0, Pr, Ra = get_model_constants(const_dict)
data_train, data_test, x, z, _ = load_data(train_size, val_size, Uf, P, T_h, T_0) 
dx, dz, dt = get_grads(x, z, const_dict, Uf)
  
#data_test = data_val[-n:]
data_test = tf.convert_to_tensor(data_test, dtype=tf.float32)

preds_vit = vit_cae.predict(data_test, verbose=0)
preds_cae = cae.predict(data_test, verbose=0)

mse_vit = (preds_vit-data_test)**2
mse_cae = (preds_cae-data_test)**2


print(f'\nRESULTS FOR {line:02d}')

print('\nRMSE')
rmse_vit = rmse(data_test, preds_vit)
rmse_cae = rmse(data_test, preds_cae)
print_results(rmse_vit, 'VIT-CAE')
print_results(rmse_cae, 'CAE')

print('\nMAE')
mae_vit = mae(data_test, preds_vit)
mae_cae = mae(data_test, preds_cae)
print_results(mae_vit, 'VIT-CAE')
print_results(mae_cae, 'CAE')

print('\nSSIM')
ssim_vit = ssim(data_test, preds_vit)
ssim_cae = ssim(data_test, preds_cae)
print_results(ssim_vit, 'VIT-CAE')
print_results(ssim_cae, 'CAE')


print(f'\nNavier Stokes')
ns_dat = ns_loss(data_test, Pr, Ra, dx, dz, dt)
ns_vit = ns_loss(preds_vit, Pr, Ra, dx, dz, dt)
ns_cae = ns_loss(preds_cae, Pr, Ra, dx, dz, dt)


for i, v in enumerate(['Mass Con.','Mom-u','Mom-w','Energy']):
  of_dat = ns_dat[...,i]
  of_vit = ns_vit[...,i]
  of_cae = ns_cae[...,i]
  
  print(f'\n{v}')
  print(f'Data   ; Mean: {np.mean(of_dat):.2e}, Med: {np.median(of_dat):.2e}, IQR: ({np.quantile(of_dat, 0.75)-np.quantile(of_dat, 0.25):.2e})')
  print(f'VIT-CAE; Mean: {np.mean(of_vit):.2e}, Med: {np.median(of_vit):.2e}, IQR: ({np.quantile(of_vit, 0.75)-np.quantile(of_vit, 0.25):.2e})')
  print(f'CAE    ; Mean: {np.mean(of_cae):.2e}, Med: {np.median(of_cae):.2e}, IQR: ({np.quantile(of_cae, 0.75)-np.quantile(of_cae, 0.25):.2e})')
















results = {'preds_vit':preds_vit[time_points+500], 'preds_cae':preds_cae[time_points+500], 'time_points':time_points}

with open(f'./results/preds_{line:02d}.pkl', 'wb') as f:
  pickle.dump(results, f)










