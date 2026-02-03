import os
import numpy as np 
import scipy.io
import scipy
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def load_xzt_long():
  with h5py.File('/afs/crc.nd.edu/group/RichterLab/for_Luca/RB_Dec18.mat', 'r') as mat_file:
    x = mat_file['xg'][:].flatten()
    z = mat_file['zg'][:].flatten()
    t = mat_file['time'][:].flatten()
  return x, z, t


def load_uwpT_long():
  with h5py.File('/afs/crc.nd.edu/group/RichterLab/for_Luca/RB_Dec18.mat', 'r') as mat_file:
    u = mat_file['u'][:]
    w = mat_file['w'][:]
    p = mat_file['p'][:]
    T = mat_file['t'][:]
  
  print(u.shape)
  uwpT = np.concatenate((u[...,None], w[...,None], p[...,None], T[...,None]), axis=-1)
  uwpT[...,-1] += 289.5
  uwpT = np.swapaxes(uwpT, 1, 2)
  x, z, t = load_xzt_long()
  return uwpT, x, z, t

 
def load_constants_long():
  const_dict = {}
  with h5py.File('/afs/crc.nd.edu/group/RichterLab/for_Luca/RB_Dec18.mat', 'r') as mat_file:
    for key, value in mat_file.items():
      if (len(value[:].flatten()) == 1):
        try:
          const_dict[key] = np.array(value, dtype=np.float32)
        except:
          continue
      
  return const_dict

def get_model_constants(const_dict):
  _, z, _ = load_xzt_long()
  Lz = z[-1] - z[0]
    
  delta_T = const_dict['T_bot']-const_dict['T_top']
  Uf = np.sqrt(const_dict['alpha']*const_dict['g']*Lz*delta_T)
  P = const_dict['rho_0'] * (Uf**2)
  
  Pr = const_dict['visco'][0,0] / const_dict['kappa'][0,0]
  Ra = (const_dict['alpha'][0,0]*delta_T[0,0]*const_dict['g'][0,0]*Lz) / (const_dict['visco'][0,0] * const_dict['kappa'][0,0])
   
  return Uf[0,0], P[0,0], const_dict['T_bot'][0,0], const_dict['T_top'][0,0], np.array(Pr, np.float32), np.array(Ra, np.float32)
 
def nondim(U_pred, Uf, P, T_h, T_0):
  u, w, p, T = U_pred[...,0,np.newaxis], U_pred[...,1,np.newaxis], U_pred[...,2,np.newaxis], U_pred[...,3,np.newaxis]
  u, w = u/Uf, w/Uf
  p = p/P
  T = (T-T_0) / (T_h-T_0) - 0.5 
  return u, w, p, T

def load_data(train_size, val_size, Uf, P, T_h, T_0):
  
  data_dim, x, z, t = load_uwpT_long() 

  u, w, p, T = nondim(data_dim, Uf, P, T_h, T_0)
  data = np.concatenate((u, w, p, T), axis=-1, dtype=np.float32)

  data_train = data[:train_size]
  data_val = data[train_size:(train_size+val_size)]
  
  return data_train, data_val, x, z, t

  
def get_grads(x, z, const_dict, Uf):
  Lz = z[-1] - z[0]
  
  x = x / Lz
  z = z / Lz
  
  dx = x[2:] - x[:-2] 
  dz = z[2:] - z[:-2]
  
  dx = np.concatenate((x[1:2] - x[:1], dx, x[-2:-1] - x[-1:]))
  dz = np.concatenate((z[1:2] - z[:1], dz, z[-2:-1] - z[-1:]))
  
  dt = const_dict['plot_interval'][0,0] * Uf / Lz
  return dx, dz, dt 
 
def DX(var, dx):
  dx = dx.reshape((1,dx.shape[0],1,1))
  ddx1 = var[...,1:2,:,:] - var[...,:1,:,:]
  ddx = var[...,2:,:,:] - var[...,:-2,:,:]
  ddx2 = var[...,-2:-1,:,:] - var[...,-1:,:,:]
  ddx = np.concatenate((ddx1, ddx, ddx2), axis=-3)
  return ddx / dx
  
def DZ(var, dz):
  dz = dz.reshape((1,1,dz.shape[0],1))
  ddz1 = var[...,:,1:2,:] - var[...,:,:1,:]
  ddz = var[...,:,2:,:] - var[...,:,:-2,:]
  ddz2 = var[...,:,-2:-1,:] - var[...,:,-1:,:]
  ddz = np.concatenate((ddz1,ddz,ddz2), axis=-2)
  return ddz / dz   

  
def plot_variable(name, time_points, data, preds_vit, preds_cae, X, Z, line):
  
  var = ['u','w','p','T']
  j = var.index(name)
  
  vmin = min(data[...,j].min(), preds_vit[...,j].min(), preds_cae[...,j].min())
  vmax = max(data[...,j].max(), preds_vit[...,j].max(), preds_cae[...,j].max())
  cmap = 'jet'
  fig, ax = plt.subplots(3, 4, figsize=(20,10), layout='constrained', sharex=True, sharey=True)

  for i, t in enumerate(time_points):
    ax[0,i].contourf(X, Z, data[i,...,j], cmap=cmap, levels=np.linspace(vmin,vmax,40))
    ax[1,i].contourf(X, Z, preds_vit[i,...,j], cmap=cmap, levels=np.linspace(vmin,vmax,40))
    ax[2,i].contourf(X, Z, preds_cae[i,...,j], cmap=cmap, levels=np.linspace(vmin,vmax,40))
    
    ax[0,i].set_title(f't = {(4500+t)*0.05:.2f}s', fontsize=15)

  ax[0,0].set_ylabel('DATA', fontsize=15)
  ax[1,0].set_ylabel('VIT-CAE', fontsize=15)
  ax[2,0].set_ylabel('CAE', fontsize=15)
  
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
  sm._A = []
  cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), orientation="horizontal", shrink=0.4, pad=0.02)
  ticks = np.linspace(vmin, vmax, 7)
  cbar.set_ticks(ticks)
  cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
  cbar.ax.tick_params(labelsize=12)

  fig.savefig(f'./figures/comparison_{line:02d}_{name}.png') 
  plt.close(fig)
  

def plot_res(name, time_points, data, preds_vit, preds_cae, X, Z, line):
  
  var = ['u','w','p','T']
  j = var.index(name)
  
  res_vit = preds_vit-data
  res_cae = preds_cae-data
  
  vmax = max(np.quantile(res_vit[...,j], .99), np.quantile(res_cae[...,j], .99))
  vmin = min(np.quantile(res_vit[...,j], .01), np.quantile(res_cae[...,j], .01))
  
  cmap = 'coolwarm'
  
  fig, ax = plt.subplots(2, 4, figsize=(20,7), layout='constrained', sharex=True, sharey=True)

  for i, t in enumerate(time_points):
    ax[0,i].contourf(X, Z, res_vit[i,...,j], cmap=cmap, levels=np.linspace(vmin,vmax,40), extend='both')
    ax[1,i].contourf(X, Z, res_cae[i,...,j], cmap=cmap, levels=np.linspace(vmin,vmax,40), extend='both')
    
    ax[0,i].set_title(f't = {(4500+t)*0.05:.2f}s', fontsize=15)

  ax[0,0].set_ylabel('VIT-CAE', fontsize=15)
  ax[1,0].set_ylabel('CAE', fontsize=15)
  
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
  sm._A = []
  cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), orientation="horizontal", extend='both', shrink=0.4, pad=0.02)
  ticks = np.linspace(vmin, vmax, 7)
  cbar.set_ticks(ticks)
  cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'))
  cbar.ax.tick_params(labelsize=12)

  fig.savefig(f'./figures/res_{line:02d}_{name}.png') 
  plt.close(fig)  
  

def plot_mc(time_points, preds_vit, preds_cae, X, Z, dx, dz, line):
  
  mc_vit = (DX(preds_vit[...,0:1], dx) + DZ(preds_vit[...,1:2], dz))**2
  mc_cae = (DX(preds_cae[...,0:1], dx) + DZ(preds_cae[...,1:2], dz))**2

  vmax = max(np.quantile(mc_vit, .95), np.quantile(mc_cae, .95))
  
  cmap = 'cubehelix_r'
  fig, ax = plt.subplots(2, 4, figsize=(20,10), layout='constrained', sharex=True, sharey=True)

  for i, t in enumerate(time_points):
    ax[0,i].contourf(X, Z, mc_vit[i,...,0], cmap=cmap, levels=np.linspace(0,vmax,40), extend='max')
    ax[1,i].contourf(X, Z, mc_cae[i,...,0], cmap=cmap, levels=np.linspace(0,vmax,40), extend='max')
    
    ax[0,i].set_title(f't = {(4500+t)*0.05:.2f}s', fontsize=15)

  ax[0,0].set_ylabel('VIT-CAE', fontsize=15)
  ax[1,0].set_ylabel('CAE', fontsize=15)
  
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
  sm._A = []
  cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), orientation="horizontal", extend='max', shrink=0.4, pad=0.02)
  ticks = np.linspace(0, vmax, 7)
  cbar.set_ticks(ticks)
  cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'))
  cbar.ax.tick_params(labelsize=12)

  fig.savefig(f'./figures/mc_{line:02d}.png') 
  plt.close(fig)  
  