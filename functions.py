import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np 
import scipy.io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.saving import register_keras_serializable
#from keras import ops
import scipy
import h5py
from scipy import optimize

tf.keras.mixed_precision.set_global_policy('mixed_float16')

def read_config_line_from_file(line_number, filename="configs.txt"):
  with open(filename, "r") as f:
    lines = f.readlines()
    
  if line_number < 1 or line_number > len(lines):
    raise ValueError(f"Line {line_number} is out of range (1-{len(lines)})")

  line = lines[line_number - 1].strip()
  cfg = {}
  
  for entry in line.split(";"):
    key, value = entry.split("=", 1)
    key = key.strip()
    value = value.strip()
    
    if "." in value:
      value_cast = float(value)
    elif ',' in value:
        value_cast = [int(s) for s in value.split(',')]
    else:
      value_cast = int(value)
    
    cfg[key] = value_cast

  return cfg


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
  Uf = tf.math.sqrt(const_dict['alpha']*const_dict['g']*Lz*delta_T)
  P = const_dict['rho_0'] * (Uf**2)
  
  Pr = const_dict['visco'][0,0] / const_dict['kappa'][0,0]
  Ra = (const_dict['alpha'][0,0]*delta_T[0,0]*const_dict['g'][0,0]*Lz) / (const_dict['visco'][0,0] * const_dict['kappa'][0,0])
   
  return Uf[0,0], P[0,0], const_dict['T_bot'][0,0], const_dict['T_top'][0,0], np.array(Pr, np.float32), np.array(Ra, np.float32)
 
def nondim(U_pred, Uf, P, T_h, T_0):
  u, w, p, T = U_pred[...,0,tf.newaxis], U_pred[...,1,tf.newaxis], U_pred[...,2,tf.newaxis], U_pred[...,3,tf.newaxis]
  u, w = u/Uf, w/Uf
  p = p/P
  T = (T-T_0) / (T_h-T_0) - 0.5 
  return u, w, p, T

 
def load_data(train_size, val_size, Uf, P, T_h, T_0):
  data_dim, x, z, t = load_uwpT_long() 
  
  u, w, p, T = nondim(data_dim, Uf, P, T_h, T_0)
  data = np.concatenate((u, w, p, T), axis=-1)

  data_train = data[:train_size]
  data_val = data[train_size:(train_size+val_size)]
  
  return np.array(data_train, dtype=np.float32), np.array(data_val, dtype=np.float32), x, z, t
 
def get_augmentation_layer(image_size):
  return keras.Sequential([
      layers.Resizing(image_size, image_size),
      layers.RandomFlip("horizontal"),
      layers.RandomRotation(factor=0.02),
      layers.RandomZoom(height_factor=0.2, width_factor=0.2),], name="data_augmentation") 

def load_ae_data(train_size, val_size, batch_size, Uf, P, T_h, T_0): 

  data_train, data_val, _, _, _ = load_data(train_size, val_size, Uf, P, T_h, T_0)
  
  data_train_tf = tf.data.Dataset.from_tensor_slices((data_train,data_train))
  #data_train_tf = data_train_tf.map(lambda x: (augmentation_layer(x), x), num_parallel_calls=tf.data.AUTOTUNE)

  data_train_tf = data_train_tf.shuffle(buffer_size=train_size)
  data_train_tf = data_train_tf.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  
  data_val_tf = tf.data.Dataset.from_tensor_slices((data_val,data_val))
  data_val_tf = data_val_tf.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return data_train_tf, data_val_tf

def res_block(x, units, kernel_size, name):
  x2 = layers.Conv2D(units, kernel_size, padding='same', name=f'{name}_Conv1')(x)
  x2 = layers.BatchNormalization(name=f'{name}_BN1')(x2)
  x2 = layers.ReLU(name=f'{name}_ReLU')(x2)
  x2 = layers.Conv2D(units, kernel_size, padding='same', name=f'{name}_conv2')(x2)
  x2 = layers.BatchNormalization(name=f'{name}_BN2')(x2)
  return layers.Add(name=f'{name}_Add')([x, x2])
  
@register_keras_serializable()  
class PatchExtract(layers.Layer):
  def __init__(self, patch_size, name=None, **kwargs):
    super().__init__(**kwargs)
    self.patch_size = patch_size

  def call(self, images):
    input_shape = tf.shape(images)
    batch_size = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]
    num_patches_h = height // self.patch_size
    num_patches_w = width // self.patch_size
    
    patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID")
        
    num_patches = tf.shape(patches)[1] * tf.shape(patches)[2]
    patch_dim = self.patch_size * self.patch_size * tf.shape(images)[-1]
    
    return tf.reshape(patches, [batch_size, num_patches, patch_dim])

  def get_config(self):
    config = super().get_config()
    config.update({"patch_size": self.patch_size})
    return config

@register_keras_serializable()  
class PatchEmbedding(layers.Layer):
  def __init__(self, num_patches, projection_dim, name=None, **kwargs):
    super().__init__(name=name, **kwargs)
    self.num_patches = num_patches
    self.projection = layers.Dense(projection_dim, name=f"{name}_Projection")
    self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim, name=f"{name}_PosEmb")

  def call(self, patch):
    positions = tf.range(start=0, limit=self.num_patches, delta=1)
    return self.projection(patch) + self.position_embedding(positions)

def mlp(x, mlp_layers, drop, name):
  for i, units in enumerate(mlp_layers):
    x = layers.Dense(units, activation=keras.activations.gelu, name=f'{name}_DenseMLP_{i+1}')(x)
    x = layers.Dropout(drop, name=f'{name}_DropMLP_{i+1}')(x)
  return x
  

def transformer_encoder(inputs, num_heads, projection_dim, mlp_layers, drop, name):
  x1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_LN1")(inputs)
  attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=drop, name=f"{name}_MHA")(x1, x1)
  x2 = layers.Add(name=f"{name}_Add1")([attention_output, inputs])
  x3 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_LN2")(x2)
  x3 = mlp(x3, mlp_layers, drop, name=f"{name}")
  return layers.Add(name=f"{name}_Add2")([x3, x2])
    
  

def build_vit_cae_model(image_size, patch_size, proj_dim, num_heads, trsf_layers, trsf_units, enc_layers, drop, kernel_size):
  
  small_dims_enc = image_size // ( 2**(len(enc_layers)) )
  
  ### ENCODER ###
  inputs = layers.Input(shape=(image_size,image_size,4), name="Encoder_Input")
  
  # ViT Part
  num_patches = (image_size // patch_size) ** 2 
  patches = PatchExtract(patch_size, name="ViT_PatchExtract")(inputs)
  encoded_patches = PatchEmbedding(num_patches, proj_dim, name="ViT_PatchEmbedding")(patches)

  x_vit = encoded_patches
  for i in range(trsf_layers):
    x_vit = transformer_encoder(x_vit, num_heads, proj_dim, trsf_units, drop, name=f"ViT_Trsf_{i+1}")

  x_vit = layers.LayerNormalization(epsilon=1e-6, name="ViT_LN")(x_vit)
  
  if x_vit.shape[-1] > enc_layers[-1]:
    x_vit = layers.Dense((enc_layers[-1]+x_vit.shape[-1])//2, activation=activations.gelu, name='ViT_DenseHalf')(x_vit)
  x_vit = layers.Dense(enc_layers[-1], activation=activations.gelu, name='ViT_Dense')(x_vit)
  x_vit = layers.Reshape((image_size // patch_size, image_size // patch_size, enc_layers[-1]), name='ViT_Reshape')(x_vit)
  
  cur_shape = x_vit.shape[-2]
  i = 1
  while cur_shape != small_dims_enc:
    if cur_shape > small_dims_enc:
      x_vit = layers.Conv2D(enc_layers[-1], kernel_size=3, strides=2, padding='same', name=f'ViTConv_{i+1}')(x_vit)
    else:
      x_vit = layers.Conv2DTranspose(enc_layers[-1], kernel_size=3, strides=2, padding='same', name=f'ViTConv_{i+1}')(x_vit)
    x_vit = layers.LayerNormalization(name=f"ViTLN_{i+1}")(x_vit)
    x_vit = layers.Activation(activations.gelu, name=f'ViTACT_{i+1}')(x_vit)
    i += 1
    cur_shape = x_vit.shape[-2]
    
  
  # Conv Part
  x_conv = inputs
  for i, nodes in enumerate(enc_layers):
    x_conv = layers.Conv2D(nodes, kernel_size=3, strides=2, padding='same', name=f'EncConv_{i+1}')(x_conv)
    x_conv = layers.LayerNormalization(name=f"EncLN_{i+1}")(x_conv)
    x_conv = layers.Activation(activations.gelu, name=f'EncACT_{i+1}')(x_conv)
    
  # Fusion
  x = layers.Concatenate(name="Fusion")([x_vit, x_conv])
  x = layers.Conv2D(enc_layers[-1], kernel_size=3, padding='same', name=f'FusionConv')(x)
  x = layers.LayerNormalization(epsilon=1e-6, name="Fusion_LN")(x)
  x = layers.Activation(activations.gelu, name=f'FusionACT')(x)
  
  ###############
  
  ### DECODER ###
  
  outputs = []
  for v in ['u','w','p','T']:
    x_c = x
    for j, nodes in enumerate(enc_layers[::-1]):
      x_c = layers.Conv2DTranspose(nodes, kernel_size=kernel_size, strides=2, padding='same', name=f'DecConv_{j+1}_{v}')(x_c)
      x_c = layers.LayerNormalization(name=f'DecLN_{j+1}_{v}')(x_c)
      x_c = layers.Activation(activations.gelu, name=f'DecACT_{j+1}_{v}')(x_c)

    x_c = layers.Conv2D(1, kernel_size, activation='tanh', padding='same', dtype=tf.float32, name=v)(x_c)
    outputs.append(x_c)
  
  x = layers.Concatenate(axis=-1, name='uwpT', dtype=tf.float32)(outputs)
  autoencoder = keras.Model(inputs, x, name="Autoencoder")
  
  return autoencoder
  
  
def get_grads(x, z, const_dict, Uf):
  Lz = z[-1] - z[0]
  
  x = x / Lz
  z = z / Lz
  
  dx = x[2:] - x[:-2] 
  dz = z[2:] - z[:-2]
  
  dx = np.concatenate((x[1:2] - x[:1], dx, x[-2:-1] - x[-1:]))
  dz = np.concatenate((z[1:2] - z[:1], dz, z[-2:-1] - z[-1:]))
  
  dt = const_dict['plot_interval'][0,0] * Uf / Lz
  return tf.cast(dx, tf.float32), tf.cast(dz, tf.float32), tf.cast(dt, tf.float32)  
  
  
def DX(var, dx):
  dx = tf.reshape(dx, [1,dx.shape[0],1,1])
  ddx1 = var[...,1:2,:,:] - var[...,:1,:,:]
  ddx = var[...,2:,:,:] - var[...,:-2,:,:]
  ddx2 = var[...,-2:-1,:,:] - var[...,-1:,:,:]
  ddx = tf.concat([ddx1, ddx, ddx2], axis=-3)
  return tf.cast(ddx, tf.float32) / dx
  
def DZ(var, dz):
  dz = tf.reshape(dz, [1,1,dz.shape[0],1])
  ddz1 = var[...,:,1:2,:] - var[...,:,:1,:]
  ddz = var[...,:,2:,:] - var[...,:,:-2,:]
  ddz2 = var[...,:,-2:-1,:] - var[...,:,-1:,:]
  ddz = tf.concat([ddz1,ddz,ddz2], axis=-2)
  return tf.cast(ddz, tf.float32) / dz 
  

def DT(var, dt):
  ddt1 = (var[...,1:2,:,:,:] - var[...,:1,:,:,:]) / dt
  ddt = (var[...,2:,:,:,:] - var[...,:-2,:,:,:]) / (2*dt)
  ddt2 = (var[...,-2:-1,:,:,:] - var[...,-1:,:,:,:]) / (-dt)
  ddt = tf.concat([ddt1,ddt,ddt2], axis=-4)
  return ddt
 
 
def rmse(x_true, x_rec):
  rmse = np.sqrt(np.mean((x_true - x_rec)**2, axis=(1,2)))
  return rmse #(n, 4)
  

def mae(x_true, x_rec):
  mae = np.mean(np.abs(x_true - x_rec), axis=(1,2))
  return mae 
  

def power_spectrum(x_true, x_rec):
  n, _, _, C = x_true.shape
  pse = np.zeros((n, C))
  for c in range(x_true.shape[-1]):
    F_true = np.fft.fftn(x_true[..., c])
    F_rec  = np.fft.fftn(x_rec[..., c])

    P_true = np.abs(F_true)**2
    P_rec  = np.abs(F_rec)**2

    pse[:, c] = np.mean(np.abs(P_true - P_rec), axis=(1,2))

  return pse
  
  
def ssim(data, preds):
  c1, c2 = 1e-5, 1e-5
  
  mu = np.mean(data, axis=(1,2))
  mu_hat = np.mean(preds, axis=(1,2))
  sigma = np.std(data, axis=(1,2))
  sigma_hat = np.std(preds, axis=(1,2))
  
  data_centered = data - mu[:, None, None, :]
  preds_centered = preds - mu_hat[:, None, None, :]
  sigma_cross = np.mean(data_centered * preds_centered, axis=(1, 2))  # shape (n, 4)
  
  luminance = (2 * mu * mu_hat + c1) / (mu**2 + mu_hat**2 + c1)
  contrast = (2 * sigma * sigma_hat + c2) / (sigma**2 + sigma_hat**2 + c2)
  structure = (sigma_cross + c2/2) / (sigma * sigma_hat + c2/2)
  
  return luminance * contrast * structure

def ns_loss(U_pred, Pr, Ra, dx, dz, dt):
     
  U_pred_x = DX(U_pred, dx)
  U_pred_z = DZ(U_pred, dz)
  U_pred_t  = DT(U_pred, dt)
  U_pred_xx = DX(U_pred_x, dx)
  U_pred_zz = DZ(U_pred_z, dz)
  
  u, w, p, T = U_pred[...,0,tf.newaxis], U_pred[...,1,tf.newaxis], U_pred[...,2,tf.newaxis], U_pred[...,3,tf.newaxis]
  u_x, w_x, p_x, T_x = U_pred_x[...,0,tf.newaxis], U_pred_x[...,1,tf.newaxis], U_pred_x[...,2,tf.newaxis], U_pred_x[...,3,tf.newaxis]
  u_z, w_z, p_z, T_z = U_pred_z[...,0,tf.newaxis], U_pred_z[...,1,tf.newaxis], U_pred_z[...,2,tf.newaxis], U_pred_z[...,3,tf.newaxis]
  u_t, w_t, T_t = U_pred_t[...,0,tf.newaxis], U_pred_t[...,1,tf.newaxis], U_pred_t[...,3,tf.newaxis]
  u_xx, w_xx, T_xx = U_pred_xx[...,0,tf.newaxis], U_pred_xx[...,1,tf.newaxis], U_pred_xx[...,3,tf.newaxis]
  u_zz, w_zz, T_zz = U_pred_zz[...,0,tf.newaxis], U_pred_zz[...,1,tf.newaxis], U_pred_zz[...,3,tf.newaxis]

  mc = u_x + w_z  
  f_u = u_t + u*u_x + w*u_z + p_x - tf.math.sqrt(Pr/Ra)*(u_xx + u_zz)
  f_w = w_t + u*w_x + w*w_z + p_z - tf.math.sqrt(Pr/Ra)*(w_xx + w_zz) - T
  f_T = T_t + u*T_x + w*T_z - (T_xx + T_zz)/tf.math.sqrt(Pr*Ra)
  
  L_u = (f_u**2).numpy() 
  L_w = (f_w**2).numpy() 
  L_mc = (mc**2).numpy() 
  L_T = (f_T**2).numpy() 
  L_f = np.concatenate((L_u, L_w, L_mc, L_T), axis=-1)

  return L_f   

def print_results(metric, model):
  mu = metric.mean()
  med = np.median(metric)
  lo = np.quantile(metric, 0.25)
  hi = np.quantile(metric, 0.75)
  model += '    ' if 'VIT' not in model else '' 
  print(f'{model}. Mean: {mu:.2e}, Med: {med:.2e}, IQR: ({hi-lo:.2e})')
    


  
  
  
  
  
  
  
  
  
  
  

