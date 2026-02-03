from functions import *
import numpy as np 
import time
import scipy
import h5py

tf.keras.utils.set_random_seed(1)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'\nNumber of GPUs: {len(gpus)}')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

line = int(os.getenv('SGE_TASK_ID'))
suffix = f'{line:02d}'
cfg = read_config_line_from_file(line)

######################################################################################
## BUILD MODEL

image_size = 256
patch_size = cfg['patch_size']
projection_dim = cfg['projection_dim']
num_heads = cfg['num_heads']
transformer_layers = cfg['transformer_layers']
transformer_units = [2*projection_dim, projection_dim]
enc_layers = cfg['enc_layers']
drop = cfg['drop']
kernel_size = cfg['kernel_size']
batch_size = cfg['batch_size']

lr = 1e-3

with tf.device('/CPU:0'):
  train_size, val_size = 4000, 500 
  const_dict = load_constants_long()
  Uf, P, T_h, T_0, Pr, Ra = get_model_constants(const_dict)
  ae_train, ae_val = load_ae_data(train_size, val_size, batch_size, Uf, P, T_h, T_0)
  
  x, z, _ = load_xzt_long()
  dx, dz, _ = get_grads(x, z, const_dict, Uf)

with tf.device('/GPU:0'): 
  dx = tf.constant(tf.reshape(dx, [1,dx.shape[0],1,1]), dtype=tf.float32)
  dz = tf.constant(tf.reshape(dz, [1,1,dz.shape[0],1]), dtype=tf.float32)
    
  
  @tf.function(input_signature=[tf.TensorSpec(shape=[batch_size,256,256,1], dtype=tf.float32)])
  def DX_tf(var):
    ddx1 = var[...,1:2,:,:] - var[...,:1,:,:]
    ddx = var[...,2:,:,:] - var[...,:-2,:,:]
    ddx2 = var[...,-2:-1,:,:] - var[...,-1:,:,:]
    ddx = tf.concat([ddx1, ddx, ddx2], axis=-3)
    return tf.cast(ddx, tf.float32) / dx
  
  @tf.function(input_signature=[tf.TensorSpec(shape=[batch_size,256,256,1], dtype=tf.float32)])
  def DZ_tf(var):
    ddz1 = var[...,:,1:2,:] - var[...,:,:1,:]
    ddz = var[...,:,2:,:] - var[...,:,:-2,:]
    ddz2 = var[...,:,-2:-1,:] - var[...,:,-1:,:]
    ddz = tf.concat([ddz1,ddz,ddz2], axis=-2)
    return tf.cast(ddz, tf.float32) / dz
     
  @tf.function(input_signature=[tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32),
                                tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32)])
  def my_loss(U_true, U_pred):
    data_losses = tf.reduce_mean(tf.math.square(U_pred-U_true), axis=[0,1,2])
    return data_losses
    
    #u_pred, w_pred = U_pred[...,0:1], U_pred[...,1:2]
    #mc_loss = tf.reduce_mean( tf.square( DX_tf(u_pred) + DZ_tf(w_pred) ), axis=[0,1,2] )
    #return tf.concat([data_losses, mc_loss], axis=0)

    
  base_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
  optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
  try:
    autoencoder = tf.keras.saving.load_model(f'./models/vit_cae_{suffix}.keras',
                                         custom_objects={'PatchExtract': PatchExtract, 'PatchEmbedding':PatchEmbedding})
    
    start_scheduler = 0
  except:
    autoencoder = build_vit_cae_model(image_size, patch_size, projection_dim, num_heads, transformer_layers, transformer_units, enc_layers, drop, kernel_size)
    start_scheduler = 300
    
  
print(autoencoder.summary())


######################################################################################
## TRAIN MODEL
n_lambdas = 4

learning_rate = lr
best_loss = float('inf') 
lambdas = tf.Variable(tf.ones([n_lambdas], tf.float32) / n_lambdas, trainable=False)
loss_history = tf.Variable(tf.zeros([n_lambdas]), dtype=tf.float32, trainable=False)

train_losses = tf.Variable(tf.zeros([n_lambdas]), dtype=tf.float32, trainable=False)
val_losses = tf.Variable(tf.zeros([n_lambdas]), dtype=tf.float32, trainable=False)

w_data = tf.Variable(tf.zeros([4]), dtype=tf.float32, trainable=False)
w_0 = tf.Variable(tf.zeros([1]), dtype=tf.float32, trainable=False)
w = tf.Variable(tf.zeros([n_lambdas]), dtype=tf.float32, trainable=False) 



factor = 0.8 
patience = 15 
min_lr = 1e-5 
best_val_loss = float('inf')
wait = 0 

## Train Step
@tf.function(input_signature=[tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32)])
def train_step(U_batch):
  with tf.GradientTape() as tape: #persistent=True
    U_pred = autoencoder(U_batch, training=True)
    losses = my_loss(U_batch, U_pred)  
    loss = tf.reduce_sum(losses*lambdas)
    scaled_loss = optimizer.get_scaled_loss(loss)
    
  scaled_gradients = tape.gradient(scaled_loss, autoencoder.trainable_variables)
  gradients = optimizer.get_unscaled_gradients(scaled_gradients)
  optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables)) 
   
  return losses
 
## Val Step 
@tf.function(input_signature=[tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32)])
def val_step(U_batch):
  U_pred = autoencoder(U_batch, training=False)
  return my_loss(U_batch, U_pred) 

## Training Loop
epochs = 1000
for epoch in range(epochs):
  
  for step, (U_batch, _) in enumerate(ae_train):
    batch_losses = train_step(U_batch)
    train_losses.assign_add(batch_losses)
       
  train_losses.assign( train_losses / (tf.cast(step, tf.float32)+1.) )
  
  if epoch >= 1:
    '''
    if epoch <= 300:
      w_data.assign(train_losses[:4] / loss_history[:4])
      lambdas.assign( tf.concat([tf.nn.softmax(w_data), w_0], axis=0) )
    else:
    '''
    w.assign(train_losses / loss_history)  
    lambdas.assign( tf.nn.softmax(w) )

  loss_history.assign(train_losses)
   
  for step, (U_batch_val, _) in enumerate(ae_val):
    val_losses.assign_add( val_step(U_batch_val) )

  val_losses.assign( val_losses / (tf.cast(step, tf.float32)+1.) )
  val_loss = tf.reduce_mean(val_losses)
  
  if epoch >= start_scheduler:
    if val_loss < best_val_loss:
      best_val_loss = val_loss.numpy()
      wait = 0
    else:
      wait += 1
      if wait >= patience:
        old_lr = base_optimizer.lr.numpy()
        new_lr = max(old_lr * factor, min_lr)
        base_optimizer.lr.assign(new_lr)
        with open(f'history_vit_cae_{suffix}.txt', 'a') as log_file:
          log_file.write(f"Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}\n")
        wait = 0
  
      
  log_train = f"{train_losses.numpy().mean():.2e}"
  log_val1 = f"(u: {val_losses.numpy()[0]:.2e}, w: {val_losses.numpy()[1]:.2e}, "
  log_val2 = f"p: {val_losses.numpy()[2]:.2e}, T: {val_losses.numpy()[3]:.2e})" #, MC: {val_losses.numpy()[4]:.2e})" 
   
  with open(f'history_vit_cae_{suffix}.txt', 'a') as log_file:
    log_file.write(f'Epoch {epoch+1}. Train: {log_train}. Val: {val_losses.numpy().mean():.2e} {log_val1+log_val2}\n')
      
  if (epoch+1) % 10 == 0:  autoencoder.save(f'./models/vit_cae_{line:02d}.keras')
   
  train_losses.assign(tf.zeros_like(train_losses))  
  val_losses.assign(tf.zeros_like(val_losses)) 
  
























