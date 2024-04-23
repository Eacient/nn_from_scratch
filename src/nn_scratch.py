import numpy as np
import gzip
import struct
from matplotlib import pyplot as plt
import os
import itertools
from tqdm import tqdm

def parse_fashion_mnist(image_filename, label_filename):
  def read_images(filename):
    with gzip.open(filename, 'rb') as f:
      magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
      image_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
      return image_data.astype(np.float32) / 255.0

  # Function to read labels
  def read_labels(filename):
    with gzip.open(filename, 'rb') as f:
      magic, num_labels = struct.unpack(">II", f.read(8))
      label_data = np.frombuffer(f.read(), dtype=np.uint8)
      return label_data

  # Read images and labels
  X = read_images(image_filename)
  y = read_labels(label_filename)
  return X, y

def softmax_loss(Z, y):
    ### BEGIN YOUR CODE
    # Compute softmax probabilities
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    softmax_probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    # Compute cross-entropy loss
    batch_size = Z.shape[0]
    loss = -np.log(softmax_probs[np.arange(batch_size), y])
    
    # Compute average loss
    avg_loss = np.mean(loss)
    
    return avg_loss
    ### END YOUR CODE

def nn_epoch_simple(X, y, W1, W2, lr = 0.1, batch=100):
  """ 
  Args:
    X (np.ndarray[np.float32]): 2D input array of size
        (num_examples x input_dim).
    y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
    theta (np.ndarrray[np.float32]): 2D array of softmax regression
        parameters, of shape (input_dim, num_classes)
    lr (float): step size (learning rate) for SGD
    batch (int): size of SGD minibatch

  Returns:
      None
  """
  num_examples = X.shape[0]
  input_dim = X.shape[1]
  hidden_dim = W1.shape[1]
  num_classes = W2.shape[1]

  # Iterate through batches
  for i in range(0, num_examples, batch):
    # Get batch of examples
    batch_size = batch if i+batch <= num_examples else num_examples - i
    X_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]

    # forward pass
    h = np.maximum(X_batch@W1, 0)
    logits = h@W2
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    del logits, exp_logits

    # Compute gradients
    # loss = -np.sum(np.log(softmax_probs[np.arange(batch_size), y_batch])) / batch_size
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot[np.arange(batch_size), y_batch] = 1
    logits_grad = softmax_probs - one_hot #[n,k]
    W2_grad = h.T@logits_grad / len(y_batch) #[h,k]
    h_grad = logits_grad@W2.T #[n, h]
    W1_grad = X_batch.T@(np.where(h>0, 1, 0) * h_grad) / len(y_batch) #[i, h]
    del one_hot, logits_grad, h_grad

    # update
    W1 -= lr * W1_grad
    W2 -= lr * W2_grad

def loss_err(h,y):
  """ Helper funciton to compute both loss and error"""
  return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)

def train_nn_simple(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
  n, k = X_tr.shape[1], y_tr.max() + 1
  np.random.seed(0)
  W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
  W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

  print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
  for epoch in range(epochs):
    nn_epoch_simple(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
    train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
    test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
    print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
          .format(epoch, train_loss, train_err, test_loss, test_err))


class DataLoader:
  def __init__(self, x, y, batch_size, shuffle=False):
    self.x = x
    self.y = y
    self.shuffle = shuffle
    self.batch_size = batch_size if batch_size != -1 else len(x)
    if not self.shuffle:
      self.ordering = np.array_split(np.arange(len(x)), range(batch_size, len(x), batch_size))
  
  def __iter__(self):
    if self.shuffle:
      self.ordering = np.array_split(np.random.permutation(len(self.x)), range(self.batch_size, len(self.x), self.batch_size))
    self.idx = 0
    self.size = len(self.ordering)
    return self

  def __next__(self):
    if self.idx >= len(self.ordering):
      raise StopIteration
    x = np.stack([self.x[i] for i in self.ordering[self.idx]])
    y = np.stack([self.y[i] for i in self.ordering[self.idx]])
    self.idx += 1
    return x, y 

class Model:
  def __init__(self, input_dim, output_dim, hidden_dim, act='relu'):
    assert act in ('relu', 'sigmoid', 'tanh', 'gelu', 'lrelu', 'swish', 'mish')
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.act = act
    self._init_params()

  def _init_params(self):
    self.W1 = np.random.randn(self.input_dim, self.hidden_dim).astype(np.float32) / np.sqrt(self.hidden_dim)
    self.W2 = np.random.randn(self.hidden_dim, self.output_dim).astype(np.float32) / np.sqrt(self.output_dim)
    self.b1 = np.zeros(self.hidden_dim).astype(np.float32)
    self.b2 = np.zeros(self.output_dim).astype(np.float32)
    self.x = None
    self.h = None
    self.softmax_probs = None
    self.W1_grad = 0
    self.W2_grad = 0
    self.b1_grad = 0
    self.b2_grad = 0

  def act_forward(self, z):
    if self.act == 'relu':
      return np.maximum(z, 0)
    elif self.act == 'tanh':
      ez = np.exp(z)
      enz = np.exp(-z)
      return (ez - enz) / (ez + enz)
    elif self.act == 'sigmoid':
      return 1 / (1 + np.exp(-z))
    else:
      raise NotImplementedError

  def act_backward(self, h_grad):
    assert self.h is not None
    if self.act == 'relu':
      return h_grad * np.where(self.h>0, 1, 0)
    elif self.act == 'tanh':
      return h_grad * (1 - self.h**2)
    elif self.act == 'sigmoid':
      return h_grad * self.h * (1-self.h)
    else:
      raise NotImplementedError

  def reset(self):
    self._init_params()

  def forward(self, X, grad=True):
    h = self.act_forward(X@self.W1+self.b1)
    logits = h@self.W2+self.b2
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    if grad:
      self.x = X
      self.h = h
      self.softmax_probs = softmax_probs
    return softmax_probs

  def backward(self, y):
    assert self.h is not None and self.softmax_probs is not None and self.x is not None
  
    batch_size = len(y)
    one_hot = np.zeros((batch_size, self.output_dim), dtype=np.float32)
    one_hot[np.arange(batch_size), y] = 1
    logits_grad = self.softmax_probs - one_hot
    b2_grad = logits_grad.mean(axis=0)
    W2_grad = self.h.T@logits_grad / batch_size
    h_grad = logits_grad@self.W2.T
    b1_grad = self.act_backward(h_grad).mean(axis=0)
    W1_grad = self.x.T@(self.act_backward(h_grad)) / batch_size
    self.W1_grad = W1_grad
    self.W2_grad = W2_grad
    self.b1_grad = b1_grad
    self.b2_grad = b2_grad
  
    self.x = None
    self.h = None
    self.softmax_probs = None

  def save_state_dict(self, model_path):
    state_dict = {
      'W1': self.W1,
      'W2': self.W2,
      'b1': self.b1,
      'b2': self.b2
    }
    np.save(model_path, state_dict, allow_pickle=True)
  
  def load_state_dict(self, model_path):
    state_dict = np.load(model_path, allow_pickle=True).item()
    self.W1 = state_dict['W1']
    self.W2 = state_dict['W2']
    self.b1 = state_dict['b1']
    self.b2 = state_dict['b2']


class SGD:
  def __init__(self, model, lr, weight_decay):
    self.model = model
    self.lr = lr
    self.weight_decay = weight_decay

  def step(self):
    W1_grad = self.model.W1_grad
    W2_grad = self.model.W2_grad
    b1_grad = self.model.b1_grad
    b2_grad = self.model.b2_grad
    assert W1_grad is not None and W2_grad is not None
    if self.weight_decay > 0:
      W1_grad += self.weight_decay * self.model.W1
      W2_grad += self.weight_decay * self.model.W2
    self.model.W1 -= self.lr * W1_grad
    self.model.W2 -= self.lr * W2_grad
    # self.model.b1 -= self.lr * b1_grad
    # self.model.b2 -= self.lr * b2_grad

  def zero_grad(self):
    self.model.W1_grad = 0
    self.model.W2_grad = 0
    self.model.b1_grad = 0
    self.model.b2_grad = 0

class StepLR:
  def __init__(self, optimizer, step_size=1, decay_rate=0.9):
    self.optimizer = optimizer
    self.epoch = 0
    self.decay_rate = decay_rate

  def step(self):
    self.optimizer.lr *= self.decay_rate
    self.epoch += 1

class Logger:
  def __init__(self):
    self.epoch_loss = []
    self.epoch_acc = []
    # self.step_loss = []
    # self.step_acc = []
    self.samples = 0
    self.loss_mean = 0
    self.acc_mean = 0

  def log_step(self, loss, acc, samples):
    # self.step_loss.append(loss)
    # self.step_acc.append(acc)
    if self.samples == 0:
      self.loss_mean = loss
      self.acc_mean = acc
    else:
      self.loss_mean = self.loss_mean / (1+samples/self.samples) + loss / (1+self.samples/samples)
      self.acc_mean = self.acc_mean / (1+samples/self.samples) + acc / (1+self.samples/samples)
    self.samples += samples
  
  def log_epoch(self):
    self.epoch_loss.append(self.loss_mean)
    self.epoch_acc.append(self.acc_mean)
    self.samples = 0
    self.loss_mean = 0
    self.acc_mean = 0

  def plot(self, mode='epoch', path=None):
    if (mode=='step'):
      raise NotImplementedError
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('acc')
    ax1.plot(np.arange(len(self.epoch_loss))+1, self.epoch_loss, linewidth=1, linestyle="solid", label="loss")
    ax1.legend()
    ax1.set_title('Loss Curve')
    ax2.plot(np.arange(len(self.epoch_acc))+1, self.epoch_acc, linewidth=1, linestyle="solid", label="acc")
    ax2.legend()
    ax2.set_title('Accuracy Curve')
    plt.show()
    if path is not None:
      plt.savefig(path)

def cross_entropy_loss(h, y):
  return -np.mean(np.log(h[np.arange(len(h)), y]))

def accuracy(h, y):
  return np.mean(np.argmax(h, axis=1) == y)

def loss_acc(h, y):
  return cross_entropy_loss(h, y), accuracy(h, y)

def nn_epoch(model, dataloader, logger, opt=None, lr_scheduler=None):
  for (x, y) in dataloader:
    h = model.forward(x)
    loss, acc = loss_acc(h, y)
    logger.log_step(loss, acc, len(y))
    if opt is not None:
      opt.zero_grad()
      model.backward(y)
      opt.step()
  if lr_scheduler is not None:
    lr_scheduler.step()
  logger.log_epoch()
  return logger.epoch_loss[-1], logger.epoch_acc[-1]

def train_val_split(x, y, ratio=0.7):
  order = np.random.permutation(len(x))
  train_index = order[:int(len(x)*0.7)]
  val_index = order[int(len(x)*0.7):]
  x_tr, y_tr = np.stack([x[i] for i in train_index]), np.stack([y[i] for i in train_index])
  x_val, y_val = np.stack([x[i] for i in val_index]), np.stack([y[i] for i in val_index])
  return x_tr, y_tr, x_val, y_val

def train_nn(data_dir, hidden_dim, act, epochs, batch_size, lr, weight_decay, step_size, decay_rate, save_dir, plot=False):
  np.random.seed(0)
  train_image_filename = f'{data_dir}/train-images-idx3-ubyte.gz'
  train_label_filename = f'{data_dir}/train-labels-idx1-ubyte.gz'
  x, y = parse_fashion_mnist(train_image_filename, train_label_filename)
  input_dim = x[0].shape[-1]
  output_dim = np.max(y) + 1
  model = Model(input_dim, output_dim, hidden_dim, act)
  x_tr, y_tr, x_val, y_val = train_val_split(x, y, ratio=0.7)
  train_loader = DataLoader(x_tr, y_tr, batch_size, shuffle=True)
  val_loader = DataLoader(x_val, y_val, batch_size, shuffle=False)
  opt = SGD(model, lr, weight_decay)
  lr_scheduler = StepLR(opt, step_size, decay_rate)
  train_logger = Logger()
  val_logger = Logger()
  best_acc = 0
  if plot:
    print("| Epoch | Train Loss | Train Acc | Val  Loss | Val  Acc |")
  epoch_range = range(epochs) if plot else tqdm(range(epochs))
  for epoch in epoch_range:
    train_loss, train_acc = nn_epoch(model, train_loader, train_logger, opt, lr_scheduler)
    val_loss, val_acc = nn_epoch(model, val_loader, val_logger)
    if plot:
      print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
          .format(epoch, train_loss, train_acc, val_loss, val_acc))
    if val_acc > best_acc:
      best_acc = val_acc
      model.save_state_dict(f'{save_dir}/model.npy')
  if plot:
    train_logger.plot(mode='epoch', path=f'{save_dir}/train_curve.png')
    val_logger.plot(mode='epoch', path=f'{save_dir}/val_curve.png')
  return best_acc
    

def param_search(data_dir, root_dir):
  # hidden_dim, act, lr, batch_size, weight_decay, lr_decay
  epochs = 10
  step_size = 1
  hidden_dims = [256]
  acts = ['relu']
  lrs = [0.03, 0.1]
  batch_sizes = [100, 1000]
  weight_decays = [0, 0.005]
  decay_rates = [0.8]
  for idx, (hidden_dim, act, batch_size, lr, weight_decay, decay_rate) in enumerate(itertools.product(hidden_dims, acts, batch_sizes, lrs, weight_decays, decay_rates)):
    save_dir = f'{root_dir}/{hidden_dim}-{act}-{batch_size}-{lr:.3f}-{weight_decay:.3f}-{decay_rate:.3f}'
    os.makedirs(save_dir, exist_ok=True)
    print('[{:02d}]: hidden_dim:{}, act:{}, batch_size:{}, lr:{:.3f}, weight_decay:{:.3f}, step_size:{}, decay_rate:{:.3f}'.format(
      idx, hidden_dim, act, batch_size, lr, weight_decay, step_size, decay_rate))
    acc = train_nn(data_dir, hidden_dim, act, epochs, batch_size, lr, weight_decay, step_size, decay_rate, save_dir)
    print('accuracy: {:.5f}'.format(acc))
    print('saved to', save_dir)
    print()

def test_nn(data_dir, model_path, hidden_dim, act, batch_size=-1):
  test_image_filename = f'{data_dir}/t10k-images-idx3-ubyte.gz'
  test_label_filename = f'{data_dir}/t10k-labels-idx1-ubyte.gz'
  x, y = parse_fashion_mnist(test_image_filename, test_label_filename)
  input_dim = x[0].shape[-1]
  output_dim = np.max(y) + 1
  model = Model(input_dim, output_dim, hidden_dim, act)
  model.load_state_dict(model_path)
  test_loader = DataLoader(x, y, batch_size)
  logger = Logger()
  loss, acc = nn_epoch(model, test_loader, logger)
  print(f'test accuracy: {acc:.5f}')
  print('W1:')
  plt.figure(figsize=(5,5))
  plt.imshow(model.W1, cmap='RdYlBu')
  plt.colorbar()
  plt.show()
  print('W2:')
  plt.figure(figsize=(5,5))
  plt.imshow(model.W2,cmap='RdYlBu')
  plt.colorbar()
  plt.show()

def model_visualizer():
  pass
