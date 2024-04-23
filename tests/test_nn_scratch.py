import numpy as np
import sys
import numdifftools as nd
sys.path.append("./src")
from nn_scratch import *

def test_parse_fashion_mnist():
    X,y = parse_fashion_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    assert X.dtype == np.float32
    assert y.dtype == np.uint8
    assert X.shape == (60000,784)
    assert y.shape == (60000,)
    # np.testing.assert_allclose(np.linalg.norm(X[:10]), 27.892084)
    # np.testing.assert_allclose(np.linalg.norm(X[:1000]), 293.0717,
    #     err_msg="""If you failed this test but not the previous one,
    #     you are probably normalizing incorrectly. You should normalize
    #     w.r.t. the whole dataset, _not_ individual images.""", rtol=1e-6)
    # np.testing.assert_equal(y[:10], [5, 0, 4, 1, 9, 2, 1, 3, 1, 4])



def test_nn_epoch_simple():

    # test nn gradients
    np.random.seed(0)
    X = np.random.randn(50,5).astype(np.float32)
    y = np.random.randint(3, size=(50,)).astype(np.uint8)
    W1 = np.random.randn(5, 10).astype(np.float32) / np.sqrt(10)
    W2 = np.random.randn(10, 3).astype(np.float32) / np.sqrt(3)
    dW1 = nd.Gradient(lambda W1_ : 
        softmax_loss(np.maximum(X@W1_.reshape(5,10),0)@W2, y))(W1)
    dW2 = nd.Gradient(lambda W2_ : 
        softmax_loss(np.maximum(X@W1,0)@W2_.reshape(10,3), y))(W2)
    W1_0, W2_0 = W1.copy(), W2.copy()
    nn_epoch_simple(X, y, W1, W2, lr=1.0, batch=50)
    np.testing.assert_allclose(dW1.reshape(5,10), W1_0-W1, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dW2.reshape(10,3), W2_0-W2, rtol=1e-4, atol=1e-4)

    # test full epoch
    X,y = parse_fashion_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    np.random.seed(0)
    W1 = np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100)
    W2 = np.random.randn(100, 10).astype(np.float32) / np.sqrt(10)
    nn_epoch_simple(X, y, W1, W2, lr=0.2, batch=100)
    # np.testing.assert_allclose(np.linalg.norm(W1), 28.437788, 
    #                            rtol=1e-5, atol=1e-5)
    # np.testing.assert_allclose(np.linalg.norm(W2), 10.455095, 
    #                            rtol=1e-5, atol=1e-5)
    # np.testing.assert_allclose(loss_err(np.maximum(X@W1,0)@W2, y),
    #                            (0.19770025, 0.06006667), rtol=1e-4, atol=1e-4) 

def test_dataloader_all():
  np.random.seed(0)
  X = np.random.randn(50,5).astype(np.float32)
  y = np.random.randint(3, size=(50,)).astype(np.uint8)

  # batch_size=-1
  dataloader = DataLoader(X,y,-1)
  for (x_b, y_b) in dataloader:
    np.testing.assert_allclose(x_b, X, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_b, y, rtol=1e-5, atol=1e-5)
    break

def test_dataloader_1():
  np.random.seed(0)
  X = np.random.randn(50,5).astype(np.float32)
  y = np.random.randint(3, size=(50,)).astype(np.uint8)

  # batch_size=1
  dataloader = DataLoader(X,y,1)
  for (x_b, y_b) in dataloader:
    np.testing.assert_allclose(x_b, X[0:1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_b, y[0:1], rtol=1e-5, atol=1e-5)
    break

def test_dataloader_32():
  np.random.seed(0)
  X = np.random.randn(50,5).astype(np.float32)
  y = np.random.randint(3, size=(50,)).astype(np.uint8)

  # batch_size=32
  dataloader = DataLoader(X,y,32)
  for (x_b, y_b) in dataloader:
    np.testing.assert_allclose(x_b, X[0:32], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_b, y[0:32], rtol=1e-5, atol=1e-5)
    break


def test_model_relu():
  np.random.seed(0)
  X = np.random.randn(50,5).astype(np.float32)
  y = np.random.randint(3, size=(50,)).astype(np.uint8)
  W1 = np.random.randn(5, 10).astype(np.float32) / np.sqrt(10)
  W2 = np.random.randn(10, 3).astype(np.float32) / np.sqrt(3)
  np.random.seed(0)
  model = Model(5, 3, 10)
  model.W1 = W1.copy()
  model.W2 = W2.copy()
  np.testing.assert_allclose(model.b1, np.zeros_like(model.b1).astype(np.float32), rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(model.b2, np.zeros_like(model.b2).astype(np.float32), rtol=1e-5, atol=1e-5)
  nn_epoch_simple(X, y, W1, W2, lr=1.0, batch=50)
  model.forward(X)
  model.backward(y)
  np.testing.assert_allclose(W1, model.W1-model.W1_grad, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(W2, model.W2-model.W2_grad, rtol=1e-5, atol=1e-5)

def test_model_tanh_backward():
  pass

def test_model_sigmoid_backward():
  pass

def test_model_state_dict():
  model = Model(5, 3, 10)
  path = './tmp.npy'
  model.save_state_dict(path)
  model.load_state_dict(path)

def test_train_sgd():
  np.random.seed(0)
  X = np.random.randn(50,5).astype(np.float32)
  y = np.random.randint(3, size=(50,)).astype(np.uint8)
  W1 = np.random.randn(5, 10).astype(np.float32) / np.sqrt(10)
  W2 = np.random.randn(10, 3).astype(np.float32) / np.sqrt(3)
  model = Model(5, 3, 10)
  model.W1 = W1.copy()
  model.W2 = W2.copy()
  opt = SGD(model, 1, 0)
  nn_epoch_simple(X, y, W1, W2, lr=1.0, batch=50)
  model.forward(X)
  model.backward(y)
  opt.step()
  np.testing.assert_allclose(W1, model.W1, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(W2, model.W2, rtol=1e-5, atol=1e-5)
  
def test_train_steplr():
  np.random.seed(0)
  model = Model(5, 3, 10)
  opt = SGD(model, 1, 0)
  lr_scheduler = StepLR(opt, 1, 0.9)
  lr_scheduler.step()
  np.testing.assert_allclose(opt.lr, 0.9, rtol=1e-5, atol=1e-5)


def test_train_logger():
  logger = Logger()
  logger.log_step(1, 1, 2)
  logger.log_step(2, 2, 1)
  logger.log_epoch()
  np.testing.assert_allclose(logger.epoch_loss[-1], 4/3, rtol=1e-5, atol=1e-5)

def test_train_nn_epoch():
  pass

def test_train_split():
  pass

def test_train_loss_acc():
  pass

if __name__ == "__main__":
  test_parse_fashion_mnist()
  test_nn_epoch_simple()
  test_dataloader_all()
  test_dataloader_1()
  test_dataloader_32()