import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


# -----------------------------------------------------------
# SIMPLE GRAPH CONVOLUTION LAYER
# -----------------------------------------------------------
class GraphConv(tf.keras.layers.Layer):
    """
    Basic GCN layer:
    Z = A_hat * X * W
    """
    def __init__(self, output_dim):
        super(GraphConv, self).__init__()
        self.output_dim = output_dim

    def build(self, input_shape):
        input_dim = input_shape[1][-1]  # X feature size
        self.w = self.add_weight(shape=(input_dim, self.output_dim),
                                 initializer="glorot_uniform",
                                 trainable=True)

    def call(self, inputs):
        A, X = inputs  # adjacency matrix + feature matrix
        A_hat = A + tf.eye(A.shape[0])  # Add self connections
        D = tf.reduce_sum(A_hat, axis=1)
        D_inv_sqrt = tf.linalg.tensor_diag(1.0 / tf.sqrt(D))
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        return A_norm @ X @ self.w


# -----------------------------------------------------------
# FULL GNN MODEL
# -----------------------------------------------------------
class GNNModel(Model):
    """
    Factory GNN:
    Input:
        - adjacency matrix A
        - feature matrix X
    Output:
        - risk score per machine (0–1)
    """

    def __init__(self