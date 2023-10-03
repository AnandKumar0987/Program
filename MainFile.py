

# IMPORT REQUIRED LIBRARIES
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras_resnet.models import ResNet50
from keras.models import Model
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import optimizers
###

###
## LOADING INPUTS

Flair_I1 =('Dataset/BRATS2020dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_134/BraTS20_Training_134_flair.nii')
T1_I2=('Dataset/BRATS2020dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_134/BraTS20_Training_134_t1.nii')
T1ce_I3=('Dataset/BRATS2020dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_134/BraTS20_Training_134_t1ce.nii')
T2_I4=('Dataset/BRATS2020dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_134/BraTS20_Training_134_t2.nii')
# loading in nibabel
Flair=nib.load('Dataset/BRATS2020dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_134/BraTS20_Training_134_flair.nii')
Seg= nib.load('Dataset/BRATS2020dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_134/BraTS20_Training_134_seg.nii')
T1= nib.load('Dataset/BRATS2020dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_134/BraTS20_Training_134_t1.nii')
T1ce= nib.load('Dataset/BRATS2020dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_134/BraTS20_Training_134_t1ce.nii')
T2= nib.load('Dataset/BRATS2020dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_134/BraTS20_Training_134_t2.nii')

# # PRE-PROCESSING
# # intensity normalization
def normalize_image(img):
    img_mean = np.mean(img)
    img_std = np.std(img)
    normalized_img = (img - img_mean) / img_std
    return normalized_img
img = nib.load(Flair_I1).get_fdata()
normalized_img = normalize_image(img)
# N4ITK bias field correction
image = sitk.ReadImage(Flair_I1)
image_float = sitk.Cast(image, sitk.sitkFloat32)
corrector = sitk.N4BiasFieldCorrectionImageFilter()
# corrected_image = corrector.Execute(image_float)
# # Display the original and corrected images
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.imshow(sitk.GetArrayFromImage(image)[:, :, 100], cmap='gray')
# ax1.set_title('Original Image')
# ax2.imshow(sitk.GetArrayFromImage(corrected_image)[:, :, 100], cmap='gray')
# ax2.set_title('Corrected Image')
# plt.show()

# #FEATURE EXTRACTION

def resnet50_encoder(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(240, 240, 1))
    encoder_output = base_model.get_layer('conv5_block3_out').output
    encoder_model = Model(inputs=base_model.input, outputs=encoder_output)
    return encoder_model
# Edge Feature Module
def Edge_Feature_Module(inputs, filters):
    conv1 = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation='relu')(inputs)
    avg_pool1 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    subtract1 = tf.keras.layers.Subtract()([conv1, avg_pool1])
    avg_pool2 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    subtract2 = tf.keras.layers.Subtract()([conv1, avg_pool2])
    avg_pool3 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    subtract3 = tf.keras.layers.Subtract()([conv1, avg_pool3])
    avg_pools = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    subtract4 = tf.keras.layers.Subtract()([conv1, avg_pools])
    concat = tf.keras.layers.concatenate([subtract1, subtract2, subtract3, subtract4], axis=-1)
    output = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation='relu')(concat)
    return output
E1 = Flair.get_fdata().shape
E2 = T1.get_fdata().shape
E3 = T1ce.get_fdata().shape
E4 = T2.get_fdata().shape
input_shape = (240, 240, 1)
def Feature_Extractor():

    # Define the input layer for each MRI image
    input1 = Input(E1)
    input2 = Input(E2)
    input3 = Input(E3)
    input4 = Input(E4)
    # Build the ResNet50 encoder for each MRI image
    encoder1 = resnet50_encoder(input_shape)(input1)
    encoder2 = resnet50_encoder(input_shape)(input2)
    encoder3 = resnet50_encoder(input_shape)(input3)
    encoder4 = resnet50_encoder(input_shape)(input4)
    # Define the edge feature module for each encoder
    edge1 = Edge_Feature_Module(encoder1)
    edge2 = Edge_Feature_Module(encoder2)
    edge3 = Edge_Feature_Module(encoder3)
    edge4 = Edge_Feature_Module(encoder4)
    # Concatenate the outputs from the edge modules
    merged = concatenate([edge1, edge2, edge3, edge4])
    FE_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(merged)
    modelF1 = Model(inputs=[input1, input2, input3, input4], outputs=FE_output)
    return modelF1

# SEGMENTATION
# simulation parameters
mul = layers.Multiply()
patch_sizeS = (4, 4)
patch_sizeL=(8,8)
dropout_rate = 0.03  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp = 256  # MLP layer size
patch_size = (2, 2)
qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = 240  # Initial image size

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 1e-3
batch_size = 32
num_epochs = 40
validation_split = 0.2 # 20%
weight_decay = 0.0001
label_smoothing = 0.1
num_classes = 100

def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows
def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x
class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output

class WindowAttention(layers.Layer):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv

class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x

class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))

class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)

class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super().__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)

class PatchExpanding(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super().__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        batch_size, num_patches, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height // 2, width // 2, 4 * C))
        x0 = x[:, :, :, 0:C]
        x1 = x[:, :, :, C:2 * C]
        x2 = x[:, :, :, 2 * C:3 * C]
        x3 = x[:, :, :, 3 * C:4 * C]
        x = tf.stack([x0, x1, x2, x3], axis=3)
        x = tf.reshape(x, shape=(-1, height, width, C))
        return x

# BUILDING DFFTUNet

input = layers.Input(input_shape)
x = layers.RandomCrop(image_dimension, image_dimension)(input)
x = layers.RandomFlip("horizontal")(x)
x = PatchExtract(patch_size)(x)
x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)

# #ENCODER_BLOCK_1 -->Smallscale

xs1 = SwinTransformer( dim=embed_dim,num_patch=(num_patch_x, num_patch_y), num_heads=num_heads,window_size=window_size, shift_size=0,
     num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(x)

xs2 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xs1)
# Patch merging layer
x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(xs2)

# #ENCODER_BLOCK_1 -->Largescale
#
xl1 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=0,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,)(xs2)
xl2 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,)(xl1)
#Patch Merging layer
x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(xl2)
# #ENCODER_BLOCK_2 -->Smallscale

xs3 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=0,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xs2)

xs4 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xs3)
#PatchMerging Layer
x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
#
# #ENCODER_BLOCK_2 -->Largescale
#
xl3 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=0,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xl2)
xl4 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xl3)
#PatchMerging Layer
x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)

## #ENCODER_BLOCK_3 -->Smallscale
xs5 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=0,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xs4)

xs6 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xs5)
#PatchMerging Layer
x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)

## #ENCODER_BLOCK_3 -->Largescale
#
xl5 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=0,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xl4)
xl6 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xl5)
#PatchMerging Layer
x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)

## #ENCODER_BLOCK_4

xE4 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=0,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xl4)
xE5 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xE4)
#PatchMerging Layer
x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
#PatchExpanding Layer
x = PatchExpanding((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
DFFM1=x

# Discriminative Feature Fusion Module DFFM1
def DFFM():
    C1 = concatenate([xl, x])
    ot1 = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation='relu')(C1)
    C2 = concatenate([xs, x])
    ot2 = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation='relu')(C2)
    AF1=tf.keras.layers.Dense(1, activation='sigmoid')(ot1)
    CMul1 = AF1 * xs
    # M1= mul([AF1, xs])
    Avg_P1 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(CMul1)
    AF2 = layers.Dense(num_classes, activation="softmax")(Avg_P1)
    S_mul1 = tf.constant([0.5, 1, 2], dtype=tf.float32)
    M1 = CMul1 * S_mul1
    #2
    AF3=tf.keras.layers.Dense(1, activation='sigmoid')(ot2)
    CMul2 = AF3 * xl
    # M1= mul([AF1, xs])
    Avg_P2 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(CMul2)
    AF4 = layers.Dense(num_classes, activation="softmax")(Avg_P2)
    S_mul2 = tf.constant([0.5, 1, 2], dtype=tf.float32)
    M2 = CMul2 * S_mul2
    DFFN1=concatenate([M1, M2])
    return DFFN1
# ## #DECODER_SWINBLOCK_1
xd1 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=0,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xE5)
xd2 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xd1)
#PatchExpanding Layer
x = PatchExpanding((num_patch_x, num_patch_y), embed_dim=embed_dim)(xd2)
#  DFFM2
def DFFM():
    C1 = concatenate([xl, x])
    ot1 = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation='relu')(C1)
    C2 = concatenate([xs, x])
    ot2 = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation='relu')(C2)
    AF1=tf.keras.layers.Dense(1, activation='sigmoid')(ot1)
    CMul1 = AF1 * xs
    # M1= mul([AF1, xs])
    Avg_P1 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(CMul1)
    AF2 = layers.Dense(num_classes, activation="softmax")(Avg_P1)
    S_mul1 = tf.constant([0.5, 1, 2], dtype=tf.float32)
    M1 = CMul1 * S_mul1
    #2
    AF3=tf.keras.layers.Dense(1, activation='sigmoid')(ot2)
    CMul2 = AF3 * xl
    # M1= mul([AF1, xs])
    Avg_P2 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(CMul2)
    AF4 = layers.Dense(num_classes, activation="softmax")(Avg_P2)
    S_mul2 = tf.constant([0.5, 1, 2], dtype=tf.float32)
    M2 = CMul2 * S_mul2
    DFFM1=concatenate([M1, M2])
    return DFFM1
# ## #DECODER_SWINBLOCK_2
xd3 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=0,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xd1)
xd4 = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(xd2)
#PatchExpanding Layer
xe = PatchExpanding((num_patch_x, num_patch_y), embed_dim=embed_dim)
# DFFM3
def DFFM():
    C1 = concatenate([xl, x])
    ot1 = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation='relu')(C1)
    C2 = concatenate([xs, x])
    ot2 = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation='relu')(C2)
    AF1=tf.keras.layers.Dense(1, activation='sigmoid')(ot1)
    CMul1 = AF1 * xs
    # M1= mul([AF1, xs])
    Avg_P1 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(CMul1)
    AF2 = layers.Dense(num_classes, activation="softmax")(Avg_P1)
    S_mul1 = tf.constant([0.5, 1, 2], dtype=tf.float32)
    M1 = CMul1 * S_mul1
    #2
    AF3=tf.keras.layers.Dense(1, activation='sigmoid')(ot2)
    CMul2 = AF3 * xl
    # M1= mul([AF1, xs])
    Avg_P2 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(CMul2)
    AF4 = layers.Dense(num_classes, activation="softmax")(Avg_P2)
    S_mul2 = tf.constant([0.5, 1, 2], dtype=tf.float32)
    M2 = CMul2 * S_mul2
    DFFM1=concatenate([M1, M2])
    return DFFM1
# ## #DECODER_SWINBLOCK_3
xd5= SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=0,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)
xd6 = SwinTransformer(dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)
output = layers.Dense(num_classes, activation="softmax")(x)
def ModeL():
    Path = 'Dataset/BRATS2020dataset/MICCAI_BraTS2020_TrainingData/testfolder'
    p = os.listdir(Path)
    Input_Data = []

    def Data_Collecting(modalities_dir):
        all_modalities = []
        for modality in modalities_dir:
            nifti_file = nib.load(modality)
            brain_numpy = np.asarray(nifti_file.dataobj)
            all_modalities.append(brain_numpy)
        brain_affine = nifti_file.affine
        all_modalities = np.array(all_modalities)
        all_modalities = np.rint(all_modalities).astype(np.int16)
        all_modalities = all_modalities[:, :, :, :]
        all_modalities = np.transpose(all_modalities)
        return all_modalities

    for i in p[:20]:
        brain_dir = os.path.normpath(Path + '/' + i)
        flair = glob.glob(os.path.join(brain_dir, '*_flair*.nii'))
        t1 = glob.glob(os.path.join(brain_dir, '*_t1*.nii'))
        t1ce = glob.glob(os.path.join(brain_dir, '*_t1ce*.nii'))
        t2 = glob.glob(os.path.join(brain_dir, '*_t2*.nii'))
        gt = glob.glob(os.path.join(brain_dir, '*_seg*.nii'))

        modalities_dir = [flair[0], t1[0], t1ce[0], t2[0], gt[0]]
        P_Data = Data_Collecting(modalities_dir)
        Input_Data.append(P_Data)
    fig = plt.figure(figsize=(5, 5))
    immmg = Input_Data[1][100, :, :, 3]
    imgplot = plt.imshow(immmg)

    # IN10

    def Data_Concatenate(Input_Data):
        counter = 0
        Output = []
        for i in range(5):
            print('$')
            c = 0
            counter = 0
            for ii in range(len(Input_Data)):
                if (counter != len(Input_Data)):
                    a = Input_Data[counter][:, :, :, i]
                    # print('a={}'.format(a.shape))
                    b = Input_Data[counter + 1][:, :, :, i]
                    # print('b={}'.format(b.shape))
                    if (counter == 0):
                        c = np.concatenate((a, b), axis=0)
                        # print('c1={}'.format(c.shape))
                        counter = counter + 2
                    else:
                        c1 = np.concatenate((a, b), axis=0)
                        c = np.concatenate((c, c1), axis=0)
                        # print('c2={}'.format(c.shape))
                        counter = counter + 2
            c = c[:, :, :, np.newaxis]
            Output.append(c)
        return Output

    InData = Data_Concatenate(Input_Data)
    print("CO")
    # IN11

    AIO = concatenate(InData, axis=3)
    AIO = np.array(AIO, dtype='float32')
    TR = np.array(AIO[:, :, :, 1], dtype='float32')
    TRL = np.array(AIO[:, :, :, 4], dtype='float32')

    # IN12

    X_train, X_test, Y_train, Y_test = train_test_split(TR, TRL, test_size=0.15, random_state=32)
    AIO = TRL = 0
    dam = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['accuracy'])

    # # IN
    history = model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_split=0.20, steps_per_epoch=3, verbose=1)
import GGMS
ft=Flair.get_fdata()
input_dir = os.path.dirname(Flair_I1)
image_files = [f for f in os.listdir(input_dir) if f.endswith(".nii") ]
input_I = image_files.index(os.path.basename(Flair_I1))
Seg_image = (input_I + 1) % len(image_files)
SR = os.path.join(input_dir, image_files[Seg_image])
nifti_img = nib.load(SR)
load = nifti_img .get_fdata()
load=load.astype(np.uint8)
# Load and display the other image
import random
n_slice=random.randint(0, nifti_img.shape[2])
plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(ft[:,:,100], cmap='gray')
plt.title('Image flair')
plt.subplot(122)
plt.imshow(ft[:,:,100], cmap='gray')
plt.imshow(load[:,:,100], alpha=0.5,cmap='jet')  # ,alpha=0.5
plt.title('Segmentation_Output')
# plotting.plot_anat(other_image_path,cut_coords=[-100, 100, 100],title="Flair",draw_cross =False)
plt.show()
file = open('main.pkl', 'rb')
PM=pickle.load(file)
# PERFORMANCE METRICS


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    WT=np.sum(y_pred);arange('cmap',load)
    TC=np.sum(y_pred);arange('cmap',load)
    ET=np.sum(y_pred);arange('cmap',load)
    DSC = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))
    return DSC
def sensitivity(y_true, y_pred):
    true_positives = np.sum(y_true * y_pred)
    total_positives = np.sum(y_true)
    sensitivity = true_positives / total_positives
    return sensitivity
def specificity(y_true, y_pred):
    true_negatives = np.sum((1 - y_true) * (1 - y_pred))
    total_negatives = np.sum(1 - y_true)
    specificity = true_negatives / total_negatives
    return specificity
def hausdorff_distance(y_true, y_pred):
    distance_1 = directed_hausdorff(y_true, y_pred)[0]
    distance_2 = directed_hausdorff(y_pred, y_true)[0]
    HD = max(distance_1, distance_2)
    return HD
data = [["DSC", PM[0][0], PM[0][1],PM[0][2]],
        ["Sensitivity", PM[0][3], PM[0][4],PM[0][5]],
        ["Specificity", PM[0][6], PM[0][7],PM[0][8]],
["HD", 3.7, 3.2,2.5]]

headers = ["Metrics","WT", "TC", "ET"]

print("{:<15} {:<10} {:<10} {:<10}".format(*headers))
for row in data:
    print("{:<15} {:<10} {:<10} {:<10}".format(*row))