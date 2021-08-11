# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Network architectures used in the StyleGAN2 paper."""

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
import math
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d, upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act

# NOTE: Do not import any application-specific modules here!
# Specify all network parameters as kwargs.

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolution or fully-connected layer.

def get_weight(shape, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense_layer(x, fmaps, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolution layer with optional upsampling or downsampling.

def conv2d_layer(x, fmaps, kernel, up=False, down=False, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    if up:
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    elif down:
        x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')
    return x


#----------------------------------------------------------------------------
# Apply bias and activation func.

def apply_bias_act(x, act='linear', alpha=None, gain=None, lrmul=1, bias_var='bias'):
    b = tf.get_variable(bias_var, shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    return fused_bias_act(x, b=tf.cast(b, x.dtype), act=act, alpha=alpha, gain=gain)

#----------------------------------------------------------------------------
# Naive upsampling (nearest neighbor) and downsampling (average pooling).

def naive_upsample_2d(x, factor=2):
    with tf.variable_scope('NaiveUpsample'):
        _N, C, H, W = x.shape.as_list()
        x = tf.reshape(x, [-1, C, H, 1, W, 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        return tf.reshape(x, [-1, C, H * factor, W * factor])

def naive_downsample_2d(x, factor=2):
    with tf.variable_scope('NaiveDownsample'):
        _N, C, H, W = x.shape.as_list()
        x = tf.reshape(x, [-1, C, H // factor, factor, W // factor, factor])
        return tf.reduce_mean(x, axis=[3,5])

#----------------------------------------------------------------------------
# Modulated convolution layer.

def modulated_conv2d_layer(x, y, fmaps, kernel, up=False, down=False, demodulate=True, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, fused_modconv=True, weight_var='weight', mod_weight_var='mod_weight', mod_bias_var='mod_bias'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1

    # Get weight.
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    ww = w[np.newaxis] # [BkkIO] Introduce minibatch dimension.

    # Modulate.
    s = dense_layer(y, fmaps=x.shape[1].value, weight_var=mod_weight_var) # [BI] Transform incoming W to style.
    s = apply_bias_act(s, bias_var=mod_bias_var) + 1 # [BI] Add bias (initially 1).
    ww *= tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype) # [BkkIO] Scale input feature maps.

    # Demodulate.
    if demodulate:
        d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [BO] Scaling factor.
        ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :] # [BkkIO] Scale output feature maps.

    # Reshape/scale input.
    if fused_modconv:
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
        w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
    else:
        x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype) # [BIhw] Not fused => scale input activations.

    # Convolution with optional up/downsampling.
    if up:
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    elif down:
        x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')

    # Reshape/scale output.
    if fused_modconv:
        x = tf.reshape(x, [-1, fmaps, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
    elif demodulate:
        x *= tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype) # [BOhw] Not fused => scale output activations.
    return x

#----------------------------------------------------------------------------
# Inverse wavelet transform.

def iwt(x):
    """
    IWT (Inverse Wavelet Transfomr) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    x shape - BCHW (channel first)
    """
    x = tf.transpose(x, [0, 2, 3, 1])

    x_LL = x[:, :, :, 0:x.shape[3]//4]
    x_LH = x[:, :, :, x.shape[3]//4:x.shape[3]//4*2]
    x_HL = x[:, :, :, x.shape[3]//4*2:x.shape[3]//4*3]
    x_HH = x[:, :, :, x.shape[3]//4*3:]

    x1 = (x_LL - x_LH - x_HL + x_HH)/4
    x2 = (x_LL - x_LH + x_HL - x_HH)/4
    x3 = (x_LL + x_LH - x_HL - x_HH)/4
    x4 = (x_LL + x_LH + x_HL + x_HH)/4

    y1 = tf.stack([x1,x3], axis=2)
    y2 = tf.stack([x2,x4], axis=2)
    shape = x.shape
    recons = tf.reshape(tf.concat([y1,y2], axis=-1), tf.stack([-1, shape[1]*2, shape[2]*2, shape[3]//4]))
    recons = tf.transpose(recons, [0, 3, 1, 2])
    return recons


def dwt(x):

    """
    DWT (Discrete Wavelet Transform) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
       x shape - BCHW (channel first)
    """
    x = tf.transpose(x, [0, 2, 3, 1])

    x1 = x[:, 0::2, 0::2, :] #x(2i−1, 2j−1)
    x2 = x[:, 1::2, 0::2, :] #x(2i, 2j-1)
    x3 = x[:, 0::2, 1::2, :] #x(2i−1, 2j)
    x4 = x[:, 1::2, 1::2, :] #x(2i, 2j)


    x_LL = x1 + x2 + x3 + x4
    x_LH = -x1 - x3 + x2 + x4
    x_HL = -x1 + x3 - x2 + x4
    x_HH = x1 - x3 - x2 + x4

    wavelets = tf.concat([x_LL,x_LH,x_HL,x_HH], axis=-1)
    wavelets = tf.transpose(wavelets, [0, 3, 1, 2])
    return wavelets

def multi_level_dwt(x, levels):
    decomp = x
    for _ in range(levels):
        decomp = dwt(decomp)

    return decomp

def multi_level_iwt(x, levels):
    recon = x
    for _ in range(levels):
        recon = iwt(recon)

    return recon
#----------------------------------------------------------------------------
# Minibatch standard deviation layer.

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                                             # [NCHW]  Input shape.
    y = tf.reshape(x, [group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3]])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
    y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MncHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
    y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
    y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
    y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  Replicate over group and pixels.
    return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

#----------------------------------------------------------------------------
# Main generator network.
# Composed of two sub-networks (mapping and synthesis) that are defined below.
# Used in configs B-F (Table 1).

def G_main(
    latents_in,                                         # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                                          # Second input: Conditioning labels [minibatch, label_size].
    train_classification    = False,                    #Train a classification network in the discriminator
    truncation_psi          = 0.5,                      # Style strength multiplier for the truncation trick. None = disable.
    truncation_cutoff       = None,                     # Number of layers for which to apply the truncation trick. None = disable.
    truncation_psi_val      = None,                     # Value for truncation_psi to use during validation.
    truncation_cutoff_val   = None,                     # Value for truncation_cutoff to use during validation.
    dlatent_avg_beta        = 0.995,                    # Decay for tracking the moving average of W during training. None = disable.
    style_mixing_prob       = 0.9,                      # Probability of mixing styles during training. None = disable.
    is_training             = False,                    # Network is under training? Enables and disables specific features.
    is_validation           = False,                    # Network is under validation? Chooses which value to use for truncation_psi.
    return_dlatents         = False,                    # Return dlatents in addition to the images?
    is_template_graph       = False,                    # True = template graph constructed by the Network class, False = actual evaluation.
    components              = dnnlib.EasyDict(),        # Container for sub-networks. Retained between calls.
    mapping_func            = 'G_mapping',              # Build func name for the mapping network.
    synthesis_func          = 'G_synthesis_stylegan2',  # Build func name for the synthesis network.
    **kwargs):                                          # Arguments for sub-networks (mapping and synthesis).

    # Validate arguments.
    assert not is_training or not is_validation
    assert isinstance(components, dnnlib.EasyDict)
    if is_validation:
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training:
        truncation_cutoff = None
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    if not is_training or (style_mixing_prob is not None and not tflib.is_tf_expression(style_mixing_prob) and style_mixing_prob <= 0):
        style_mixing_prob = None

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network('G_synthesis', func_name=globals()[synthesis_func], **kwargs)
    num_layers = components.synthesis.input_shape[1]
    dlatent_size = components.synthesis.input_shape[2]
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_mapping', func_name=globals()[mapping_func], dlatent_broadcast=num_layers, **kwargs)

    # Setup variables.
    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)
    dlatent_avg = tf.get_variable('dlatent_avg', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False)

    # Evaluate mapping network.
    dlatents = components.mapping.get_output_for(latents_in, labels_in, is_training=is_training, **kwargs)
    dlatents = tf.cast(dlatents, tf.float32)

    # Update moving average of W.
    if dlatent_avg_beta is not None:
        with tf.variable_scope('DlatentAvg'):
            batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
            update_op = tf.assign(dlatent_avg, tflib.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    # Perform style mixing regularization.
    if style_mixing_prob is not None:
        with tf.variable_scope('StyleMix'):
            latents2 = tf.random_normal(tf.shape(latents_in))
            dlatents2 = components.mapping.get_output_for(latents2, labels_in, is_training=is_training, **kwargs)
            dlatents2 = tf.cast(dlatents2, tf.float32)
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = num_layers - tf.cast(lod_in, tf.int32) * 2
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
                lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)

    # Apply truncation trick.
    if truncation_psi is not None:
        with tf.variable_scope('Truncation'):
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            layer_psi = np.ones(layer_idx.shape, dtype=np.float32)
            if truncation_cutoff is None:
                layer_psi *= truncation_psi
            else:
                layer_psi = tf.where(layer_idx < truncation_cutoff, layer_psi * truncation_psi, layer_psi)
            dlatents = tflib.lerp(dlatent_avg, dlatents, layer_psi)

    # Evaluate synthesis network.
    deps = []
    if 'lod' in components.synthesis.vars:
        deps.append(tf.assign(components.synthesis.vars['lod'], lod_in))
    with tf.control_dependencies(deps):
        images_out = components.synthesis.get_output_for(dlatents, is_training=is_training, force_clean_graph=is_template_graph, **kwargs)

    # Return requested outputs.
    #images_out = tf.identity(images_out, name='images_out')
    if return_dlatents:
        return images_out, dlatents
    return images_out

#----------------------------------------------------------------------------
# Mapping network.
# Transforms the input latent code (z) to the disentangled latent code (w).
# Used in configs B-F (Table 1).

def G_mapping(
    latents_in,                             # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                              # Second input: Conditioning labels [minibatch, label_size].
    latent_size             = 512,          # Latent vector (Z) dimensionality.
    label_size              = 0,            # Label dimensionality, 0 if no labels.
    dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
    dlatent_broadcast       = None,         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
    mapping_layers          = 8,            # Number of mapping layers.
    mapping_fmaps           = 512,          # Number of activations in the mapping layers.
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
    mapping_nonlinearity    = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    normalize_latents       = True,         # Normalize latent vectors (Z) before feeding them to the mapping layers?
    dtype                   = 'float32',    # Data type to use for activations and outputs.
    **_kwargs):                             # Ignore unrecognized keyword args.

    act = mapping_nonlinearity

    # Inputs.
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    # Embed labels and concatenate them with latents.
    if label_size:
        with tf.variable_scope('LabelConcat'):
            w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
            y = tf.matmul(labels_in, tf.cast(w, dtype))
            x = tf.concat([x, y], axis=1)

    # Normalize latents.
    if normalize_latents:
        with tf.variable_scope('Normalize'):
            x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8)

    # Mapping layers.
    for layer_idx in range(mapping_layers):

        if layer_idx == 0 and label_size:
            with tf.variable_scope('Dense00'):
                fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
                x = apply_bias_act(dense_layer(x, fmaps=fmaps, lrmul=mapping_lrmul), act=act, lrmul=mapping_lrmul)
        else:
            with tf.variable_scope('Dense%d' % layer_idx):
                fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
                x = apply_bias_act(dense_layer(x, fmaps=fmaps, lrmul=mapping_lrmul), act=act, lrmul=mapping_lrmul)

    # Broadcast.
    if dlatent_broadcast is not None:
        with tf.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')

#----------------------------------------------------------------------------
# StyleGAN synthesis network with revised architecture (Figure 2d).
# Implements progressive growing, but no skip connections or residual nets (Figure 7).
# Used in configs B-D (Table 1).

def G_synthesis_stylegan_revised(
    dlatents_in,                        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
    num_channels        = 3,            # Number of output color channels.
    resolution          = 1024,         # Output resolution.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    fused_modconv       = True,         # Implement modulated_conv2d_layer() as a single fused op?
    structure           = 'auto',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    force_clean_graph   = False,        # True = construct a clean graph that looks nice in TensorBoard, False = default behavior.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    if is_template_graph: force_clean_graph = True
    if force_clean_graph: randomize_noise = False
    if structure == 'auto': structure = 'linear' if force_clean_graph else 'recursive'
    act = nonlinearity
    num_layers = resolution_log2 * 2 - 2
    images_out = None

    # Primary inputs.
    dlatents_in.set_shape([None, num_layers, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)

    # Noise inputs.
    noise_inputs = []
    for layer_idx in range(num_layers - 1):
        res = (layer_idx + 5) // 2
        shape = [1, 1, 2**res, 2**res]
        noise_inputs.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))

    # Single convolution layer with all the bells and whistles.
    def layer(x, layer_idx, fmaps, kernel, up=False):
        x = modulated_conv2d_layer(x, dlatents_in[:, layer_idx], fmaps=fmaps, kernel=kernel, up=up, resample_kernel=resample_kernel, fused_modconv=fused_modconv)
        if randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        x += noise * tf.cast(noise_strength, x.dtype)
        return apply_bias_act(x, act=act)

    # Early layers.
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1])
        with tf.variable_scope('Conv'):
            x = layer(x, layer_idx=0, fmaps=nf(1), kernel=3)

    # Building blocks for remaining layers.
    def block(res, x): # res = 3..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            with tf.variable_scope('Conv0_up'):
                x = layer(x, layer_idx=res*2-5, fmaps=nf(res-1), kernel=3, up=True)
            with tf.variable_scope('Conv1'):
                x = layer(x, layer_idx=res*2-4, fmaps=nf(res-1), kernel=3)
            return x
    def torgb(res, x): # res = 2..resolution_log2
        with tf.variable_scope('ToRGB_lod%d' % (resolution_log2 - res)):
            return apply_bias_act(modulated_conv2d_layer(x, dlatents_in[:, res*2-3], fmaps=num_channels, kernel=1, demodulate=False, fused_modconv=fused_modconv))

    # Fixed structure: simple and efficient, but does not support progressive growing.
    if structure == 'fixed':
        for res in range(3, resolution_log2 + 1):
            x = block(res, x)
        images_out = torgb(resolution_log2, x)

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        images_out = torgb(2, x)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(res, x)
            img = torgb(res, x)
            with tf.variable_scope('Upsample_lod%d' % lod):
                images_out = upsample_2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = tflib.lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
        def grow(x, res, lod):
            y = block(res, x)
            img = lambda: naive_upsample_2d(torgb(res, y), factor=2**lod)
            img = cset(img, (lod_in > lod), lambda: naive_upsample_2d(tflib.lerp(torgb(res, y), upsample_2d(torgb(res - 1, x)), lod_in - lod), factor=2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()
        images_out = grow(x, 3, resolution_log2 - 3)

    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')

#----------------------------------------------------------------------------
# StyleGAN2 synthesis network (Figure 7).
# Implements skip connections and residual nets (Figure 7), but no progressive growing.
# Used in configs E-F (Table 1).

def G_synthesis_stylegan2(
    dlatents_in,                        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
    num_channels        = 3,            # Number of output color channels.
    resolution          = 1024,         # Output resolution.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
    architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    fused_modconv       = True,         # Implement modulated_conv2d_layer() as a single fused op?
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity
    num_layers = resolution_log2 * 2 - 2
    images_out = None

    # Primary inputs.
    dlatents_in.set_shape([None, num_layers, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)

    # Noise inputs.
    noise_inputs = []
    for layer_idx in range(num_layers - 1):
        res = (layer_idx + 5) // 2
        shape = [1, 1, 2**res, 2**res]
        noise_inputs.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))

    # Single convolution layer with all the bells and whistles.
    def layer(x, layer_idx, fmaps, kernel, up=False):
        x = modulated_conv2d_layer(x, dlatents_in[:, layer_idx], fmaps=fmaps, kernel=kernel, up=up, resample_kernel=resample_kernel, fused_modconv=fused_modconv)
        if randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        x += noise * tf.cast(noise_strength, x.dtype)
        return apply_bias_act(x, act=act)

    # Building blocks for main layers.
    def block(x, res): # res = 3..resolution_log2
        t = x
        with tf.variable_scope('Conv0_up'):
            x = layer(x, layer_idx=res*2-5, fmaps=nf(res-1), kernel=3, up=True)
        with tf.variable_scope('Conv1'):
            x = layer(x, layer_idx=res*2-4, fmaps=nf(res-1), kernel=3)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-1), kernel=1, up=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x
    def upsample(y):
        with tf.variable_scope('Upsample'):
            return upsample_2d(y, k=resample_kernel)
    def torgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope('ToRGB'):
            t = apply_bias_act(modulated_conv2d_layer(x, dlatents_in[:, res*2-3], fmaps=num_channels, kernel=1, demodulate=False, fused_modconv=fused_modconv))
            return t if y is None else y + t

    # Early layers.
    y = None
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1])
        with tf.variable_scope('Conv'):
            x = layer(x, layer_idx=0, fmaps=nf(1), kernel=3)
        if architecture == 'skip':
            y = torgb(x, y, 2)

    # Main layers.
    for res in range(3, resolution_log2 + 1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            x = block(x, res)
            if architecture == 'skip':
                y = upsample(y)
            if architecture == 'skip' or res == resolution_log2:
                y = torgb(x, y, res)
    images_out = y

    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')




#----------------------------------------------------------------------------
# StyleGAN2_wavelets synthesis network. Learnable weights
# Used in configs E-F (Table 1).

def G_synthesis_wavelets_NU(
    dlatents_in,                        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
    num_channels        = 3,            # Number of output color channels.
    resolution          = 1024,         # Output resolution.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
    architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet', 'wavelets'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    fused_modconv       = True,         # Implement modulated_conv2d_layer() as a single fused op?
    decomp_level        = 1,            # Wavelet decomposition level. level=1 = 2x2, level=2 = 4x4 ...
    multi_output        = False,        # Output a list of images from all stages of the network
    reuse_add_weights   = False,        # Denotes whether the same weights should be shared between all chanel adding layers or not.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet', 'wavelets']
    act = nonlinearity
    num_layers = resolution_log2 * 2 - 2
    images_out = None

    # Primary inputs.
    dlatents_in.set_shape([None, num_layers, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)

    # Noise inputs.
    noise_inputs = []
    for layer_idx in range(num_layers - 1):
        res = (layer_idx + 5) // 2
        shape = [1, 1, 2**res, 2**res]
        noise_inputs.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))

    # Single convolution layer with all the bells and whistles.
    def layer(x, layer_idx, fmaps, kernel, up=False):
        x = modulated_conv2d_layer(x, dlatents_in[:, layer_idx], fmaps=fmaps, kernel=kernel, up=up, resample_kernel=resample_kernel, fused_modconv=fused_modconv)
        if randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        x += noise * tf.cast(noise_strength, x.dtype)
        return apply_bias_act(x, act=act)

    # Building blocks for main layers.
    def block(x, res): # res = 3..resolution_log2
        t = x
        with tf.variable_scope('Conv0_up'):
            x = layer(x, layer_idx=res*2-5, fmaps=nf(res-1), kernel=3, up=True)
        with tf.variable_scope('Conv1'):
            x = layer(x, layer_idx=res*2-4, fmaps=nf(res-1), kernel=3)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-1), kernel=1, up=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def torgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope('ToRGB'):
            t = apply_bias_act(modulated_conv2d_layer(x, dlatents_in[:, res*2-3], fmaps=num_channels, kernel=1, demodulate=False, fused_modconv=fused_modconv))
            return t if y is None else y + t
    def to_wavelet(x,y,levels, res):
        #x - 512xresxres, y-3xresxres
        #Change to descrete wavlet transform form 12xresxres
        with tf.variable_scope('ToWavelet'):
            t = apply_bias_act(modulated_conv2d_layer(x, dlatents_in[:, res*2-3], fmaps=num_channels * (4**levels), kernel=1, demodulate=False, fused_modconv=fused_modconv))
            return t if y is None else y + t

    def add_channels(y, levels, name='Add_Channels', reuse=False):
        # y-3xresxres
        with tf.variable_scope(name, reuse=reuse and tf.compat.v1.AUTO_REUSE):
            y = apply_bias_act(conv2d_layer(y, fmaps=num_channels * (4**levels), kernel=1))
            return y


    def inverse_wavelet(y, levels):
        # Inverse descrete wavelet transformation
        #input: 3 * (2**levels) x base_res x base_res, output: 3x base_res * (2**levels) x base_res * (2**levels)
        with tf.variable_scope('IWT'):
            y = multi_level_iwt(y, levels)
            return y

    # Early layers.
    power = math.log2(shape[2])
    even = power%2 == 0
    resid = 0 if even or decomp_level==1 else 1
    y = None

    initial_decomp = decomp_level if resid == 0 else 1

    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1])
        with tf.variable_scope('Conv'):
            x = layer(x, layer_idx=0, fmaps=nf(1), kernel=3)
        if architecture == 'skip':
            y = torgb(x, y, 2)
        if architecture == 'wavelets':
            # if resid==0:
                y = to_wavelet(x,y,initial_decomp,2)

    # Main layers.
    images_out = []

    for res in range(3, resolution_log2):
        initial = resid==1 and res==3

        res_string    = '%dx%d' % (2**res, 2**res)
        channel_scope = '%s/Add_Channels' % ("" if reuse_add_weights else (res_string + "/"))
        if initial:
            with tf.variable_scope(res_string):
                x = block(x, res)
                y = inverse_wavelet(y, initial_decomp)

            images_out.append(y)
            y = add_channels(y, decomp_level, name=channel_scope, reuse=reuse_add_weights)

            with tf.variable_scope(res_string):
                y = to_wavelet(x, y, decomp_level, res)

            continue

        up_step = res % decomp_level == resid
        
        if not up_step:
            x = upsample_2d(x)
            continue

        if architecture == 'wavelets':
            with tf.variable_scope(res_string):
                x = block(x, res)
                y = inverse_wavelet(y, decomp_level)

            images_out.append(y)
            y = add_channels(y, decomp_level, name=channel_scope, reuse=reuse_add_weights)

            with tf.variable_scope(res_string):
                y = to_wavelet(x, y, decomp_level, res)

            if resid==1 and res==resolution_log2-2:
                break

    y = inverse_wavelet(y, decomp_level)
    images_out.append(y)

    for image in images_out:
        assert image.dtype == tf.as_dtype(dtype)

    if not multi_output:
        return tf.identity(images_out[-1], name='images_out')

    return images_out

#----------------------------------------------------------------------------
# StyleGAN2_wavelets2 synthesis network. Non-learnable weights

def G_synthesis_wavelets(
        dlatents_in,  # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        dlatent_size=512,  # Disentangled latent (W) dimensionality.
        num_channels=3,  # Number of output color channels.
        resolution=1024,  # Output resolution.
        fmap_base=16 << 10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        randomize_noise=True,
        # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        architecture='skip',  # Architecture: 'orig', 'skip', 'resnet', 'wavelets'.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations. None = no filtering.
        fused_modconv=True,  # Implement modulated_conv2d_layer() as a single fused op?
        multi_output=False,  # Output a list of images from all stages of the network
        **_kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4

    def nf(stage):
        return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

    assert architecture in ['orig', 'skip', 'resnet', 'wavelets']
    act = nonlinearity
    num_layers = resolution_log2 * 2 - 2
    images_out = None

    # Primary inputs.
    dlatents_in.set_shape([None, num_layers, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)

    # Noise inputs.
    noise_inputs = []
    for layer_idx in range(num_layers - 1):
        res = (layer_idx + 5) // 2
        shape = [1, 1, 2 ** res, 2 ** res]
        noise_inputs.append(
            tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(),
                            trainable=False))

    # Single convolution layer with all the bells and whistles.
    def layer(x, layer_idx, fmaps, kernel, up=False):
        x = modulated_conv2d_layer(x, dlatents_in[:, layer_idx], fmaps=fmaps, kernel=kernel, up=up,
                                   resample_kernel=resample_kernel, fused_modconv=fused_modconv)
        if randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        x += noise * tf.cast(noise_strength, x.dtype)
        return apply_bias_act(x, act=act)

    # Building blocks for main layers.
    def block(x, res):  # res = 3..resolution_log2
        t = x
        with tf.variable_scope('Conv0_up'):
            x = layer(x, layer_idx=res * 2 - 5, fmaps=nf(res - 1), kernel=3, up=True)
        with tf.variable_scope('Conv1'):
            x = layer(x, layer_idx=res * 2 - 4, fmaps=nf(res - 1), kernel=3)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res - 1), kernel=1, up=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def upsample(y):
        with tf.variable_scope('Upsample'):
            return upsample_2d(y, k=resample_kernel)

    def torgb(x, y, res):  # res = 2..resolution_log2
        with tf.variable_scope('ToRGB'):
            t = apply_bias_act(
                modulated_conv2d_layer(x, dlatents_in[:, res * 2 - 3], fmaps=num_channels, kernel=1, demodulate=False,
                                       fused_modconv=fused_modconv))
            return t if y is None else y + t

    def to_wavelet(x, y, res):
        # x - 512xresxres, y-3xresxres
        # Change to descrete wavlet transform form 12xresxres
        with tf.variable_scope('ToWavelet'):
            t = apply_bias_act(modulated_conv2d_layer(x, dlatents_in[:, res * 2 - 3], fmaps=num_channels * 4, kernel=3,
                                                      demodulate=False, fused_modconv=fused_modconv))
            return t if y is None else y + t

    def inverse_wavelet(y):
        # Inverse descrete wavelet transformation
        # input: 12xresxres, output: 3xres*2xres*2
        with tf.variable_scope('IWT'):
            y = iwt(y)
            return y

    def wavelet_transform(y):
        # descrete wavelet transformation
        # input: 3xresxres, output: 12xres/2xres/2
        with tf.variable_scope('DWT'):
            y = dwt(y)
            return y

    # Early layers.
    y = None
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1])
        with tf.variable_scope('Conv'):
            x = layer(x, layer_idx=0, fmaps=nf(1), kernel=3)
        if architecture == 'skip':
            y = torgb(x, y, 2)
        if architecture == 'wavelets':
            y = to_wavelet(x, y, 2)

    # Main layers.
    images_out = []
    for res in range(3, resolution_log2):
        with tf.variable_scope('%dx%d' % (2 ** res, 2 ** res)):
            x = block(x, res)
            if architecture == 'wavelets':
                y = inverse_wavelet(y)
                images_out.append(y)
                y = upsample(y)
                y = wavelet_transform(y)
                y = to_wavelet(x, y, res)
    y = inverse_wavelet(y)
    images_out.append(y)

    for image in images_out:
        assert image.dtype == tf.as_dtype(dtype)

    if not multi_output:
        return tf.identity(images_out[-1], name='images_out')

    return images_out


#----------------------------------------------------------------------------
# StyleGAN2 synthesis network with wavelets only at the last stage. Non-learnable weights

def G_synthesis_wavelets_cap(
        dlatents_in,  # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        dlatent_size=512,  # Disentangled latent (W) dimensionality.
        num_channels=3,  # Number of output color channels.
        resolution=1024,  # Output resolution.
        fmap_base=16 << 10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        randomize_noise=True,
        # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        architecture='skip',  # Architecture: 'orig', 'skip', 'resnet', 'wavelets'.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations. None = no filtering.
        fused_modconv=True,  # Implement modulated_conv2d_layer() as a single fused op?
        multi_output=False,  # Output a list of images from all stages of the network
        **_kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4

    def nf(stage):
        return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

    assert architecture in ['orig', 'skip', 'resnet', 'wavelets']
    act = nonlinearity
    num_layers = resolution_log2 * 2 - 2
    images_out = None

    # Primary inputs.
    dlatents_in.set_shape([None, num_layers, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)

    # Noise inputs.
    noise_inputs = []
    for layer_idx in range(num_layers - 1):
        res = (layer_idx + 5) // 2
        shape = [1, 1, 2 ** res, 2 ** res]
        noise_inputs.append(
            tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(),
                            trainable=False))

    # Single convolution layer with all the bells and whistles.
    def layer(x, layer_idx, fmaps, kernel, up=False):
        x = modulated_conv2d_layer(x, dlatents_in[:, layer_idx], fmaps=fmaps, kernel=kernel, up=up,
                                   resample_kernel=resample_kernel, fused_modconv=fused_modconv)
        if randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        
        if x.shape[2] != noise.shape[2]:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
            
        x += noise * tf.cast(noise_strength, x.dtype)
        return apply_bias_act(x, act=act)

    # Building blocks for main layers.
    def block(x, res):  # res = 3..resolution_log2
        t = x
        with tf.variable_scope('Conv0_up'):
            x = layer(x, layer_idx=res * 2 - 5, fmaps=nf(res - 1), kernel=3, up=True)
        with tf.variable_scope('Conv1'):
            x = layer(x, layer_idx=res * 2 - 4, fmaps=nf(res - 1), kernel=3)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res - 1), kernel=1, up=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def upsample(y):
        with tf.variable_scope('Upsample'):
            return upsample_2d(y, k=resample_kernel)

    def torgb(x, y, res):  # res = 2..resolution_log2
        with tf.variable_scope('ToRGB'):
            t = apply_bias_act(
                modulated_conv2d_layer(x, dlatents_in[:, res * 2 - 3], fmaps=num_channels, kernel=1, demodulate=False,
                                       fused_modconv=fused_modconv))
            return t if y is None else y + t

    def to_wavelet(x, y, res):
        # x - 512xresxres, y-3xresxres
        # Change to descrete wavlet transform form 12xresxres
        with tf.variable_scope('ToWavelet'):
            t = apply_bias_act(modulated_conv2d_layer(x, dlatents_in[:, res * 2 - 3], fmaps=num_channels * 4, kernel=1,
                                                      demodulate=False, fused_modconv=fused_modconv))
            return t if y is None else y + t

    def inverse_wavelet(y):
        # Inverse descrete wavelet transformation
        # input: 12xresxres, output: 3xres*2xres*2
        with tf.variable_scope('IWT'):
            y = iwt(y)
            return y

    def wavelet_transform(y):
        # descrete wavelet transformation
        # input: 3xresxres, output: 12xres/2xres/2
        with tf.variable_scope('DWT'):
            y = dwt(y)
            return y

    # Early layers.
    y = None
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1])
        with tf.variable_scope('Conv'):
            x = layer(x, layer_idx=0, fmaps=nf(1), kernel=3)
        if architecture == 'skip':
            y = torgb(x, y, 2)
        if architecture == 'wavelets':
            y = torgb(x, y, 2)

    images_out = []
    for res in range(3, resolution_log2 - 1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            x = block(x, res)
            images_out.append(y)
            if architecture == 'wavelets':
                y = upsample(y)
            if architecture == 'wavelets' or res == resolution_log2:
                y = torgb(x, y, res)

    temp_res = resolution_log2
    with tf.variable_scope('%dx%d' % (2**temp_res, 2**temp_res)):
        temp_x = block(x, res)
        images_out.append(y)
        y = upsample(y)
        y = torgb(temp_x, y, temp_res)

    # Main layers.
    for res in range(resolution_log2 - 1, resolution_log2):
        with tf.variable_scope('%dx%d_wavelet' % (2 ** res, 2 ** res)):
            x = block(x, res)
            if architecture == 'wavelets':
                # y = inverse_wavelet(y)
                images_out.append(y)
                y = upsample(y)
                y = wavelet_transform(y)
                y = to_wavelet(x, y, res)
    
    y = inverse_wavelet(y)
    images_out.append(y)

    for image in images_out:
        assert image.dtype == tf.as_dtype(dtype)

    if not multi_output:
        return tf.identity(images_out[-1], name='images_out')

    return images_out

#----------------------------------------------------------------------------
# Original StyleGAN discriminator.
# Used in configs B-D (Table 1).

def D_stylegan(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,         # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    structure           = 'auto',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    if structure == 'auto': structure = 'linear' if is_template_graph else 'recursive'
    act = nonlinearity

    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks for spatial layers.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=1), act=act)
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            with tf.variable_scope('Conv0'):
                x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=3), act=act)
            with tf.variable_scope('Conv1_down'):
                x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True, resample_kernel=resample_kernel), act=act)
            return x

    # Fixed structure: simple and efficient, but does not support progressive growing.
    if structure == 'fixed':
        x = fromrgb(images_in, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            x = block(x, res)

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            with tf.variable_scope('Downsample_lod%d' % lod):
                img = downsample_2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = tflib.lerp_clip(x, y, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
        def grow(res, lod):
            x = lambda: fromrgb(naive_downsample_2d(images_in, factor=2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            y = cset(y, (lod_in > lod), lambda: tflib.lerp(x, fromrgb(naive_downsample_2d(images_in, factor=2**(lod+1)), res - 1), lod_in - lod))
            return y()
        x = grow(3, resolution_log2 - 3)

    # Final layers at 4x4 resolution.
    with tf.variable_scope('4x4'):
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        x = apply_bias_act(dense_layer(x, fmaps=max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True)
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out

#----------------------------------------------------------------------------
# StyleGAN2 discriminator (Figure 7).
# Implements skip connections and residual nets (Figure 7), but no progressive growing.
# Used in configs E-F (Table 1).

def D_stylegan2(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,         # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity

    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)

    # Building blocks for main layers.
    def fromrgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res-1), kernel=1), act=act)
            return t if x is None else x + t
    def block(x, res): # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=3), act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True, resample_kernel=resample_kernel), act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-2), kernel=1, down=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x
    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    # Main layers.
    x = None
    y = images_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample(y)

    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        x = apply_bias_act(dense_layer(x, fmaps=max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True)
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out



#----------------------------------------------------------------------------
#Discriminator architecture for G1 learnable params
def D_stylegan2_wavelets_NU(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,         # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet', 'wavelets'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    decomp_level        = 1,            # Wavelet decomposition level
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet', 'wavelets']
    act = nonlinearity

    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)

    # Building blocks for main layers.
    def block(x, res): # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=3), act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True, resample_kernel=resample_kernel), act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-2), kernel=1, down=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def from_wavelets(x, y, res, down=True): # res = 2..resolution_log2
        with tf.variable_scope('FromWavelet'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res-1), kernel=3, down=down), act=act)
            return t if x is None else x + t

    def remove_channels(y):
        # y-3xresxres
        with tf.variable_scope('Add_Channels'):
            y = apply_bias_act(conv2d_layer(y, fmaps=num_channels, kernel=1, down=False))
        with tf.variable_scope('Add_Channels_down2'):
            y = apply_bias_act(conv2d_layer(y, fmaps=num_channels, kernel=1, down=False))
            return y

    def wavelet_transform(y, levels):
        # descrete wavelet transformation
        # input: 3xresxres, output: 3 * (2**levels) x res / (2**levels) x res / (2**levels)
        with tf.variable_scope('DWT'):
            y = multi_level_dwt(y, levels)
            return y

    # Main layers.

    even = resolution_log2 % 2 == 0
    resid = 0 if even or decomp_level==1 else 1

    x = None
    y = images_in
    y = wavelet_transform(y, decomp_level)
    down = True
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            
            initial   = resid==1 and res==3

            down_step = res % decomp_level == resid            

            if not initial and not down_step:
                x = downsample_2d(x)
                continue
            
            if architecture == 'wavelets' or res == resolution_log2:
                x = from_wavelets(x, y, res, down=down)
                down = True
            x = block(x, res)

            if architecture == 'wavelets':
                actual_decomp_level = decomp_level if not initial else 1
                y = remove_channels(y)
                y = wavelet_transform(y, actual_decomp_level)

    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'wavelets':
            x = from_wavelets(x, y, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        x = apply_bias_act(dense_layer(x, fmaps=max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True)
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out

#----------------------------------------------------------------------------
def D_stylegan2_wavelets(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,         # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet', 'wavelets'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet', 'wavelets']
    act = nonlinearity

    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)

    # Building blocks for main layers.

    def block(x, res): # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=3), act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True, resample_kernel=resample_kernel), act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-2), kernel=1, down=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x
    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    def from_wavelets(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope('FromWavelet'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res-1), kernel=1), act=act)
            return t if x is None else x + t
    def inverse_wavelet(y):
        # Inverse descrete wavelet transformation
        #input: 12xresxres, output: 3xres*2xres*2
        with tf.variable_scope('IWT'):
            y = iwt(y)
            return y
    def wavelet_transform(y):
        # descrete wavelet transformation
        #input: 3xresxres, output: 12xres/2xres/2
        with tf.variable_scope('DWT'):
            y = dwt(y)
            return y

    # Main layers.
    x = None
    y = images_in
    y = wavelet_transform(y)
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'wavelets' or res == resolution_log2:
                x = from_wavelets(x, y, res)
            x = block(x, res)
            if architecture == 'wavelets':
                y = inverse_wavelet(y)
                y = downsample(y)
                y = wavelet_transform(y)
    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'wavelets':
            x = from_wavelets(x, y, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        x = apply_bias_act(dense_layer(x, fmaps=max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True)
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out


#----------------------------------------------------------------------------
#D1 for added supervision for every resolution
def D_multi_level_wavelets1(
    multi_images_in,                          # First input: List of multi-res images [[minibatch, channel, height, width]].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,         # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet', 'wavelets'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    decomp_level        = 1,             #Wavelet decomposition level
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet', 'wavelets']
    act = nonlinearity

    for idx, res_images in enumerate(multi_images_in):
        res_images.set_shape([None, num_channels, 2**(idx+3), 2**(idx+3)])
        multi_images_in[idx] = tf.cast(res_images, dtype)

    labels_in.set_shape([None, label_size])
    labels_in = tf.cast(labels_in, dtype)

    # Building blocks for main layers.
    def block(x, res): # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=3), act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True, resample_kernel=resample_kernel), act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-2), kernel=1, down=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x
    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    def from_wavelets(x, y, res, name): # res = 2..resolution_log2
        with tf.variable_scope('FromWavelet%s'%name):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res-1), kernel=1, down=True), act=act)
            return t if x is None else x + t

    def add_channels(y, levels):
        # y-3xresxres
        with tf.variable_scope('Add_Channels'):
            y = apply_bias_act(conv2d_layer(y, fmaps=num_channels * (4 ** levels), kernel=1, down=True))
        with tf.variable_scope('Add_Channels_down2'):
            y = apply_bias_act(conv2d_layer(y, fmaps=num_channels * (4 ** levels), kernel=1, down=True))
            return y

    def inverse_wavelet(y, levels):
        # Inverse descrete wavelet transformation
        # input: 3 * (2**levels) x base_res x base_res, output: 3x base_res * (2**levels) x base_res * (2**levels)
        with tf.variable_scope('IWT'):
            y = multi_level_iwt(y, levels)
            return y

    def wavelet_transform(y, levels):
        # descrete wavelet transformation
        # input: 3xresxres, output: 3 * (2**levels) x res / (2**levels) x res / (2**levels)
        with tf.variable_scope('DWT'):
            y = multi_level_dwt(y, levels)
            return y

    # Main layers.
    x = None
    y = multi_images_in[-1] # for main downsapling path
    y = wavelet_transform(y, decomp_level)
    down = True
    for depth_idx, res in enumerate(range(resolution_log2, 2, -1)):

        y_res = multi_images_in[-(depth_idx+1)]
        y_res = wavelet_transform(y_res, 1)

        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'wavelets' or res == resolution_log2:
                with tf.variable_scope('Concat'):
                    x = tf.concat([from_wavelets(x, y, res, 'D'), from_wavelets(x, y_res, res,'G')], axis=1)

            x = block(x, res)
            if architecture == 'wavelets':
                y = inverse_wavelet(y, decomp_level)
                y = add_channels(y, decomp_level)

    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'wavelets':
            x = from_wavelets(x, y, 2, 'D')
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        x = apply_bias_act(dense_layer(x, fmaps=max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True)
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out

#D2 for added supervision for every resolution
def D_multi_level_wavelets2(
    multi_images_in,                    # First input: List of multi-res images [[minibatch, channel, height, width]].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,         # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet', 'wavelets'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    **_kwargs):                         # Ignore unrecognized keyword args.



    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet', 'wavelets']
    act = nonlinearity


    for idx, res_images in enumerate(multi_images_in):
        res_images.set_shape([None, num_channels, 2**(idx+3), 2**(idx+3)])
        multi_images_in[idx] = tf.cast(res_images, dtype)

    labels_in.set_shape([None, label_size])
    labels_in = tf.cast(labels_in, dtype)

    # Building blocks for main layers.
    def block(x, res): # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=3), act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True, resample_kernel=resample_kernel), act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-2), kernel=1, down=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    def from_wavelets(x, y, res, name): # res = 2..resolution_log2
        with tf.variable_scope('FromWavelet%s'%name):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res-1), kernel=1), act=act)
            return t if x is None else x + t

    def inverse_wavelet(y):
        # Inverse descrete wavelet transformation
        #input: 12xresxres, output: 3xres*2xres*2
        with tf.variable_scope('IWT'):
            y = iwt(y)
            return y

    def wavelet_transform(y):
        # descrete wavelet transformation
        #input: 3xresxres, output: 12xres/2xres/2
        with tf.variable_scope('DWT'):
            y = dwt(y)
            return y

    # Main layers.
    x = None
    y = multi_images_in[-1] # for main downsapling path
    y = wavelet_transform(y)
    for depth_idx, res in enumerate(range(resolution_log2, 2, -1)):

        y_res = multi_images_in[-(depth_idx+1)]
        y_res = wavelet_transform(y_res)

        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'wavelets' or res == resolution_log2:
                with tf.variable_scope('Concat'):
                    x = tf.concat([from_wavelets(x, y, res, 'D'), from_wavelets(x, y_res, res,'G')], axis=1)
            x = block(x, res)
            if architecture == 'wavelets':
                y = inverse_wavelet(y)
                y = downsample(y)
                y = wavelet_transform(y)
    # Final layers.
    with tf.variable_scope('4x4'):
        #y_res = images_in[4]
        #y_res = wavelet_transform(y_res)

        if architecture == 'wavelets':
            #x = tf.concat([from_wavelets(x, y, 2, 'D'), from_wavelets(x, y_res, 2, 'G')], axis=1)
            x = from_wavelets(x, y, 2, 'D')

        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        x = apply_bias_act(dense_layer(x, fmaps=max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True)
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out

def D_multi_disc_wavelets(
    multi_images_in,                    # First input: List of multi-res images [[minibatch, channel, height, width]].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,         # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet', 'wavelets'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    shared_layers       = False,        # Enable to share layers between discriminators
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet', 'wavelets']
    act = nonlinearity


    for idx, res_images in enumerate(multi_images_in):
        res_images.set_shape([None, num_channels, 2**(idx+3), 2**(idx+3)])
        multi_images_in[idx] = tf.cast(res_images, dtype)

    labels_in.set_shape([None, label_size])
    labels_in = tf.cast(labels_in, dtype)

    # Building blocks for main layers.
    def block(x, res): # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=3), act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True, resample_kernel=resample_kernel), act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-2), kernel=1, down=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    def from_wavelets(x, y, res, name): # res = 2..resolution_log2
        with tf.variable_scope('FromWavelet%s'%name):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res-1), kernel=1), act=act)
            return t if x is None else x + t

    def inverse_wavelet(y):
        # Inverse descrete wavelet transformation
        #input: 12xresxres, output: 3xres*2xres*2
        with tf.variable_scope('IWT'):
            y = iwt(y)
            return y

    def wavelet_transform(y):
        # descrete wavelet transformation
        #input: 3xresxres, output: 12xres/2xres/2
        with tf.variable_scope('DWT'):
            y = dwt(y)
            return y

    # Main layers.

    scores_out = 0.
    for idx, image in enumerate(reversed(multi_images_in)):
        reuse_flag   = idx > 0 and not shared_layers
        scope_suffix = "_%s" % (idx, ) if shared_layers else ""

        # base_res = 2**(idx+3)
        base_res_log2 = (resolution_log2 - idx)
        # print(base_res_log2)
        x = None
        y = image # for main downsapling path
        y = wavelet_transform(y)
        for depth_idx, res in enumerate(range(base_res_log2, 2, -1)):
            with tf.variable_scope('%dx%d' % (2**res, 2**res) + scope_suffix, reuse=reuse_flag):
                # print(idx)
                # print(architecture)
                if architecture == 'wavelets' or res == base_res_log2:
                    # print("in if")
                    with tf.variable_scope('Concat'):
                        x = from_wavelets(x, y, res, 'D')

                x = block(x, res)
                if architecture == 'wavelets':
                    y = inverse_wavelet(y)
                    y = downsample(y)
                    y = wavelet_transform(y)

        # Final layers.
        with tf.variable_scope('4x4' + scope_suffix, reuse=reuse_flag):
            #y_res = images_in[4]
            #y_res = wavelet_transform(y_res)

            if architecture == 'wavelets':
                #x = tf.concat([from_wavelets(x, y, 2, 'D'), from_wavelets(x, y_res, 2, 'G')], axis=1)
                x = from_wavelets(x, y, 2, 'D')

            if mbstd_group_size > 1:
                with tf.variable_scope('MinibatchStddev'):
                    x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
            with tf.variable_scope('Conv'):
                x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
            with tf.variable_scope('Dense0'):
                x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

        # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
        with tf.variable_scope('Output' + scope_suffix, reuse=reuse_flag):
            x = apply_bias_act(dense_layer(x, fmaps=max(labels_in.shape[1], 1)))
            if labels_in.shape[1] > 0:
                x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True)
        scores_out += x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out    