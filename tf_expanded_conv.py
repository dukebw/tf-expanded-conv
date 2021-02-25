import contextlib
import functools
import os

import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim


tf.disable_v2_behavior()


def _fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.

    Pads the input such that if it was used in a convolution with 'VALID' padding,
    the output would have the same dimensions as if the unpadded input was used
    in a convolution with 'SAME' padding.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
      rate: An integer, rate for atrous convolution.

    Returns:
      output: A tensor of size [batch, height_out, width_out, channels] with the
        input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    kernel_size_effective = [
        kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
        kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
    ]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg[0], pad_end[0]], [pad_beg[1], pad_end[1]], [0, 0]]
    )
    return padded_inputs


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _split_divisible(num, num_ways, divisible_by=8):
    """Evenly splits num, num_ways so each piece is a multiple of divisible_by."""
    assert num % divisible_by == 0
    assert num / num_ways >= divisible_by
    # Note: want to round down, we adjust each split to match the total.
    base = num // num_ways // divisible_by * divisible_by
    result = []
    accumulated = 0
    for i in range(num_ways):
        r = base
        while accumulated + r < num * (i + 1) / num_ways:
            r += divisible_by
        result.append(r)
        accumulated += r
    assert accumulated == num
    return result


@contextlib.contextmanager
def _set_arg_scope_defaults(defaults):
    """Sets arg scope defaults for all items present in defaults.

    Args:
      defaults: dictionary/list of pairs, containing a mapping from
      function to a dictionary of default args.

    Yields:
      context manager where all defaults are set.
    """
    if hasattr(defaults, "items"):
        items = list(defaults.items())
    else:
        items = defaults
    if not items:
        yield
    else:
        func, default_arg = items[0]
        with slim.arg_scope(func, **default_arg):
            with _set_arg_scope_defaults(items[1:]):
                yield


def expand_input_by_factor(n, divisible_by=8):
    return lambda num_inputs, **_: _make_divisible(num_inputs * n, divisible_by)


def split_conv(input_tensor, num_outputs, num_ways, scope, divisible_by=8, **kwargs):
    """Creates a split convolution.

    Split convolution splits the input and output into
    'num_blocks' blocks of approximately the same size each,
    and only connects $i$-th input to $i$ output.

    Args:
      input_tensor: input tensor
      num_outputs: number of output filters
      num_ways: num blocks to split by.
      scope: scope for all the operators.
      divisible_by: make sure that every part is divisiable by this.
      **kwargs: will be passed directly into conv2d operator
    Returns:
      tensor
    """
    b = input_tensor.get_shape().as_list()[3]

    if num_ways == 1 or min(b // num_ways, num_outputs // num_ways) < divisible_by:
        # Don't do any splitting if we end up with less than 8 filters
        # on either side.
        return slim.conv2d(input_tensor, num_outputs, [1, 1], scope=scope, **kwargs)

    outs = []
    input_splits = _split_divisible(b, num_ways, divisible_by=divisible_by)
    output_splits = _split_divisible(num_outputs, num_ways, divisible_by=divisible_by)
    inputs = tf.split(input_tensor, input_splits, axis=3, name="split_" + scope)
    base = scope
    for i, (input_tensor, out_size) in enumerate(zip(inputs, output_splits)):
        scope = base + "_part_%d" % (i,)
        n = slim.conv2d(input_tensor, out_size, [1, 1], scope=scope, **kwargs)
        n = tf.identity(n, scope + "_output")
        outs.append(n)
    return tf.concat(outs, 3, name=scope + "_concat")


@slim.add_arg_scope
def expanded_conv(
    input_tensor,
    num_outputs,
    expansion_size=expand_input_by_factor(6),
    stride=1,
    rate=1,
    kernel_size=(3, 3),
    residual=True,
    normalizer_fn=None,
    split_projection=1,
    split_expansion=1,
    split_divisible_by=8,
    expansion_transform=None,
    depthwise_location="expansion",
    depthwise_channel_multiplier=1,
    use_explicit_padding=False,
    padding="SAME",
    inner_activation_fn=None,
    depthwise_activation_fn=None,
    project_activation_fn=tf.identity,
    depthwise_fn=slim.separable_conv2d,
    expansion_fn=split_conv,
    projection_fn=split_conv,
    scope=None,
):
    """Depthwise Convolution Block with expansion.

    Builds a composite convolution that has the following structure
    expansion (1x1) -> depthwise (kernel_size) -> projection (1x1)

    Args:
      input_tensor: input
      num_outputs: number of outputs in the final layer.
      expansion_size: the size of expansion, could be a constant or a callable.
        If latter it will be provided 'num_inputs' as an input. For forward
        compatibility it should accept arbitrary keyword arguments.
        Default will expand the input by factor of 6.
      stride: depthwise stride
      rate: depthwise rate
      kernel_size: depthwise kernel
      residual: whether to include residual connection between input
        and output.
      normalizer_fn: batchnorm or otherwise
      split_projection: how many ways to split projection operator
        (that is conv expansion->bottleneck)
      split_expansion: how many ways to split expansion op
        (that is conv bottleneck->expansion) ops will keep depth divisible
        by this value.
      split_divisible_by: make sure every split group is divisible by this number.
      expansion_transform: Optional function that takes expansion
        as a single input and returns output.
      depthwise_location: where to put depthwise covnvolutions supported
        values None, 'input', 'output', 'expansion'
      depthwise_channel_multiplier: depthwise channel multiplier:
      each input will replicated (with different filters)
      that many times. So if input had c channels,
      output will have c x depthwise_channel_multpilier.
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      padding: Padding type to use if `use_explicit_padding` is not set.
      inner_activation_fn: activation function to use in all inner convolutions.
      If none, will rely on slim default scopes.
      depthwise_activation_fn: activation function to use for deptwhise only.
        If not provided will rely on slim default scopes. If both
        inner_activation_fn and depthwise_activation_fn are provided,
        depthwise_activation_fn takes precedence over inner_activation_fn.
      project_activation_fn: activation function for the project layer.
      (note this layer is not affected by inner_activation_fn)
      depthwise_fn: Depthwise convolution function.
      expansion_fn: Expansion convolution function. If use custom function then
        "split_expansion" and "split_divisible_by" will be ignored.
      projection_fn: Projection convolution function. If use custom function then
        "split_projection" and "split_divisible_by" will be ignored.

      scope: optional scope.

    Returns:
      Tensor of depth num_outputs

    Raises:
      TypeError: on inval
    """
    conv_defaults = {}
    dw_defaults = {}
    if inner_activation_fn is not None:
        conv_defaults["activation_fn"] = inner_activation_fn
        dw_defaults["activation_fn"] = inner_activation_fn
    if depthwise_activation_fn is not None:
        dw_defaults["activation_fn"] = depthwise_activation_fn
    # pylint: disable=g-backslash-continuation
    with tf.variable_scope(scope, default_name="expanded_conv") as s, tf.name_scope(
        s.original_name_scope
    ), slim.arg_scope((slim.conv2d,), **conv_defaults), slim.arg_scope(
        (slim.separable_conv2d,), **dw_defaults
    ):
        prev_depth = input_tensor.get_shape().as_list()[3]
        if depthwise_location not in [None, "input", "output", "expansion"]:
            raise TypeError(
                "%r is unknown value for depthwise_location" % depthwise_location
            )
        if use_explicit_padding:
            if padding != "SAME":
                raise TypeError(
                    "`use_explicit_padding` should only be used with " '"SAME" padding.'
                )
            padding = "VALID"
        depthwise_func = functools.partial(
            depthwise_fn,
            num_outputs=None,
            kernel_size=kernel_size,
            depth_multiplier=depthwise_channel_multiplier,
            stride=stride,
            rate=rate,
            normalizer_fn=normalizer_fn,
            padding=padding,
            scope="depthwise",
        )
        # b1 -> b2 * r -> b2
        #   i -> (o * r) (bottleneck) -> o
        input_tensor = tf.identity(input_tensor, "input")
        net = input_tensor

        if depthwise_location == "input":
            if use_explicit_padding:
                net = _fixed_padding(net, kernel_size, rate)
            net = depthwise_func(net, activation_fn=None)
            net = tf.identity(net, name="depthwise_output")

        if callable(expansion_size):
            inner_size = expansion_size(num_inputs=prev_depth)
        else:
            inner_size = expansion_size

        # NOTE(brendan): workaround to match PyTorch MobileNetV2
        # if inner_size > net.shape[3]:
        if True:
            if expansion_fn == split_conv:
                expansion_fn = functools.partial(
                    expansion_fn,
                    num_ways=split_expansion,
                    divisible_by=split_divisible_by,
                    stride=1,
                )
            net = expansion_fn(
                net, inner_size, scope="expand", normalizer_fn=normalizer_fn
            )
            net = tf.identity(net, "expansion_output")

        if depthwise_location == "expansion":
            if use_explicit_padding:
                net = _fixed_padding(net, kernel_size, rate)
            net = depthwise_func(net)
            net = tf.identity(net, name="depthwise_output")

        if expansion_transform:
            net = expansion_transform(expansion_tensor=net, input_tensor=input_tensor)
        # Note in contrast with expansion, we always have
        # projection to produce the desired output size.
        if projection_fn == split_conv:
            projection_fn = functools.partial(
                projection_fn,
                num_ways=split_projection,
                divisible_by=split_divisible_by,
                stride=1,
            )
        net = projection_fn(
            net,
            num_outputs,
            scope="project",
            normalizer_fn=normalizer_fn,
            activation_fn=project_activation_fn,
        )
        if depthwise_location == "output":
            if use_explicit_padding:
                net = _fixed_padding(net, kernel_size, rate)
            net = depthwise_func(net, activation_fn=None)
            net = tf.identity(net, name="depthwise_output")

        if callable(residual):  # custom residual
            net = residual(input_tensor=input_tensor, output_tensor=net)
        elif (
            residual
            and
            # stride check enforces that we don't add residuals when spatial
            # dimensions are None
            stride == 1
            and
            # Depth matches
            net.get_shape().as_list()[3] == input_tensor.get_shape().as_list()[3]
        ):
            net += input_tensor
        return tf.identity(net, name="output")


def tf_expanded_conv():
    dummy_input = tf.placeholder(dtype=tf.float32, shape=(1, 32, 32, 3))
    defaults = {
        (slim.batch_norm,): {"center": True, "scale": True, "epsilon": 1e-5},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            "normalizer_fn": slim.batch_norm,
            "activation_fn": tf.nn.relu6,
        },
        (expanded_conv,): {
            "expansion_size": expand_input_by_factor(6),
            "split_expansion": 1,
            "normalizer_fn": slim.batch_norm,
            "residual": True,
            "use_explicit_padding": True,
        },
        (slim.conv2d, slim.separable_conv2d): {"padding": "SAME"},
    }
    with _set_arg_scope_defaults(defaults):
        net = expanded_conv(dummy_input, num_outputs=24, stride=2, rate=1)

    tf_bns = {
        "gamma": [gv for gv in tf.global_variables() if "gamma" in gv.name],
        "beta": [gv for gv in tf.global_variables() if "beta" in gv.name],
        "moving_mean": [gv for gv in tf.global_variables() if "moving_mean" in gv.name],
        "moving_variance": [
            gv for gv in tf.global_variables() if "moving_variance" in gv.name
        ],
    }
    # NOTE(brendan): randomly initialize BN so it doesn't get optimized out
    initializer_ops = []
    for k, tf_bn_param in tf_bns.items():
        for tf_bn_param_for_layer in tf_bn_param:
            init_op = tf.assign(tf_bn_param_for_layer, np.random.randn(24))
            initializer_ops.append(init_op)

    with tf.Session() as sess:
        sess.run([*initializer_ops, tf.global_variables_initializer()])

        tf.saved_model.simple_save(
            sess,
            "tf_expanded_conv",
            inputs={"inputs": dummy_input},
            outputs={"out": net},
        )


if __name__ == "__main__":
    tf_expanded_conv()
