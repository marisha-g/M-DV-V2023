# %%
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import feature_transformed
from tensorflow_probability.python.math.psd_kernels.internal import util

__all__ = ['GeometricAnisotropy']

# %%
class GeometricAnisotropy(feature_transformed.FeatureTransformed):
  """Kernel that first rescales all feature dimensions 
  and then rotates input data with 3D rotation matrix.
  Given a kernel `k`, parameter `scale_diag` and `angle`, and inputs `x` and `y`, 
  this kernel first rescales the input by computing `x / scale_diag` and
  `y / scale_diag`, then calculate the rotation matrix, and passing this to `k`.
  """
  
  def __init__(
      self,
      kernel,
      lambda1=None, 
      lambda2=None,
      angle=None,
      validate_args=False,
      name='GeometricAnisotropy'):
    """Construct a geometric anisotropy kernel instance.
    Args:
      kernel: `PositiveSemidefiniteKernel` instance. Inputs are rescaled and
        passed in to this kernel. Parameters to `kernel` must be broadcastable
        with `scale_diag` and `angle`.

      scale_diag: Floating point `Tensor` that controls how sharp or wide the
        kernel shape is. `scale_diag` must have at least `kernel.feature_ndims`
        dimensions, and extra dimensions must be broadcastable with parameters
        of `kernel`. This is a "diagonal" in the sense that if all the feature
        dimensions were flattened, `scale_diag` acts as the inverse of a
        diagonal matrix.

      inverse_scale_diag: Non-negative floating point `Tensor` that is treated
        as `1 / scale_diag`. Only one of `scale_diag` or `inverse_scale_diag`
        should be provided.
        Default value: None

      angle: Floating point `Tensor` that determines the orientation of the ellipses.
        Default value: None

      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
        
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    if (lambda1 is None):
      raise ValueError(
          'Must specify `lambda1`.')
    if (lambda2 is None):
      raise ValueError(
          'Must specify `lambda2`.')
    if (angle is None):
      raise ValueError(
          'Must specify `angle`.')
    with tf.name_scope(name):
      dtype = tf.float64
      self._lambda1 = tensor_util.convert_nonref_to_tensor(
          lambda1, dtype=dtype, name='lambda1')
      self._lambda2 = tensor_util.convert_nonref_to_tensor(
          lambda2, dtype=dtype, name='lambda2')
      self._angle = tensor_util.convert_nonref_to_tensor(
          angle, dtype=tf.float64, name='angle')
      
      def transform_input(x, feature_ndims, example_ndims):
        """Computes `x / scale_diag`."""
        #assert feature_ndims == 2, "feature_ndims must be 2."
        
        diag_values = tf.stack([self.lambda1, self.lambda2])
        diag_matrix = tf.linalg.tensor_diag(diag_values)
        d = tf.cast(diag_matrix, dtype=tf.float64)

        # Add 2D rotation transformation
        if angle is not None:
          _cos = tf.cos(self.angle) 
          _sin = tf.sin(self.angle)
          cos = tf.cast(_cos, dtype=tf.float64)
          sin = tf.cast(_sin, dtype=tf.float64)

        rotation = tf.stack([cos, -sin,
                             sin, cos])
        rotation = tf.reshape(rotation, (2,2))

        x = tf.matmul(x, rotation)
        x = tf.matmul(x, d)
        return x 

      super(GeometricAnisotropy, self).__init__(
          kernel,
          transformation_fn=transform_input,
          validate_args=validate_args,
          name=name,
          parameters=parameters)

  @property
  def lambda1(self):
    return self._lambda1

  @property
  def lambda2(self):
    return self._lambda2
  
  @property
  def angle(self):
    return self._angle
  
  @classmethod
  def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top
    return dict(
        kernel=parameter_properties.BatchedComponentProperties(),
        lambda1=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.kernel.feature_ndims,
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        lambda2=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.kernel.feature_ndims,
            default_constraining_bijector_fn=softplus.Softplus), 
        angle=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.kernel.feature_ndims, 
            default_constraining_bijector_fn=softplus.Softplus))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if (self._lambda1 is not None and
        is_init != tensor_util.is_ref(self._lambda1)):
      assertions.append(assert_util.assert_non_negative(
          self._lambda1,
          message='`lambda1` must be non-negative.'))
    if (self._lambda2 is not None and
        is_init != tensor_util.is_ref(self._lambda2)):
      assertions.append(assert_util.assert_positive(
          self._lambda2,
          message='`lambda2` must be positive.'))
    return assertions
  
