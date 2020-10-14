# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers
from keras.layers import deserialize as deserialize_layer

from .utils.caps_utils import dict_to_list, list_to_dict
from LVQ import constraints

import numpy as np


class Module(Layer):
    """Module Abstract Class

    Conventions:
        - internally is each input considered as dict input; the internal preprocessing provides an correct
          connection to sub-function. We need another data type to identify uniquely if a input is from a module or not
        - output and input of real module is always a dict (!) and just that ! The dict exist only outside the module
          for the internal layer registration it is again a list of tensors
        - additional inputs like losses are non-dict an basic keras type. It is hard to trace where keras uses these
          inputs after the storage (also true for masks (!) and shapes, updates, losses) (!)
          ! consider this carefully if you call methods which have parameters from both groups like (add_losses)
        - the shape of inputs are considered as lists (without the dict around). dicts haven't a shape property.
        - the first dimension after batch_size is the channel dimension of the capsule:
          (batch_size, channels, dim1, dim2 , dim3, ..., dimN)

    Abstract body for own module:

class MyModule(Module):
    def __init__(self, output_shape, **kwargs):
        self.output_shape = output_shape
        # be sure to call this at the end
        super(MyModule, self).__init__(**kwargs)

    def _call(self, inputs, **kwargs):
        # be sure to call this at the first position
        return inputs

    def _build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

     def _compute_mask(self, inputs, masks=None):
        # if needed implement masking here
        return masks

    def _compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        config = {'layer_init': self._layer_init}
        super_config = super(Module, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))

    """
    def __init__(self,
                 module_input=None,
                 module_output=None,
                 support_sparse_signal=False,
                 support_full_signal=True,
                 **kwargs):
        """init method of a module"""
        # register capsule params here! different to Layer params (!)
        self._needed_kwargs = {'_proto_distrib',
                               '_proto_extension',
                               '_capsule_extension',
                               '_max_proto_number_in_capsule',
                               '_equally_distributed',
                               'proto_number',
                               'capsule_number',
                               'owner',
                               'sparse_signal'}
        self._proto_distrib = None
        self._proto_extension = None
        self._capsule_extension = None
        self._max_proto_number_in_capsule = None
        self._equally_distributed = None
        self.proto_number = None
        self.capsule_number = None
        self.owner = None
        if not hasattr(self, 'sparse_signal'):
            self.sparse_signal = None

        # defines if the signal is dict (module input) or standard keras; is fixed after first module call (except for
        # capsules). It is just fixed automatically if it is None! Since, could be predefined.
        self._module_input = module_input
        self._module_output = module_output

        # set the signal support flags to handle the correct callings
        if not support_sparse_signal and not support_full_signal:
            raise ValueError('At least one must be supported (`True`). You provide `support_sparse_signal` and '
                             '`support_full_signal` with `False`.')
        self._support_sparse_signal = support_sparse_signal
        self._support_full_signal = support_full_signal

        # take kwargs, pop Capsule kwargs out and put the base Layer
        kwargs = self._set_params(**kwargs)

        super(Module, self).__init__(**kwargs)

        # default owner is the instance itself
        if self.owner is None:
            self.owner = self.name

    def __call__(self, inputs, **kwargs):
        """Call command: basic layer call with some pre-processing"""
        if not self.built:
            # after a module is called the input type is fixed; it causes problems to make it dynamic! During the
            # model compile of keras the compute_output_shape is could be called. We can lost the reference to the true
            # value if we keep it dynamic. Further if we keep it dynamic we have to handle in all methods both cases:
            # lists, dicts. (see Keras -> topology -> container -> run_internal_graph)
            if self._module_input is None:
                self._module_input = self._is_module_input(inputs)
            # set output type to input type if not pre-defined
            if self._module_output is None:
                self._module_output = self._is_module_input(inputs)
            # after this capsule params can't be changed
            kwargs = self._set_params(**kwargs)
            # self.name = self.name + '_owner_' + self.owner
        else:
            # capsule params are not transmitted; pop out
            kwargs = self._del_needed_kwargs(**kwargs)

        self._is_callable(inputs)

        # compute outputs (outputs is list, tuple or a keras tensor)
        outputs = super(Module, self).__call__(self._to_layer_input(inputs), **kwargs)

        # Check activity_regularizer and throw error if the attribute exist:
        if hasattr(self, 'activity_regularizer'):
            raise AttributeError("'activity_regularizer' should not be used for capsules, because there is no support "
                                 "to apply different regularizers over the single output entities. Use "
                                 "'output_regularizers' instead.")

        # Apply output_regularizers if any:
        if hasattr(self, 'output_regularizers') and self.output_regularizers is not None:
            if callable(self.output_regularizers):
                if isinstance(outputs, (list, tuple)):
                    regularization_losses = [self.output_regularizers(x) for x in outputs]
                    self.add_loss(regularization_losses, inputs)
                else:
                    self.add_loss([self.output_regularizers(outputs)], inputs)
            else:
                if isinstance(outputs, (list, tuple)):
                    if len(self.output_regularizers) != len(outputs):
                        raise ValueError("The list of 'output_regularizers' must have the same length as outputs. You "
                                         "provide: len(outputs)=" + str(len(outputs)) + " != len(output_regularizers)="
                                         + str(len(self.output_regularizers)))
                    regularization_losses = []
                    for i, x in enumerate(outputs):
                        if self.output_regularizers[i] is not None:
                            regularization_losses.append(self.output_regularizers[i](x))

                    if regularization_losses:
                        self.add_loss(regularization_losses, inputs)
                else:
                    if len(self.output_regularizers) != 1:
                        raise ValueError("The list of 'output_regularizers' must have the same length as outputs. You "
                                         "provide: len(outputs)=" + str(1) + " != len(output_regularizers)="
                                         + str(len(self.output_regularizers)))
                    if self.output_regularizers[0] is not None:
                        self.add_loss([self.output_regularizers[0](outputs)], inputs)

        return self._to_module_output(outputs, module_output=self._module_output)

    def build(self, input_shape):
        # same structure as Layer
        self._is_callable()
        if self.sparse_signal:
            self._build_sparse(input_shape)
        else:
            self._build(input_shape)
        super(Module, self).build(input_shape)

    def _build(self, input_shape):
        # implement your build method here
        pass

    def _build_sparse(self, input_shape):
        # implement your build method here
        pass

    def call(self, inputs, **kwargs):
        was_module_input = self._is_module_input(inputs)
        inputs = self._to_module_input(inputs)
        self._is_callable(inputs)
        if self.sparse_signal:
            outputs = self._call_sparse(inputs, **kwargs)
        else:
            outputs = self._call(inputs, **kwargs)
        return self._to_module_output(outputs, module_output=was_module_input)

    def _call(self, inputs, **kwargs):
        # implement your call method here
        return inputs

    def _call_sparse(self, inputs, **kwargs):
        # implement your call method here
        return inputs

    def compute_output_shape(self, input_shape):
        self._is_callable()
        if self.sparse_signal:
            return self._compute_output_shape_sparse(input_shape)
        else:
            return self._compute_output_shape(input_shape)

    def _compute_output_shape(self, input_shape):
        """implement your output_shape inference here"""
        return input_shape

    def _compute_output_shape_sparse(self, input_shape):
        """implement your output_shape inference here"""
        return input_shape

    def compute_mask(self, inputs, mask=None):
        inputs = self._to_module_input(inputs)
        self._is_callable(inputs)
        # It could be possible that we have to transform the signal back (same structure like call())
        if self.sparse_signal:
            return self._compute_mask_sparse(inputs, mask)
        else:
            return self._compute_mask(inputs, mask)

    def _compute_mask(self, inputs, mask):
        """Computes an output mask tensor.

        Copy of base implementation from Keras.

        # Arguments
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.

        # Returns
            None or a tensor (or list of tensors,
                one per output tensor of the layer).
        """
        if not self.supports_masking:
            if mask is not None:
                if isinstance(mask, list):
                    if any(m is not None for m in mask):
                        raise TypeError('Module ' + self.name + ' does not support masking, but was passed an '
                                        'input_mask: ' + str(mask))
                else:
                    raise TypeError('Module ' + self.name + ' does not support masking, but was passed an input_'
                                    'mask: ' + str(mask))
            # masking not explicitly supported: return None as mask
            return None
        # if masking is explicitly supported, by default
        # carry over the input mask
        return mask

    def _compute_mask_sparse(self, inputs, mask):
        return self._compute_mask(inputs, mask)

    def _set_params(self, **kwargs):
        """Set needed_kwargs of Capsule by **kwargs and return the reduced kwargs list."""
        for key in self._needed_kwargs:
            if key in kwargs:
                # pop key-value-pairs!
                setattr(self, key, kwargs.pop(key))
        # return modified kwargs
        return kwargs

    def _get_params(self):
        """Get needed_kwargs of Capsule"""
        params = {}
        for key in self._needed_kwargs:
            params.update({key: getattr(self, key)})
        return params

    def _is_callable(self, inputs=None):
        """Check if the module is callable: check if it was pre-processed."""
        for key in self._needed_kwargs:
            if getattr(self, key) is None:
                raise ValueError('The module ' + self.name + ' is not proper initialized for calls. Some '
                                 'of the parameters are not unequal None (that\'s the convention).  Be sure that '
                                 'the module is assigned to a proper capsule. Parameter ' + key + ' is None. The '
                                 'owner capsule is :' + str(self.owner))
        if self._module_input is None:
            raise ValueError('The input type of the module is not specified. Can\'t process automatically if the input'
                             ' is a module input or not. This is usually the case if you call a module method which'
                             ' is not callable without internal pre-processing.')
        if self._module_output is None:
            raise ValueError('The output type of the module is not specified. Can\'t process automatically if the '
                             'output is a module output or not. This is usually the case if you call a module method '
                             'which is not callable without internal pre-processing.')
        if inputs is not None:
            if self._module_input != self._is_module_input(inputs):
                raise TypeError('The input type of ' + self.name + ' is ' + ('dict (module signal)' if
                                self._module_input else '[list, tuple, tensor]') + ' and you provide '
                                + str(type(inputs)))
        if self.sparse_signal not in (True, False):
            raise ValueError('Specify if it is a sparse signal or not (True or False). ' + str(self.sparse_signal) +
                             ' is not supported.')
        if not self._support_sparse_signal and self.sparse_signal:
            raise ValueError('The module ' + self.name + " doesn't support sparse signals but you try to call them "
                             "with one.")
        if not self._support_full_signal and not self.sparse_signal:
            raise ValueError('The module ' + self.name + " doesn't support full signals but you try to call them "
                             "with one.")

        return True

    def _del_needed_kwargs(self, **kwargs):
        """del needed_kwargs to avoid spoiling of the Layer methods"""
        for key in self._needed_kwargs:
            if key in kwargs:
                del kwargs[key]
        return kwargs

    def _del_module_args(self, **kwargs):
        """del args to avoid multiple values in kwargs"""
        if 'module_input' in kwargs:
            del kwargs['module_input']
        if 'module_output' in kwargs:
            del kwargs['module_output']
        if 'support_sparse_signal' in kwargs:
            del kwargs['support_sparse_signal']
        if 'support_full_signal' in kwargs:
            del kwargs['support_full_signal']
        return kwargs

    def _to_module_output(self, inputs, module_output):
        """convert a module input (dict) to a layer signal (list or tensor)"""
        if inputs is not None:
            if module_output:
                if not self._is_module_input(inputs):
                    return list_to_dict(inputs)
                else:
                    return inputs
            else:
                if self._is_module_input(inputs):
                    return dict_to_list(inputs)
                else:
                    return inputs
        else:
            return inputs

    def _to_layer_input(self, inputs):
        """transforms the input back to a dict if it was a module signal"""
        if self._module_input:
            if inputs is not None:
                if self._is_module_input(inputs):
                    return dict_to_list(inputs)
                else:
                    return inputs
            else:
                return inputs
        else:
            return inputs

    def _to_module_input(self, inputs):
        """transforms the input back to a dict if it was a module signal"""
        if self._module_input:
            if inputs is not None:
                if not self._is_module_input(inputs):
                    return list_to_dict(inputs)
                else:
                    return inputs
            else:
                return inputs
        else:
            return inputs

    @staticmethod
    def _is_module_input(inputs):
        return isinstance(inputs, dict)

    def assert_input_compatibility(self, inputs):
        self._is_callable()
        super(Module, self).assert_input_compatibility(self._to_layer_input(inputs))

    def get_input_at(self, node_index):
        self._is_callable()
        inputs = super(Module, self).get_input_at(node_index)
        return self._to_module_output(inputs, self._module_input)

    def get_output_at(self, node_index):
        self._is_callable()
        outputs = super(Module, self).get_output_at(node_index)
        return self._to_module_output(outputs, self._module_output)

    @property
    def input(self):
        self._is_callable()
        inputs = super(Module, self).input
        return self._to_module_output(inputs, self._module_input)

    @property
    def output(self):
        self._is_callable()
        outputs = super(Module, self).output
        return self._to_module_output(outputs, self._module_output)

    def add_loss(self, losses, inputs=None):
        self._is_callable()
        inputs = self._to_layer_input(inputs)
        super(Module, self).add_loss(losses, inputs)

    def add_update(self, updates, inputs=None):
        self._is_callable()
        inputs = self._to_layer_input(inputs)
        super(Module, self).add_update(updates, inputs)

    def get_updates_for(self, inputs):
        self._is_callable()
        inputs = self._to_layer_input(inputs)
        return super(Module, self).get_updates_for(inputs)

    def get_losses_for(self, inputs):
        self._is_callable()
        inputs = self._to_layer_input(inputs)
        return super(Module, self).get_losses_for(inputs)

    def get_config(self):
        config = {'module_input': self._module_input,
                  'module_output': self._module_output,
                  'support_sparse_signal': self._support_sparse_signal,
                  'support_full_signal': self._support_full_signal}
        super_config = super(Module, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class _ModuleWrapper(Module):
    """Wrapper to interpret Keras layers as module

    Just a simple wrapper. Should be not used outside of this class. We don't provide direct links to the layer
    properties to avoid sync problems.
    """
    # it is assumed that a layer never deals with dicts of inputs!
    def __init__(self,
                 layer,
                 scope_key,
                 **kwargs):
        self.layer = layer
        self.scope_key = scope_key

        super(_ModuleWrapper, self).__init__(support_sparse_signal=True,
                                             support_full_signal=True,
                                             **self._del_module_args(**kwargs))

    def __call__(self, inputs, **kwargs):
        if not self.built:
            if self._module_input is None:
                self._module_input = self._is_module_input(inputs)
            if self._module_output is None:
                self._module_output = self._is_module_input(inputs)
            # after this capsule params can't be changed
            kwargs = self._set_params(**kwargs)
            # self.layer.name = self.layer.name + '_owner_' + self.owner
        else:
            kwargs = self._del_needed_kwargs(**kwargs)

        self._is_callable(inputs)

        if self._module_input:
            outputs = inputs
            self._check_scope_key(inputs)
            outputs[self.scope_key] = self.layer.__call__(inputs[self.scope_key], **kwargs)
        else:
            outputs = self.layer.__call__(inputs, **kwargs)
        return outputs

    def _build(self, input_shape):
        if self._module_input:
            self._check_scope_key(input_shape)
            self.layer.build(input_shape[self.scope_key])
        else:
            self.layer.build(input_shape)

    def _build_sparse(self, input_shape):
        self._build(input_shape)

    def _call(self, inputs, **kwargs):
        kwargs = self._del_needed_kwargs(**kwargs)
        if self._module_input:
            self._check_scope_key(inputs)
            return inputs.update({self.scope_key: self.layer.call(inputs[self.scope_key], **kwargs)})
        else:
            return self.layer.call(inputs, **kwargs)

    def _call_sparse(self, inputs, **kwargs):
        return self._call(inputs, **kwargs)

    def _compute_output_shape(self, input_shape):
        if self._module_input:
            self._check_scope_key(input_shape)
            input_shape[self.scope_key] = self.layer.compute_output_shape(input_shape[self.scope_key])
            return input_shape
        else:
            return self.layer.compute_output_shape(input_shape)

    def _compute_output_shape_sparse(self, input_shape):
        return self._compute_output_shape(input_shape)

    def _compute_mask(self, inputs, mask=None):
        if self._module_input:
            self._check_scope_key(inputs)
            mask[self.scope_key] = self.layer.compute_mask(inputs[self.scope_key], mask)
            return mask
        else:
            return self.layer.compute_mask(inputs, mask)

    def _compute_mask_sparse(self, inputs, mask):
        return self._compute_mask(inputs, mask)

    def get_config(self):
        return self.layer.get_config()

    def _check_scope_key(self, inputs):
        if self.scope_key not in inputs.keys():
            raise ValueError('The `scope_key` is not in the dict.')


class Capsule(Module):
    """Capsule

    Be careful if you iterate through the stack. The list could be not unique (distinguish between list and set)!
    """
    # and which modules methods must be overloaded to work properly
    # provide plot function, like summary
    def __init__(self,
                 prototype_distribution=1,
                 sparse_signal=False,
                 **kwargs):
        # store user input for return
        # the Capsule is build by call (all special caps paras are then init)
        self.proto_distribution = prototype_distribution
        self.sparse_signal = sparse_signal
        self._module_stack = []

        # scope keys to handle the application of keras layers to different keys in the dict
        self.scope_keys = []

        # call at the beginning avoid conflicts in the setting of params
        # if it is a real module is specified in build with respect to the setting,; thus definition is made
        # during the runtime!
        super(Capsule, self).__init__(support_sparse_signal=True,
                                      support_full_signal=True,
                                      **self._del_module_args(**kwargs))

        # It's up to the modules to support masking, trainable
        self.supports_masking = True
        self.trainable = True

    def __call__(self, inputs, **kwargs):
        """Modification of kears Layer.__call__

        We let all the checks of the input on the module side with the basic layer __call__
        """
        # input type of a capsule can be changed over calls; there is no need to fix it...let the validity check up to
        # the module
        self._module_input = self._is_module_input(inputs)
        # we don't need this parameter but we have to set it: It's up to the modules if the output is dict().
        self._module_output = self._is_module_input(inputs)
        with K.name_scope(self.name):
            if not self.built:
                if isinstance(inputs, (list, tuple, dict)):
                    input_shape = []
                    for key in range(len(inputs)):
                        input_shape.append(K.int_shape(inputs[key]))
                else:
                    input_shape = K.int_shape(inputs)
                # we build the capsule manually to avoid the error messages of _is_callable(). Super class can't be
                # called
                self.build(input_shape)
            self._is_callable()
            # set capsule paras in kwargs (overwrite other); this is the reason why a capsule can call a capsule!
            kwargs.update(self._get_params())
            return super(Capsule, self).call(inputs, **kwargs)

    def build(self, input_shape):
        # we have to overload this method, otherwise it can't be build manually
        if self.sparse_signal:
            self._build_sparse(input_shape)
        else:
            self._build(input_shape)
        super(Module, self).build(input_shape)

    def _build(self, input_shape):
        """ None: not needed: module init independent to proto distrib
                tuple (x, y): x num protos per capsule, y capsule number
                list: len(list) = num of capsules, list[i] = protos in i
                int: int is num of capsules one proto per class
                return: nd.array
        """
        if not self.built:
            d = self.proto_distribution
            if isinstance(d, tuple) and len(d) == 2:
                if d[0] < 1 or d[1] < 1:
                    raise ValueError('The number of prototypes per capsule and the number of capsules must be greater '
                                     'than 0. You provide: ' + str(d))
                distrib = []
                capsule_extension = []
                for i in range(d[1]):
                    distrib.append(list(range(i * d[0], (i+1) * d[0])))
                    capsule_extension.extend(list(i * np.ones(d[0], dtype=int)))
                proto_extension = list(range(int(d[0] * d[1])))

            elif isinstance(d, list) and len(d) > 0:
                # proto_extension = np.array([], dtype=int)
                proto_extension = []
                capsule_extension = []
                distrib = []
                for i, d_ in enumerate(d):
                    if d_ < 1:
                        raise ValueError('The number of prototypes per capsule must be greater '
                                         'than 0. You provide: ' + str(d_) + ' at index: ' + str(i))
                    distrib.append(list(range(sum(d[0:i]), sum(d[0:(i+1)]))))
                    # proto_extension = np.concatenate((proto_extension,
                    #                                   np.arange(sum(d[0:i]), sum(d[0:i + 1]), dtype=int)))
                    # proto_extension = np.concatenate((proto_extension,
                    #                                   proto_extension[-1] * np.ones(max(d) - d[i], dtype=int)))
                    proto_extension.extend(list(range(sum(d[0:i]), sum(d[0:i + 1]))))
                    proto_extension.extend(list(proto_extension[-1] * np.ones(max(d) - d_, dtype=int)))
                    capsule_extension.extend(list(i * np.ones(d_, dtype=int)))

            elif isinstance(d, int):
                if d < 1:
                    raise ValueError('The number of capsules must be greater than 0. You provide:' + str(d))
                distrib = []
                for i in range(d):
                    distrib.append([i])
                proto_extension = list(range(d))
                capsule_extension = list(range(d))

            else:
                raise TypeError("The argument must be a 2D-tuple, list or int. You pass : '" + str(d) + "'.")

            self._proto_distrib = distrib
            self._proto_extension = proto_extension
            self._capsule_extension = capsule_extension
            self._max_proto_number_in_capsule = 0 if distrib is None else max([len(x) for x in distrib])
            self.capsule_number = 0 if distrib is None else len(distrib)
            self.proto_number = 0 if distrib is None else sum([len(x) for x in distrib])
            self._equally_distributed = False if proto_extension is None else \
                proto_extension == list(range(self.proto_number))

    def _build_sparse(self, input_shape):
        self._build(input_shape)

    def _call(self, inputs, **kwargs):
        """inputs are [vectors, d] or flattened in first capsule"""
        outputs = inputs
        for module in self._module_stack:
            outputs = module(outputs, **kwargs)

        return outputs

    def _call_sparse(self, inputs, **kwargs):
        return self._call(inputs, **kwargs)

    def _compute_output_shape(self, input_shape):
        """input shape is just a tuple keras_shape"""
        # let further checks up to modules
        output_shape = input_shape
        # must call  output inference of stack
        for module in self._module_stack:
            output_shape = module.compute_output_shape(output_shape)
        return output_shape

    def _compute_output_shape_sparse(self, input_shape):
        return self._compute_output_shape(input_shape)

    def _compute_mask(self, inputs, mask=None):
        # which could be an indicator for missing squeeze.
        outputs = inputs
        for module in self._module_stack:
            mask = module.compute_mask(outputs, mask)
            outputs = module(outputs, **self._get_params())
        return outputs

    def _compute_mask_sparse(self, inputs, mask):
        return self._compute_mask(inputs, mask)

    # overload tis method so that is up to the module to pre-process the signal
    def _to_module_output(self, inputs, module_output=False):
        return inputs

    # overload tis method so that it is up to the module to pre-process the signal
    def _to_layer_input(self, inputs):
        return inputs

    # overload tis method so that it is up to the module to pre-process the signal
    def _to_module_input(self, inputs):
        return inputs

    @staticmethod
    def _getattr(module, attr=None):
        # return nested layer or module if None
        if attr is None:
            if isinstance(module, _ModuleWrapper):
                return module.layer
            else:
                return module
        else:
            if isinstance(module, _ModuleWrapper):
                return getattr(module.layer, attr)
            else:
                return getattr(module, attr)

    @staticmethod
    def _setattr(module, attr, value):
        if isinstance(module, _ModuleWrapper):
            setattr(module.layer, attr, value)
        else:
            setattr(module, attr, value)

    @staticmethod
    def _hasattr(module, attr):
        if isinstance(module, _ModuleWrapper):
            return hasattr(module.layer, attr)
        else:
            return hasattr(module, attr)

    def _get_modules(self):
        stack = []
        for module in self._module_stack:
            if stack.count(module) == 0:
                stack.append(module)
        return stack

    # Todo: test scope_keys
    def add(self, modules, scope_keys=0):
        # we don't support adding (in general any change at the module_stack) after the module was built. Possibly, it
        # leads to ambiguous capsules in a graph. You can reuse the modules if a second use of modules is needed.
        if not self.built:
            # key: to select over which key of the dict should a keras layer applied. It's ignored for modules and if the
            # signal is not a dict; if int --> repeated to length of modules, if list must match
            if not isinstance(modules, (list, tuple)):
                modules = [modules]
            if not isinstance(scope_keys, (list, tuple)):
                scope_keys = list(np.ones((len(modules),), dtype=int) * scope_keys)

            if len(scope_keys) != len(modules):
                raise ValueError('The number of `scope_keys` must be equal the number of `modules`. You provide: '
                                 'len(scope_keys)=' + str(len(scope_keys)) + " != len(modules)=" + str(len(modules)))

            for i, module in enumerate(modules):
                if not isinstance(module, (Layer, Module)):
                    raise TypeError('The added module must be an instance of class Layer or Module. Found: ' + str(module)
                                    + ". Maybe you forgot to initialize the module.")
                # check if is a module by checking the existence of some attributes
                if hasattr(module, '_module_input') and hasattr(module, '_support_sparse_signal'):
                    self._module_stack.append(module)
                    self.scope_keys.append(None)
                # each type which is not a module is considered as layer type
                else:
                    idx = None
                    for m in self._get_modules():
                        # check if layer was added as module before to avoid a new wrapping
                        if module == self._getattr(m):
                            idx = self._module_stack.index(m)
                            break
                    if idx is None:
                        self._module_stack.append(_ModuleWrapper(module, scope_keys[i]))
                    else:
                        if self._module_stack[idx].scope_key != scope_keys[i]:
                            raise ValueError('The keras layer ' + self._getattr(self._module_stack[idx]).name +
                                             'was wrapped before by the capsule ' + self.name + ' and is already in the '
                                             'stack. Now you call the layer again with a different `scope_key`, which'
                                             ' is not possible. Old scope_key=' + str(self._module_stack[idx].scope_key) +
                                             ' unequal new scope_key=' + str(scope_keys[i]) + '.')
                        self._module_stack.append(self._module_stack[idx])

                    self.scope_keys.append(scope_keys[i])

            return self
        else:
            raise AttributeError('You cannot add layers or modules to a capsule after it was built. If you want to '
                                 'reuse modules in a another capsule, implement a new capsule and add all the needed '
                                 'layers/modules manually.')

    @property
    def module_stack(self):
        stack = []
        for module in self._module_stack:
            stack.append(self._getattr(module))
        return stack

    @property
    def modules(self):
        """Get all used modules (not the stack)
        """
        stack = []
        for module in self._get_modules():
            stack.append(self._getattr(module))
        return stack

    def get_module(self, name=None, index=None):
        """Retrieves a module based on either its name (unique) or index.

        Slightly modified copy from Keras.

        Indices are based on the order of adding.

        # Arguments
            name: String, name of module.
            index: Integer, index of module.

        # Returns
            A module instance.

        # Raises
            ValueError: In case of invalid module name or index.
        """
        # It would be unreliable to build a dictionary
        # based on layer names, because names can potentially
        # be changed at any point by the user
        # without the container being notified of it.
        if index is not None:
            if len(self._module_stack) <= index:
                raise ValueError('Was asked to retrieve module at index ' + str(index) + ' but capsule only has ' +
                                 str(len(self._module_stack)) + ' modules.')
            else:
                return self._getattr(self._module_stack[index])
        else:
            if not name:
                raise ValueError('Provide either a module name or module index.')

        for module in self._module_stack:
            if self._getattr(module, 'name') == name:
                return self._getattr(module)

        raise ValueError('No such module: ' + name)

    @property
    def proto_distrib(self):
        return self._proto_distrib

    @property
    def trainable_weights(self):
        """Trainable weights over _get_modules() and not the stack."""
        weights = []
        for module in self._get_modules():
            weights += self._getattr(module, 'trainable_weights')
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        for module in self._get_modules():
            weights += self._getattr(module, 'non_trainable_weights')
        return weights

    def get_weights(self):
        """Retrieves the weights of the capsule.

        # Returns
            A flat list of Numpy arrays.
        """
        weights = []
        for module in self._get_modules():
            weights += self._getattr(module, 'weights')
        return K.batch_get_value(weights)

    def set_weights(self, weights):
        """Sets the weights of the capsule.

        # Arguments
            weights: A list of Numpy arrays with shapes and types matching
                the output of `capsule.get_weights()`.
        """
        tuples = []
        for module in self._get_modules():
            num_param = len(self._getattr(module, 'weights'))
            layer_weights = weights[:num_param]
            for sw, w in zip(self._getattr(module, 'weights'), layer_weights):
                tuples.append((sw, w))
            weights = weights[num_param:]
        K.batch_set_value(tuples)

    def get_input_shape_at(self, node_index):
        return self._getattr(self._module_stack[0], 'get_input_shape_at')(node_index)

    def get_output_shape_at(self, node_index):
        return self._getattr(self._module_stack[-1], 'get_output_shape_at')(node_index)

    def get_input_at(self, node_index):
        return self._getattr(self._module_stack[0], 'get_input_at')(node_index)

    def get_output_at(self, node_index):
        return self._getattr(self._module_stack[-1], 'get_output_at')(node_index)

    def get_input_mask_at(self, node_index):
        return self._getattr(self._module_stack[0], 'get_input_mask_at')(node_index)

    def get_output_mask_at(self, node_index):
        return self._getattr(self._module_stack[-1], 'get_output_mask_at')(node_index)

    @property
    def input(self):
        return self._getattr(self._module_stack[0], 'input')

    @property
    def output(self):
        return self._getattr(self._module_stack[-1], 'output')

    @property
    def input_mask(self):
        return self._getattr(self._module_stack[0], 'input_mask')

    @property
    def output_mask(self):
        return self._getattr(self._module_stack[-1], 'output_mask')

    @property
    def input_shape(self):
        return self._getattr(self._module_stack[0], 'input_shape')

    @property
    def output_shape(self):
        return self._getattr(self._module_stack[-1], 'output_shape')

    def get_config(self):
        config = {'prototype_distribution': self.proto_distribution}
        super_config = super(Capsule, self).get_config()
        config = dict(list(super_config.items()) + list(config.items()))

        stack_config = []
        for module in self._module_stack:
            stack_config.append(self._getattr(module, 'name'))
        config.update({'module_stack': stack_config,
                       'scope_keys': self.scope_keys})

        modules_config = {}
        for module in self._get_modules():
            module_config = {'class_name': (self._getattr(module, '__class__')).__name__,
                             'config': (self._getattr(module, 'get_config'))()}
            modules_config.update({self._getattr(module, 'name'): module_config})
        config.update({'modules': modules_config})

        return config

    @classmethod
    def from_config(cls, config):
        modules_config = config.pop('modules')

        def process_modules():
            modules = {}
            # import here; at the beginning leads to errors
            from .modules import globals_modules as globs

            for module_name in modules_config:
                module = deserialize_layer(modules_config[module_name],
                                           custom_objects=globs())
                modules.update({module_name: module})

            return modules

        modules = process_modules()

        stack_config = config.pop('module_stack')
        module_stack = []
        for name in stack_config:
            module_stack.append(modules[name])

        scope_keys = config.pop('scope_keys')

        capsule = cls(**config)
        capsule.add(module_stack, scope_keys)
        return capsule

    def summary(self, line_length=None, positions=None, print_fn=print):
        """Prints a string summary of the network.

        Copy from Keras with slight modifications.

        # Arguments
            line_length: Total length of printed lines
                (e.g. set this to adapt the display to different
                terminal window sizes).
            positions: Relative or absolute positions of log elements
                in each line. If not provided,
                defaults to `[.53, .73, .85, 1.]`.
            print_fn: Print function to use.
                It will be called on each line of the summary.
                You can set it to a custom function
                in order to capture the string summary.
        """
        def print_row(fields, positions_):
            line = ''
            for j in range(len(fields)):
                if j > 0:
                    line = line[:-1] + ' '
                line += str(fields[j])
                line = line[:positions_[j]]
                line += ' ' * (positions_[j] - len(line))
            print_fn(line)

        def print_module_summary(module):
            """Prints a summary for a single module.

            # Arguments
                module: target module.
            """
            called_at = []
            idx = -1
            for _ in range(self._module_stack.count(module)):
                idx = self._module_stack.index(module, idx+1)
                called_at.append(idx)
            fields = [self._getattr(module, 'name') + ' (' + (self._getattr(module, '__class__')).__name__ + ')',
                      module.owner,
                      str((self._getattr(module, 'count_params'))()),
                      str(called_at)]
            print_row(fields, positions)

        self._is_callable()
        line_length = line_length or 98
        positions = positions or [.53, .73, .85, 1.]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ['Module (type)', 'Owner', 'Param #', 'Called at pos.']

        print_fn('\n' + '_' * line_length)
        print_fn('Capsule name: {}'.format(self.name))
        print_fn('Number of prototypes: {:,}'.format(self.proto_number))
        print_fn('Number of capsules: {:,}'.format(self.capsule_number))

        print_fn('=' * line_length)
        print_row(to_display, positions)
        print_fn('=' * line_length)

        modules = self._get_modules()
        for i in range(len(modules)):
            print_module_summary(modules[i])
            if i == len(modules) - 1:
                print_fn('=' * line_length)
            else:
                print_fn('_' * line_length)

        trainable_count = self.count_params()
        non_trainable_count = int(
            np.sum([K.count_params(p) for p in set(self.non_trainable_weights)]))

        print_fn('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print_fn('Trainable params: {:,}'.format(trainable_count))
        print_fn('Non-trainable params: {:,}'.format(non_trainable_count))
        print_fn('_' * line_length + '\n')

    def count_params(self):
        """Count the total number of scalars composing the weights.

        # Returns
            An integer count.
        """
        return int(np.sum([K.count_params(p) for p in set(self.weights)]))


# Todo: test self.input_to_dict
class InputModule(Module):
    """Should handle init of second d input (two-dimensional), reshape to vector size if needed

    return: list of tensor [inputs, d]

    signal_shape: if list or tuple (manually specified shape):
        e.g.: (-1, dim) --> reshape over all: Capsule dimension 1 --> vectors
              (channels, dim1, -1) --> reshape just over channels: capsule dimension 2: --> image
        if int: short cut for (-1, dim)
        if None: bypassing of inputs...no reshape

    if signal is not module:
        - generating of diss and

    Here we have to tile the signal. There is no way around to avoid the storage expensive operation, because we need
    need access to all pre-processed vectors at the routing. In classical layer we can avoid this by formulating the
    operation as one matrix operation.

    The shape of signals is: (batch, num_capsules, channels, dim1, ..., dimN) --> tile to num_capsules

    It's not necessary to tile to proto_number. proto_number is always greater or equal caps_number. Thus if you wanna
    have a prototype wise processing define in a first capsule caps_number as proto_number and make your prototype
    based processing. Then, define a second capsule with proto_number and caps_number and make the remaining processing.

    If you wanna have no tiling, because you wanna make a general processing of the input stream. Define a capsule
    layer with just one capsule and make your processing. Continue with a second capsule and etc.

    shapes of inputs:
        - if module signal is False:
            - inputs = [batch, dim1', ..., dimN']
        - if module signal is True:
            - inputs = [signals, diss]
            - signals = [batch, capsule_number_of_previous, dim1, ..., dimN]
            - diss = [batch, capsule_number_of_previous]
    shapes of the outputs:
        - outputs = [signals, diss]
        - if sparse is False:
            - signals = [batch, capsule_number, channels, dim1, ..., dimN]
          else:
            - signals = [batch, 1, channels, dim1, ..., dimN]
        - diss = [batch, proto_number, channels]

    If module_input is True:
        - make the tiling of diss and signals
        - make a reshape if signal_shape is not None
        - check that diss fits proto_number
        - overwrite old diss if init_diss_initializer is not None or if a dissimilarity_tensor is given
          (check shape)
    is False:
        - make reshape
        - tiling
        - init of diss or routing of tensor
    """
    def __init__(self,
                 signal_shape=None,
                 input_to_dict=False,
                 init_diss_initializer=None,
                 init_diss_regularizer=None,
                 init_diss_constraint='NonNeg',
                 signal_regularizer=None,
                 diss_regularizer=None,
                 **kwargs):

        if signal_shape is not None:
            if isinstance(signal_shape, (list, tuple)) and len(signal_shape) >= 2:
                self._signal_shape = list(signal_shape)
            elif isinstance(signal_shape, (list, tuple)) and len(signal_shape) == 1 and \
                    isinstance(signal_shape[0], int):
                self._signal_shape = signal_shape[0]
            elif isinstance(signal_shape, int):
                self._signal_shape = signal_shape
            else:
                raise ValueError("'signal_shape' must be list or tuple with len()>=1, int or None.")
        else:
            self._signal_shape = signal_shape

        # read input from list and convert it to dict
        if isinstance(input_to_dict, bool):
            self.input_to_dict = input_to_dict
        else:
            raise TypeError("input_to_dict must be bool. You provide: " + str(input_to_dict))

        self.init_diss_initializer = initializers.get(init_diss_initializer) \
            if init_diss_initializer is not None else None
        self.init_diss_regularizer = regularizers.get(init_diss_regularizer)
        self.init_diss_constraint = constraints.get(init_diss_constraint)

        self.dissimilarity = None
        # without batch dim
        self.signal_shape = None

        self.output_regularizers = [regularizers.get(signal_regularizer),
                                    regularizers.get(diss_regularizer)]

        super(InputModule, self).__init__(module_output=True,
                                          support_sparse_signal=True,
                                          support_full_signal=True,
                                          **self._del_module_args(**kwargs))

    def _build(self, input_shape):
        if not self.built:
            # check that the input_shape has the correct shape for automatic conversion
            if self.input_to_dict:
                if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
                    raise TypeError("If you provide a module input as list or tuple the length must be two. Otherwise"
                                    "we can't convert the signal automatically to a module input. You provide: "
                                    + str(input_shape))

            if self._module_input or self.input_to_dict:
                signal_shape = shape_inference(input_shape[0][1:], self._signal_shape)
            else:
                signal_shape = shape_inference(input_shape[1:], self._signal_shape)

            # init of dissimilarity
            if self.init_diss_initializer is not None:
                self.dissimilarity = self.add_weight(shape=(signal_shape[0],),
                                                     initializer=self.init_diss_initializer,
                                                     name='dissimilarity',
                                                     regularizer=self.init_diss_regularizer,
                                                     constraint=self.init_diss_constraint)
            # routing of dissimilarity
            else:
                # routing not possible
                if not self._module_input and not self.input_to_dict:
                    raise TypeError("If the input is not a module input you have to provide a valid dissimilarity "
                                    "initializer like 'zeros' and not None.")
                # check if diss has the correct channel dimension
                else:
                    if signal_shape[0] != input_shape[1][1]:
                        raise ValueError("The number of channels after reshape of signals must be equal to the number "
                                         "of channels for diss. You provide: diss[1]=" + str(input_shape[1][1]) +
                                         " and signals[1]=" + str(signal_shape[0]))

            # set signal_shape if all tests are passed
            self.signal_shape = signal_shape

            # Set input spec.
            if self._module_input or self.input_to_dict:
                self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                                   InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]
            else:
                self.input_spec = InputSpec(shape=(None,) + tuple(input_shape[1:]))

    def _build_sparse(self, input_shape):
        self._build(input_shape)

    def _call(self, inputs, **kwargs):
        if self._module_input or self.input_to_dict:
            signals = inputs[0]
            diss = inputs[1]

        else:
            signals = inputs
            diss = None

        batch_size = None

        with K.name_scope('get_signals'):
            # just call reshape if needed
            if self._signal_shape is not None:
                batch_size = K.shape(signals)[0] if batch_size is None else batch_size
                signals = K.reshape(signals, (batch_size,) + self.signal_shape)

            signals = K.expand_dims(signals, axis=1)

            if not self.sparse_signal:
                signals = K.tile(signals, [1, self.capsule_number] + list(np.ones((len(self.signal_shape)), dtype=int)))

        with K.name_scope('get_dissimilarities'):
            if self.dissimilarity is not None:
                batch_size = K.shape(signals)[0] if batch_size is None else batch_size
                diss = K.expand_dims(K.expand_dims(self.dissimilarity, 0), 0)
                diss = K.tile(diss, [batch_size, self.proto_number, 1])
            else:
                diss = K.expand_dims(diss, 1)
                diss = K.tile(diss, [1, self.proto_number, 1])

        return {0: signals, 1: diss}

    def _call_sparse(self, inputs, **kwargs):
        return self._call(inputs, **kwargs)

    def _compute_output_shape(self, input_shape):
        """Coming soon
        """
        if input_shape is None or not isinstance(input_shape, (list, tuple)) or not len(input_shape) >= 2:
            raise ValueError("'input_shape' must be list or tuple with len()>=2.")

        if not self.built:
            if self._module_input or self.input_to_dict:
                batch_size = input_shape[0][0]
                signal_shape = input_shape[0][2:]
            else:
                batch_size = input_shape[0]
                signal_shape = input_shape[1:]

            signal_shape = shape_inference(signal_shape, self._signal_shape)

            return [(batch_size, self.capsule_number) + signal_shape,
                    (batch_size, self.proto_number, signal_shape[0])]

        else:
            if self._module_input or self.input_to_dict:
                if tuple(self.input_spec[0].shape[1:]) != tuple(input_shape[0][1:]):
                    raise ValueError('Input is incompatible with module ' + self.name + ': expected signal shape='
                                     + str(tuple(self.input_spec[0].shape[1:])) + ', found signal shape='
                                     + str(tuple(input_shape[0][1:])))
                if tuple(self.input_spec[1].shape[1:]) != tuple(input_shape[1][1:]):
                    raise ValueError('Input is incompatible with module ' + self.name + ': expected diss shape='
                                     + str(tuple(self.input_spec[1].shape[1:])) + ', found diss shape='
                                     + str(tuple(input_shape[1][1:])))
                batch_size = input_shape[0][0]
            else:
                if tuple(self.input_spec.shape[1:]) != tuple(input_shape[1:]):
                    raise ValueError('Input is incompatible with module ' + self.name + ': expected input shape='
                                     + str(tuple(self.input_spec.shape[1:])) + ', found input shape='
                                     + str(tuple(input_shape[1:])))
                batch_size = input_shape[0]

            return [(batch_size, self.capsule_number) + self.signal_shape,
                    (batch_size, + self.proto_number, self.signal_shape[0])]

    def _compute_output_shape_sparse(self, input_shape):
        full_output_shape = self._compute_output_shape(input_shape)
        signal_shape = list(full_output_shape[0])
        signal_shape[1] = 1

        return [tuple(signal_shape), full_output_shape[1]]

    def get_config(self):
        config = {'signal_shape': self._signal_shape,
                  'input_to_dict': self.input_to_dict,
                  'init_diss_initializer': initializers.serialize(self.init_diss_initializer),
                  'init_diss_regularizer': regularizers.serialize(self.init_diss_regularizer),
                  'init_diss_constraint': constraints.serialize(self.init_diss_constraint),
                  'signal_regularizer': regularizers.serialize(self.output_regularizers[0]),
                  'diss_regularizer': regularizers.serialize(self.output_regularizers[1])}
        super_config = super(InputModule, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


# Todo: OutputModule test
class OutputModule(Module):
    """push the number of capsules as channels and set capsule_number to one

    shapes of the outputs:
        - outputs = [signals, diss]
        - signals = [batch, capsule_number, channels, dim1, ..., dimN]
        - diss = [batch, proto_number, channels]
    """
    def __init__(self,
                 squeeze_capsule_dim=False,
                 output_to_list=False,
                 **kwargs):

        # needed if you just want to use the capsule as a vector processing unit. Initialize a capsule layer with one
        # capsule and squeeze the dimension at the end.
        if isinstance(squeeze_capsule_dim, bool):
            self.squeeze_capsule_dim = squeeze_capsule_dim
        else:
            raise TypeError("squeeze_capsule_dim must be bool. You provide: " + str(squeeze_capsule_dim))

        # convert the output from dict to list if needed
        if isinstance(output_to_list, bool):
            self.output_to_list = output_to_list
        else:
            raise TypeError("output_to_list must be bool. You provide: " + str(output_to_list))

        super(OutputModule, self).__init__(module_input=True,
                                           module_output=not self.output_to_list,
                                           support_sparse_signal=True,
                                           support_full_signal=True,
                                           **self._del_module_args(**kwargs))

    def _build(self, input_shape):
        if not self.built:
            if input_shape[0][1] != input_shape[1][1]:
                raise ValueError("The number of capsules must be equal to the number of prototypes. You provide "
                                 + str(input_shape[0][1]) + "!=" + str(input_shape[1][1]))

            if self.squeeze_capsule_dim:
                if input_shape[0][1] != 1:
                    raise ValueError("To squeeze the capsule dimension, the dimension must be one. You provide: "
                                     + str(input_shape))

            self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                               InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _build_sparse(self, input_shape):
        # manipulate input_shape to call the full build method instead of a new implementation
        signal_shape = list(input_shape[0])
        signal_shape[1] = input_shape[1][1]

        self._build([tuple(signal_shape), input_shape[1]])

        self.input_spec = [InputSpec(shape=(None,) + tuple(input_shape[0][1:])),
                           InputSpec(shape=(None,) + tuple(input_shape[1][1:]))]

    def _call(self, inputs, **kwargs):
        if self.squeeze_capsule_dim:
            return {0: K.squeeze(inputs[0], 1), 1: K.squeeze(inputs[1], 1)}
        else:
            return inputs

    def _call_sparse(self, inputs, **kwargs):
        return self._call(inputs, **kwargs)

    def _compute_output_shape(self, input_shape):
        if self.squeeze_capsule_dim:
            signals = list(input_shape[0])
            diss = list(input_shape[1])

            del signals[1]
            del diss[1]
            return [tuple(signals), tuple(diss)]
        else:
            return input_shape

    def _compute_output_shape_sparse(self, input_shape):
        return self._compute_output_shape(input_shape)

    def get_config(self):
        config = {'squeeze_capsule_dim': self.squeeze_capsule_dim,
                  'output_to_list': self.output_to_list}
        super_config = super(OutputModule, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


# Todo: SplitModule test
class SplitModule(Module):
    """No check for empty tensors after the split
    """
    def __init__(self,
                 axis=-1,
                 index=-1,
                 signal_regularizer=None,
                 diss_regularizer=None,
                 **kwargs):
        self.axis = axis
        self.index = index

        self.permute = None

        self.output_regularizers = [regularizers.get(signal_regularizer),
                                    regularizers.get(diss_regularizer)]

        super(SplitModule, self).__init__(module_input=False,
                                          module_output=True,
                                          support_sparse_signal=True,
                                          support_full_signal=True,
                                          **self._del_module_args(**kwargs))

    def _build(self, input_shape):
        if not self.built:
            permute = list(range(len(input_shape)))
            tmp = permute[self.axis]
            permute[self.axis] = permute[0]
            permute[0] = tmp

            self.permute = permute

            self.input_spec = InputSpec(shape=(None,) + tuple(input_shape[1:]))

    def _build_sparse(self, input_shape):
        self._build(input_shape)

    def _call(self, inputs, **kwargs):
        x = K.permute_dimensions(inputs, self.permute)

        return {0: K.permute_dimensions(x[:self.index], self.permute),
                1: K.permute_dimensions(x[self.index:], self.permute)}

    def _call_sparse(self, inputs, **kwargs):
        return self._call(inputs, **kwargs)

    def _compute_output_shape(self, input_shape):
        # use numpy to make a shape inference
        if input_shape[self.axis] is None:
            return [input_shape, input_shape]
        else:
            output_shape = list(input_shape)

            dummy = list(range(input_shape[self.axis]))

            output_shape0 = output_shape.copy()
            output_shape0[self.axis] = len(dummy[:self.index])

            output_shape1 = output_shape.copy()
            output_shape1[self.axis] = len(dummy[self.index:])

            return [tuple(output_shape0), tuple(output_shape1)]

    def _compute_output_shape_sparse(self, input_shape):
        return self._compute_output_shape(input_shape)

    def get_config(self):
        config = {'axis': self.axis,
                  'index': self.index,
                  'signal_regularizer': regularizers.serialize(self.output_regularizers[0]),
                  'diss_regularizer': regularizers.serialize(self.output_regularizers[1])}
        super_config = super(SplitModule, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


def shape_inference(input_shape, target_shape):
    """coming soon

    return: signal shape without batch dimension
    """
    if target_shape is not None:
        if isinstance(target_shape, int):
            target_shape = [-1, target_shape]
        else:
            target_shape = target_shape

        if None in input_shape:
            raise ValueError("You have to fully define the input_shape without `None`. You provide: "
                             + str(input_shape))

        if target_shape.count(0) == 0 and np.all(np.array(target_shape) >= -1):
            if target_shape.count(-1) == 1:
                # find shape inference index
                idx = target_shape.index(-1)
                target_shape[idx] = 1

                # shape inference possible?
                if np.prod(input_shape) % np.prod(target_shape) != 0:
                    target_shape[idx] = -1
                    raise ValueError('Cannot reshape tensor of shape ' + str(tuple(input_shape)) +
                                     ' into shape ' + str(tuple(target_shape)) + '.')

                # compute missing dimension
                else:
                    dim = np.prod(input_shape) // np.prod(target_shape)
                    target_shape[idx] = dim

            elif target_shape.count(-1) > 1:
                raise ValueError('Can only infer one unknown dimension. You provide ' + str(tuple(target_shape)))

        else:
            raise ValueError('Cannot reshape to the specified shape: ' + str(tuple(target_shape)))

    else:
        target_shape = list(input_shape)

    # final shape check
    if np.prod(target_shape) != np.prod(input_shape):
        raise ValueError('Cannot reshape a tensor of shape ' + str(tuple(input_shape)) + ' into shape '
                         + str(tuple(target_shape)) + '.')

    return tuple(target_shape)
