### Future Knowledge to Investigate
- Hierarchical Temporal Memory (HTM)
- Cascade Forests
- Capsule Networks (CapsNets),...
- Generative Adversarial Networks
- NeRF
- [Temporal Graph Networks (TGNs)](https://arxiv.org/abs/2006.10637)
- [machine learning technical debt](https://matthewmcateer.me/blog/machine-learning-technical-debt)
- GPT-3

### Tensorflow

TensorFlow 2.1 will be the last TF release supporting Python 2. Python 2 support officially ends an January 1, 2020. As announced earlier, TensorFlow will also stop supporting Python 2 starting January 1, 2020, and no more releases are expected in 2019.
Major Features and Improvements

    The tensorflow pip package now includes GPU support by default (same as tensorflow-gpu) for both Linux and Windows. This runs on machines with and without NVIDIA GPUs. tensorflow-gpu is still available, and CPU-only packages can be downloaded at tensorflow-cpu for users who are concerned about package size.
    Windows users: Officially-released tensorflow Pip packages are now built with Visual Studio 2019 version 16.4 in order to take advantage of the new /d2ReducedOptimizeHugeFunctions compiler flag. To use these new packages, you must install "Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019", available from Microsoft's website here.
        This does not change the minimum required version for building TensorFlow from source on Windows, but builds enabling EIGEN_STRONG_INLINE can take over 48 hours to compile without this flag. Refer to configure.py for more information about EIGEN_STRONG_INLINE and /d2ReducedOptimizeHugeFunctions.
        If either of the required DLLs, msvcp140.dll (old) or msvcp140_1.dll (new), are missing on your machine, import tensorflow will print a warning message.
    The tensorflow pip package is built with CUDA 10.1 and cuDNN 7.6.
    tf.keras
        Experimental support for mixed precision is available on GPUs and Cloud TPUs. See usage guide.
        Introduced the TextVectorization layer, which takes as input raw strings and takes care of text standardization, tokenization, n-gram generation, and vocabulary indexing. See this end-to-end text classification example.
        Keras .compile .fit .evaluate and .predict are allowed to be outside of the DistributionStrategy scope, as long as the model was constructed inside of a scope.
        Experimental support for Keras .compile, .fit, .evaluate, and .predict is available for Cloud TPUs, Cloud TPU, for all types of Keras models (sequential, functional and subclassing models).
        Automatic outside compilation is now enabled for Cloud TPUs. This allows tf.summary to be used more conveniently with Cloud TPUs.
        Dynamic batch sizes with DistributionStrategy and Keras are supported on Cloud TPUs.
        Support for .fit, .evaluate, .predict on TPU using numpy data, in addition to tf.data.Dataset.
        Keras reference implementations for many popular models are available in the TensorFlow Model Garden.
    tf.data
        Changes rebatching for tf.data datasets + DistributionStrategy for better performance. Note that the dataset also behaves slightly differently, in that the rebatched dataset cardinality will always be a multiple of the number of replicas.
        tf.data.Dataset now supports automatic data distribution and sharding in distributed environments, including on TPU pods.
        Distribution policies for tf.data.Dataset can now be tuned with 1. tf.data.experimental.AutoShardPolicy(OFF, AUTO, FILE, DATA) 2. tf.data.experimental.ExternalStatePolicy(WARN, IGNORE, FAIL)
    tf.debugging
        Add tf.debugging.enable_check_numerics() and tf.debugging.disable_check_numerics() to help debugging the root causes of issues involving infinities and NaNs.
    tf.distribute
        Custom training loop support on TPUs and TPU pods is avaiable through strategy.experimental_distribute_dataset, strategy.experimental_distribute_datasets_from_function, strategy.experimental_run_v2, strategy.reduce.
        Support for a global distribution strategy through tf.distribute.experimental_set_strategy(), in addition to strategy.scope().
    TensorRT
        TensorRT 6.0 is now supported and enabled by default. This adds support for more TensorFlow ops including Conv3D, Conv3DBackpropInputV2, AvgPool3D, MaxPool3D, ResizeBilinear, and ResizeNearestNeighbor. In addition, the TensorFlow-TensorRT python conversion API is exported as tf.experimental.tensorrt.Converter.
    Environment variable TF_DETERMINISTIC_OPS has been added. When set to "true" or "1", this environment variable makes tf.nn.bias_add operate deterministically (i.e. reproducibly), but currently only when XLA JIT compilation is not enabled. Setting TF_DETERMINISTIC_OPS to "true" or "1" also makes cuDNN convolution and max-pooling operate deterministically. This makes Keras Conv*D and MaxPool*D layers operate deterministically in both the forward and backward directions when running on a CUDA-enabled GPU.

Breaking Changes

    Deletes Operation.traceback_with_start_lines for which we know of no usages.
    Removed id from tf.Tensor.__repr__() as id is not useful other than internal debugging.
    Some tf.assert_* methods now raise assertions at operation creation time if the input tensors' values are known at that time, not during the session.run(). This only changes behavior when the graph execution would have resulted in an error. When this happens, a noop is returned and the input tensors are marked non-feedable. In other words, if they are used as keys in feed_dict argument to session.run(), an error will be raised. Also, because some assert ops don't make it into the graph, the graph structure changes. A different graph can result in different per-op random seeds when they are not given explicitly (most often).
    The following APIs are not longer experimental: tf.config.list_logical_devices, tf.config.list_physical_devices, tf.config.get_visible_devices, tf.config.set_visible_devices, tf.config.get_logical_device_configuration, tf.config.set_logical_device_configuration.
    tf.config.experimentalVirtualDeviceConfiguration has been renamed to tf.config.LogicalDeviceConfiguration.
    tf.config.experimental_list_devices has been removed, please use
    tf.config.list_logical_devices.

Bug Fixes and Other Changes

    tf.data
        Fixes concurrency issue with tf.data.experimental.parallel_interleave with sloppy=True.
        Add tf.data.experimental.dense_to_ragged_batch().
        Extend tf.data parsing ops to support RaggedTensors.
    tf.distribute
        Fix issue where GRU would crash or give incorrect output when a tf.distribute.Strategy was used.
    tf.estimator
        Added option in tf.estimator.CheckpointSaverHook to not save the GraphDef.
        Moving the checkpoint reader from swig to pybind11.
    tf.keras
        Export depthwise_conv2d in tf.keras.backend.
        In Keras Layers and Models, Variables in trainable_weights, non_trainable_weights, and weights are explicitly deduplicated.
        Keras model.load_weights now accepts skip_mismatch as an argument. This was available in external Keras, and has now been copied over to tf.keras.
        Fix the input shape caching behavior of Keras convolutional layers.
        Model.fit_generator, Model.evaluate_generator, Model.predict_generator, Model.train_on_batch, Model.test_on_batch, and Model.predict_on_batch methods now respect the run_eagerly property, and will correctly run using tf.function by default. Note that Model.fit_generator, Model.evaluate_generator, and Model.predict_generator are deprecated endpoints. They are subsumed by Model.fit, Model.evaluate, and Model.predict which now support generators and Sequences.
    tf.lite
        Legalization for NMS ops in TFLite.
        add narrow_range and axis to quantize_v2 and dequantize ops.
        Added support for FusedBatchNormV3 in converter.
        Add an errno-like field to NNAPI delegate for detecting NNAPI errors for fallback behaviour.
        Refactors NNAPI Delegate to support detailed reason why an operation is not accelerated.
        Converts hardswish subgraphs into atomic ops.
    Other
        Critical stability updates for TPUs, especially in cases where the XLA compiler produces compilation errors.
        TPUs can now be re-initialized multiple times, using tf.tpu.experimental.initialize_tpu_system.
        Add RaggedTensor.merge_dims().
        Added new uniform_row_length row-partitioning tensor to RaggedTensor.
        Add shape arg to RaggedTensor.to_tensor; Improve speed of RaggedTensor.to_tensor.
        tf.io.parse_sequence_example and tf.io.parse_single_sequence_example now support ragged features.
        Fix while_v2 with variables in custom gradient.
        Support taking gradients of V2 tf.cond and tf.while_loop using LookupTable.
        Fix bug where vectorized_map failed on inputs with unknown static shape.
        Add preliminary support for sparse CSR matrices.
        Tensor equality with None now behaves as expected.
        Make calls to tf.function(f)(), tf.function(f).get_concrete_function and tf.function(f).get_initialization_function thread-safe.
        Extend tf.identity to work with CompositeTensors (such as SparseTensor)
        Added more dtypes and zero-sized inputs to Einsum Op and improved its performance
        Enable multi-worker NCCL all-reduce inside functions executing eagerly.
        Added complex128 support to RFFT, RFFT2D, RFFT3D, IRFFT, IRFFT2D, and IRFFT3D.
        Add pfor converter for SelfAdjointEigV2.
        Add tf.math.ndtri and tf.math.erfinv.
        Add tf.config.experimental.enable_mlir_bridge to allow using MLIR compiler bridge in eager model.
        Added support for MatrixSolve on Cloud TPU / XLA.
        Added tf.autodiff.ForwardAccumulator for forward-mode autodiff
        Add LinearOperatorPermutation.
        A few performance optimizations on tf.reduce_logsumexp.
        Added multilabel handling to AUC metric
        Optimization on zeros_like.
        Dimension constructor now requires None or types with an __index__ method.
        Add tf.random.uniform microbenchmark.
        Use _protogen suffix for proto library targets instead of _cc_protogen suffix.
        Moving the checkpoint reader from swig to pybind11.
        tf.device & MirroredStrategy now supports passing in a tf.config.LogicalDevice
        If you're building Tensorflow from source, consider using bazelisk to automatically download and use the correct Bazel version. Bazelisk reads the .bazelversion file at the root of the project directory.
