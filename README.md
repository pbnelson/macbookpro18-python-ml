### PLATFORM 

0. MacBook Pro 18,4
0. Monterey v12.3.1
0. Apple M1 Max (arm)
0. CPU Cores: 10 (8 performance and 2 efficiency)
0. GPU Cores: 32 Familiy Apple 7
0. 64GB LPDDR5 (200GB/s memory bandwidth)
0. NPU Cores: 16-core Apple Neural Engine (Neural Processing Units)




### Links

0. Apple's CoreML tool: https://coremltools.readme.io/docs/installation
0. Apple's CoreML quickstart: https://coremltools.readme.io/docs/introductory-quickstart
0. Kera's benchmark tests: https://github.com/keras-team/keras/tree/master/keras/benchmarks/keras_examples_benchmarks
0. Francois Chollet's book site: https://github.com/fchollet/deep-learning-with-python-notebooks
0. Ray Wenderlich, for GPU-accelerated ML on MacOS: https://towardsdatascience.com/gpu-accelerated-machine-learning-on-macos-48d53ef1b545
0. Ray Wenderlich, for scikit-learn intro: https://www.raywenderlich.com/174-beginning-machine-learning-with-scikit-learn
0. Riccardo Di Sipio's OpenCL test: https://towardsdatascience.com/gpu-accelerated-machine-learning-on-macos-48d53ef1b545
0. Tensorflow-metal macos demo: https://makeoptim.com/en/deep-learning/tensorflow-metal
0. Apple Neural Engine post: https://github.com/hollance/neural-engine





#### Monitor CPU & GPU

0. Open `Activity Monitor.app`
0. Press Cmd+3 to view CPU usage
0. Press Cmd+4 to view GPU usage





#### need openblas to run tensorflow operations

````bash
brew install openblas

````




#### be sure to be on Python v3.8, exactly

````bash
brew reinstall python@3.8
brew unlink python
brew link --force python@3.8

# confirm this returns 3.8
python --version

# update to latest pip
python -m pip install -U pip

````



#### create a new `tensorflow-metal` virtual environment, and activate it

See: https://python.land/virtual-environments/virtualenv

````bash
cd ~
python -m venv tensorflow-metal
source ~/tensorflow-metal/bin/activate
# the cmdline prompt should show (tensorflow-metal)$

# note, to deactivate this venv, simply use command: `deactivate`

````



#### need this HDF5 dependency, custom-built, for tensorflow-macos

````bash

# skip these steps if already done/built previously
brew install cython
brew install hdf5
cd ~/repos
git clone https://github.com/h5py/h5py
cd h5py
python -m pip install cython pkgconfig
python setup.py build build_ext --include-dirs=/opt/homebrew/include --library-dirs=/opt/homebrew/lib

# always need to install into venv, however
python -m pip install cython pkgconfig wheel numpy
python -m pip install . --no-build-isolation

````


#### Tool for opencl checking

````bash
brew install clinfo
clinfo | grep Version # returns 1.2, Feb 12 2022

````




#### install apple's Ml packages, and tensorflow for macos

````bash
python -m pip  install coremltools
python -m pip  install tensorflow-macos
python -m pip  install tensorflow-metal

python -c "import tensorflow; print(tensorflow.config.list_physical_devices());"
# should print: [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

````



#### install popular ML packages

````bash
python -m pip install pandas
python -m pip install numpy
python -m pip install scipy
python -m pip install pyopencl
python -m pip install scikit-learn
python -m pip install keras
python -m pip install theano
python -m pip install matplotlib
python -m pip install graphviz
python -m pip install pydot-ng
python -m pip install opencv-python  # computer vision library
python -m pip install torch # aka pytorch, an ML framework


# UNAVAILABLE: python -m pip install plaidml
# UNAVAILABLE: python -m pip install plaidml-keras

````



#### running keras test script


##### install dependencies

````bash
brew install bazel

````


````bash
cd ~/repos
git clone https://github.com/fchollet/keras
cd ~/repos/keras/keras
bazel run keras/benchmarks/keras_examples_benchmarks:cifar10_cnn_benchmark_test

````