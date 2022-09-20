# Mac ARM and Python Machine Learning


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
0. Ray Wenderlichh, for Keras and CoreML intro: https://www.raywenderlich.com/188-beginning-machine-learning-with-keras-core-ml
0. Riccardo Di Sipio's OpenCL test: https://towardsdatascience.com/gpu-accelerated-machine-learning-on-macos-48d53ef1b545

0. Tensorflow-metal macos demo: https://makeoptim.com/en/deep-learning/tensorflow-metal
0. Apple Neural Engine post: https://github.com/hollance/neural-engine

0. Youtube classification using naive bayes: https://www.youtube.com/watch?v=60pqgfT5tZM
0. Youtube multi-label text classification: https://www.youtube.com/watch?v=DkCF5xb0840




#### Monitor CPU & GPU

0. Open `Activity Monitor.app`
0. Press Cmd+3 to view CPU usage
0. Press Cmd+4 to view GPU usage





#### need openblas to run tensorflow operations

````bash
brew install openblas
brew info openblas | grep openblas:  # returns 0.3.21

````




#### be sure to be on Python v3.8, exactly

````bash
brew reinstall python@3.8
brew unlink python
brew unlink python@3
brew unlink python@3.8
brew link python@3.8 --force
alias python=python3

# confirm this returns 3.8
python3 --version

# update to latest pip
python3 -m pip install --upgrade pip

````



#### create a new `tensorflow-metal` virtual environment, and activate it

See: https://python.land/virtual-environments/virtualenv

````bash
cd ~
python3 -m venv tensorflow-metal
source ~/tensorflow-metal/bin/activate
# the cmdline prompt should show (tensorflow-metal)$
# note, to deactivate this venv, simply use command: `deactivate`

````


#### update pip, install numpy, etc.

````bash
python -m pip install --upgrade pip

```




#### need this HDF5 dependency, custom-built, for tensorflow-macos

````bash

# skip these steps if already done/built previously
brew install cython  # installs version 0.29.32_1
brew install hdf5 # installs version 1.12.2_2
cd ~/repos
git clone https://github.com/h5py/h5py
cd h5py
python -m pip install cython pkgconfig wheel numpy
python setup.py build build_ext --include-dirs=/opt/homebrew/include --library-dirs=$(brew --prefix hdf5)/lib

# always need to install into venv, however
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


#### install pyopencl

This requires the Python.h file, which is in a deep virtual-env subfolder. 

````bash
export CFLAGS="-I${VIRTUAL_ENV}/lib/python3.8/site-packages/tensorflow/include/external/local_config_python/python_include/"
python -m pip install pyopencl

````






#### install popular ML packages

````bash
python -m pip install pandas
python -m pip install numpy
python -m pip install scipy
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
cd ~/repos/keras
bazel run keras/benchmarks/keras_examples_benchmarks:cifar10_cnn_benchmark_test

````






## Addendum: Coral USB (Edge TPU)

For adding Coral USB support to the above. These instructions presume the Python 3.8 `tensorflow-metal` virtual environment was setup and activated.


### PLATFORM 

0. Coral USB Accelerator (i.e. Edge TPU co-processor)


### Links

0. Getting started: https://coral.ai/docs/accelerator/get-started/#runtime-on-mac
0. Coral software repository: https://coral.ai/software/#coral-python-api
0. Coral sample projects and tutorials: https://coral.ai/examples/
0. Keras and Google Edge TPU's: https://towardsdatascience.com/using-keras-on-googles-edge-tpu-7f53750d952
0. Official Github repo for Coral examples: https://github.com/google-coral/pycoral/tree/master/examples
0. Prebuilt Coral Models: https://coral.ai/models/




#### Activate the virtual environment

The `tensorflow-metal` environment should have been created already, see above.

````bash
cd ~
python3 -m venv tensorflow-metal
source ~/tensorflow-metal/bin/activate
# the cmdline prompt should show (tensorflow-metal)$
# note, to deactivate this venv, simply use command: `deactivate`

````



#### install Mac Edge TPU runtime 

````bash
cd ~/Downloads
curl -LO https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20220308.zip
unzip edgetpu_runtime_20220308.zip
cd edgetpu_runtime
sudo bash install.sh  # answer N to max freq question
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0

````



#### Install the PyCoral github repository full of examples

````bash
cd ~/repos
git clone https://github.com/google-coral/pycoral.git
cd pycoral
bash examples/install_requirements.sh
ls -alF examples

````



#### Run one of the PyCoral examples

This classified the image as a macaw.

````bash
cd ~/repos/pycoral
open test_data/parrot.jpg
python3 examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/parrot.jpg

````

Output of above should be something like this: 
    
    examples/classify_image.py:79: DeprecationWarning: ANTIALIAS is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.LANCZOS instead.
      image = Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS)
    ----INFERENCE TIME----
    Note: The first inference on Edge TPU is slow because it includes loading the model into Edge TPU memory.
    14.4ms
    4.4ms
    4.4ms
    4.4ms
    4.4ms
    -------RESULTS--------
    Ara macao (Scarlet Macaw): 0.75781
    

