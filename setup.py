from setuptools import setup

setup(name='bayes_sim_ig',
      version='0.1.preview1',
      description='BayesSimIG: Scalable Parameter Inference for Adaptive Domain Randomization with Isaac Gym.',
      author='NVIDIA CORPORATION',
      author_email='',
      url='http://developer.nvidia.com/TODO',
      license='Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.',
      packages=['bayes_sim_ig'],
      python_requires='>=3.6',
      install_requires=['numpy', 'gym', 'sklearn', 'ghalton',
                        'pyyaml', 'pynvml', 'matplotlib', 'moviepy',
                        # specifying 1.8.0 for pytorch to make IG work, e.g.
                        # for Anymal task using from torch.tensor import Tensor
                        # which is not available in torch 1.9.0
                        # TODO: Ask IG team to fix Anymal task import
                        'torch==1.8.0',
                        'tensorboard', 'tensorboardX'
                        ],
)
