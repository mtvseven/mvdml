#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from setuptools import setup

setup(name='mvdml',
      version='0.1.1',
      description='A minimalist framework for me to perform double machine learning',
      url='http://github.com/mtvseven/mvdml',
      author='Mark T. Vandre',
      author_email='mtvseven@hotmail.com',
      license='MIT',
      packages=['mvdml'],
      install_requires=[
          'numpy'
      ],
      zip_safe=False)
