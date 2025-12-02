from setuptools import setup
import setuptools

setup(name='wbfm',
      version='1.4.0',
      description='Whole Brain Freely Moving neuron tracking and analysis',
      url='https://github.com/Zimmer-lab/wbfm',
      author='Charles Fieseler',
      author_email='charles.fieseler@gmail.com',
      license='MIT',
      packages=setuptools.find_namespace_packages(),
      zip_safe=False,
    include_package_data=True,
      package_data={
            "wbfm": ["**/*.yaml"],
      },
      )
