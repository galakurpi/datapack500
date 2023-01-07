from setuptools import setup
import setuptools

setup(
   include_package_data=True,
   name='datapack',
   version='0.1.0',
   author='Jon Galarraga',
   author_email='jgalarraga004@ikasle.ehu.eus',
   packages=setuptools.find_packages(),
   url='www.google.com',
   license='LICENSE.txt',
   description='A Python package for manipulating datasets',
   long_description=open('README.txt').read(),
   tests_require=['pytest'],
   install_requires=['pytest', 'numpy', 'pandas', 'seaborn', 'matplotlib', 'plotly'],
)