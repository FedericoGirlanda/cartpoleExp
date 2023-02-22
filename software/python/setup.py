from setuptools import setup, find_packages

setup(
    name='quanser_linear_inverted_pendulum',
    author='Max',
    version='1.0.0',
    url="",
    packages=find_packages(),
    install_requires=[
        # general
        'scipy',
        'ipykernel',
        'pyyaml',
        'argparse',
        'lxml',
        'numpy~=1.23.2',
        'matplotlib~=3.1.2',
        'sympy',
        'pandas',

        # optimal control
        #'drake==1.5.0',
        'pydrake',
        'filterpy',
        'cma'
    ],
    classifiers=[
          'Development Status :: 5 - Stable',
          'Environment :: Console',
          'Intended Audience :: Academic Usage',
          'Programming Language :: Python',
          ],
)