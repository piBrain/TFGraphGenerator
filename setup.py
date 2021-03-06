from setuptools import setup, find_packages

setup(
        name='tfgraphgen',
        version='0.6.13',
        description='Symbolic Graph Generation Tensorflow Helper Library',
        url='http://pibrain.io',author='Ian T Butler',
        license='MIT',
        packages=find_packages(),
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: piBrain Engineers',
            'Topic :: Software Development :: Helper Library',
            'License :: MIT License',
            'Programming Language :: Python :: 3.5',
        ],
        keywords='symbolic graph tensorflow',
        install_requires=['tensorflow']
)


