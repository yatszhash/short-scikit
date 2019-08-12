import sys

from setuptools import find_packages, setup

mecab_package = "mecab-python-windows" if sys.platform == "win32" else "mecab-python3"
optional_requires = ["pyarrow",
                     mecab_package,
                     "janome",
                     # "fasttext==0.8.22",
                     "fastText @ git+https://github.com/facebookresearch/fastText.git",
                     "pywavelets"
                     ]

setup(
    name='shortscikit',
    version='0.0.1',
    packages=find_packages(exclude=['tests', '.circleci']),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "gensim",
        "pybind11",
        "cython"
    ],
    tests_require=[
                      "nose2",
                  ] + optional_requires,
    extras_require={
        'all': optional_requires
    },
    # dependency_links=["git+https: // github.com / facebookresearch / fastText.git#egg=fasttext-0.8.22"],
    url='https://github.com/yatszhash/short-scikit',
    license='MIT',
    author='yatszhash',
    author_email='=yatszhash@users.noreply.github.com',
    description='short-circuit to pre-processing and feature extraction with scikit-learn by reducing to write boiler '
                'codes',
    test_suite='nose2.collector.collector',
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Utilities",
    ]
)
