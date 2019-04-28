from setuptools import find_packages, setup

setup(
    name='shortscikit',
    version='0.0.1',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "pywt",
        "gensim",
        "pywavelets",
        "pybind11",
        "cython",
        # "fasttext==0.8.22",
        "fasttext @ git+https://github.com/facebookresearch/fastText.git"
    ],
    tests_require=[
        "nose2"
    ],
    # dependency_links=["git+https: // github.com / facebookresearch / fastText.git#egg=fasttext-0.8.22"],
    url='https://github.com/yatszhash/short-scikit',
    license='MIT',
    author='yatszhash',
    author_email='=yatszhash@users.noreply.github.com',
    description='',
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
