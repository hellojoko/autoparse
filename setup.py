from distutils.core import setup

setup(
    name="autoparse",
    packages=["autoparse"],
    version="0.1",
    license="MIT",
    description="Learn parsers from unlabelled data for formatted string mining",
    author="CASTES Charly",
    author_email="ch.castes@hotmail.fr",
    url="https://github.com/hellojoko/autoparse",
    download_url="https://github.com/hellojoko/autoparse/archive/v_01.tar.gz",
    keywords=["AUTOMATON", "LOG", "PARSER", "UNSUPERVISED"],
    install_requires=["networkx", "matplotlib"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
)
