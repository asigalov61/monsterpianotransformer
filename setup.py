from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="monsterpianotransformer",
    version="25.1.86",
    description="Ultra-fast and very well fitted solo Piano music transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alex Lev",
    author_email="alexlev61@proton.me",
    url="https://github.com/asigalov61/monsterpianotransformer",
    project_urls={
        "SoundCloud": "https://soundcloud.com/aleksandr-sigalov-61/sets/monster-piano-transformer",
        "Output Samples": "https://github.com/asigalov61/monsterpianotransformer/tree/main/monsterpianotransformer/output_samples",
        "Examples": "https://github.com/asigalov61/monsterpianotransformer/tree/main/monsterpianotransformer/examples",
        "Issues": "https://github.com/asigalov61/monsterpianotransformer/issues",
        "Documentation": "https://github.com/asigalov61/monsterpianotransformer",
        "Discussions": "https://github.com/asigalov61/monsterpianotransformer/discussions",
        "Source Code": "https://github.com/asigalov61/monsterpianotransformer",
        "Official GitHub Repo": "https://github.com/asigalov61/monsterpianotransformer",
        "Hugging Face Models Repo": "https://huggingface.co/asigalov61/Monster-Piano-Transformer",
        "Hugging Face Spaces Demo": "https://huggingface.co/spaces/asigalov61/Monster-Piano-Transformer"
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'monsterpianotransformer': ['/', 'seed_midis/', 'examples/', 'artwork/', 'gradio/', 'training_code/', 'output_samples/'],
    },
    keywords=['MIDI', 'music', 'music ai', 'music transformer', 'piano transformer'],
    python_requires='>=3.6',
    license='Apache Software License 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',  
        'Operating System :: OS Independent',        
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ],
)