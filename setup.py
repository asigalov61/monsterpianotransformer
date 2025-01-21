from setuptools import setup, find_packages
import os

# Install fluidsynth using apt
# os.system('apt-get update && apt-get install -y fluidsynth')

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="monsterpianotransformer",
    version="25.1.31",
    description="Ultra-fast and very well fitted solo Piano music transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alex Lev",
    author_email="alexlev61@proton.me",
    url="https://github.com/asigalov61/monsterpianotransformer",
    project_urls={
        "Examples": "https://github.com/asigalov61/monsterpianotransformer/example",
        "Issues": "https://github.com/asigalov61/monsterpianotransformer/issues",
        "Documentation": "https://github.com/asigalov61/monsterpianotransformer/docs",
        "Discussions": "https://github.com/asigalov61/monsterpianotransformer/discussions",
        "Source Code": "https://github.com/asigalov61/monsterpianotransformer",
        "Official GitHub Repo": "https://github.com/asigalov61/monsterpianotransformer",
        "Hugging Face Models Repo": "https://huggingface.co/asigalov61/Monster-Piano-Transformer",
        "Hugging Face Spaces Deom": "https://huggingface.co/spaces/asigalov61/Monster-Piano-Transformer"
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'monsterpianotransformer': ['/', 'seed_midis/', 'examples/', 'artwork/'],
    },
    keywords=['MIDI', 'music', 'music ai', 'music transformer'],
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
