from setuptools import setup, find_packages

setup(
    name="vastai-sdk",
    version="0.1.19",
    description="SDK for Vast.ai GPU Cloud Service",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Chris McKenzie, Lucas Armand, Zuby Javed",
    author_email="chris@vast.ai, lucas@vast.ai, zuby@vast.ai",
    url="https://github.com/vast-ai/vast-sdk",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25",
        "jsonschema>=3.2",
        "xdg>=1.0.0",
        "borb~=2.0.17",
        "python-dateutil",
        "pyparsing",
        "pytz",
        "urllib3",
        "aiohttp",
        "asyncio",
    ],
    license = "MIT AND (Apache-2.0 OR BSD-2-Clause)",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    project_urls={
        "Homepage": "https://vast.ai",
        "Source": "https://github.com/vast-ai/vast-sdk",
    },
)
