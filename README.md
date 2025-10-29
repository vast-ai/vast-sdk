# Vast.ai Python SDK
The official Vast.ai SDK pip package.
[![PyPI version](https://badge.fury.io/py/vastai-sdk.svg)](https://badge.fury.io/py/vastai-sdk)

## Install
```bash
pip install vastai-sdk
```
## Example

```python
import vastai_sdk
vastai = vastai_sdk.VastAI()
```


```python
$ pip install vastai-sdk
$ python
Python 3.11.2 (main, Aug 26 2024, 07:20:54) [GCC 12.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import vastai_sdk
>>> import os
>>> VAST_API_KEY = "my-api-key
>>> v = vastai_sdk.VastAI(VAST_API_KEY) # or just set VAST_API_KEY environment variable
>>> v.search_offers()
```


## Calling Help
```python
>>> help(v.create_instance)
```
