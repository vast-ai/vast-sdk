# 🪄 Vast.ai Magic Comes To Python!
It's everything you love from the [Vast.ai CLI](https://github.com/vast-ai/vast-python) tool, wrapped neatly in an easy-to-use Python interface!

## 📦 What's in the pip?
Why, it’s not just an SDK—it’s an entire development philosophy in a single import statement! With just a humble `pip install`, you unlock:

 * ⚡ **Lightning-fast integrations**: So easy, it practically writes your code for you.
 * 🛡️ **Error-free operations**: Bugs? Banished. Exceptions? Extinct. Our SDK makes them a thing of the past!
 * 🌍 **Infinite scalability**: Whether you’re running on a potato or the world’s fastest supercomputer, we’ve got you covered!

## 👀 Let's Sneak A Peek!
Under the hood we are using what the [CLI tool](https://github.com/vast-ai/vast-python) uses and so the documentation is the same. The arguments are the same. 

🐚 shell: `vastai cast --spell='abracadabra'` 

🐍 python: `vastai.cast(spell='abracadabra')`

Just a little something like this and we're ready to roll!
```python
import vastai_sdk
vastai = vastai_sdk.VastAI()
```

In fact, try this right now. I'll wait!

```python
$ pip install vastai-sdk
$ python
Python 3.11.2 (main, Aug 26 2024, 07:20:54) [GCC 12.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import vastai_sdk
>>> v = vastai_sdk.VastAI()
>>> v.search_offers()
````
This is easy, you got this! 

### What about the return values?
JSONable objects, exactly as `--raw` would send to your pretty terminal. It's really the same.

### Alright, but what about an API key, what's the catch?
You can provide it in the class instantiation: `vastai.VastAI("My-magnificent-key")`

OR, if you leave it blank it will look for a key in the same place as the CLI, right there in your friendly `$HOME` directory.

The `creds_source` @property will tell you where what's being used came from. Example:

```python
>>> v=vastai_sdk.VastAI("Not-My-Real-Key-Don't-Worry!")
>>> v.creds_source
'CODE'
>>> v.api_key
"Not-My-Real-Key-Don't-Worry!"
>>>
```

### Introspection, `__doc__`, `__sig__`?
Yes, yes, and yes. It's all in there. Try this at the handy python prompt

```python
>>> help(v.create_instance)
```
Pretty nice, right? Now do this! (No Spoilers!)

```python
>>> help(v.<tab>
```

All the helpers are there so your vscode, emacs, ipython, and neovim sessions will fly as your fingertips tab away.

### Help, support, all that stuff?
Sure. Just head over to GitHub issues.

Thanks for using [Vast.ai](https://vast.ai). We 💖 you!
