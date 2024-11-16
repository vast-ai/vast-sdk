# 🪄 Vast.ai Magic Comes To Python!
It's everything you love from the [Vast.ai cli](https://github.com/vast-ai/vast-python) tool, wrapped neatly in an easy-to-use Python interface!

## 📦 What's in the pip?
Why, it’s not just an SDK—it’s an entire development philosophy in a single import statement! With just a humble `pip install`, you unlock:

 * ⚡ **Lightning-fast integrations**: So easy, it practically writes your code for you.
 * 🛡️ **Error-free operations**: Bugs? Banished. Exceptions? Extinct. Our SDK makes them a thing of the past!
 * 🌍 **Infinite scalability**: Whether you’re running on a potato or the world’s fastest supercomputer, we’ve got you covered!

## 📚 Documentation, Support, And More!
Under the hood we are using what the [cli tool](https://github.com/vast-ai/vast-python) uses and so the documentation is the same. The arguments are the same. 

🐚 shell: `vastai cast --spell='abracadabra'` 

🐍 python: `vastai.cast(spell='abracadabra')`

### What about the return values?
Well what about them? You get jsonable objects, exactly as `--raw` would send to your pretty terminal. It's really the same.

### Alright, but what about an API key, what's the catch?
You can provide it in the class instantiation: `vastai.VastAI("My-magnificent-key")`

OR, if you leave it blank it will look for a key in the same place as the cli, right there in your friendly `$HOME` directory.

### Introspection, `__doc__`, `__sig__`?
Yes, yes, and yes. It's all in there. 

Your vscode, emacs, ipython, and neovim sessions will fly as your fingertips tab away.

### Help, support, all that stuff?
Sure. Just head over to GitHub issues.

Thanks for using Vast.ai. We 💖 you!
