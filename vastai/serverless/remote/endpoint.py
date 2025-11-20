import os

def get_mode():
    return mode

class Endpoint:
    def __init__(self,name):
        self.name = name
    # ADD NEW ENDPOINT METHODS HERE, GATE WITH THE RIGHT MODE

mode = os.getenv('VAST_REMOTE_DISPATCH_MODE', 'client')

