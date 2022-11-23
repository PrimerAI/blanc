import json
import os

with open(os.path.join(os.path.dirname(__file__), "version.json")) as reader:
    __version__ = json.load(reader)["version"]
