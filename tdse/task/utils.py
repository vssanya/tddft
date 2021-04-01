import importlib

def load(name):
    module = importlib.import_module(name)
    task = module.Task()
    task.load()
    return task
