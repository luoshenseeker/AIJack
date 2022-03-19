from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from ..common.exeptions import LoadError

def formatCheck(dic):
    if dic['update_type'] not in [
        'fedAVG', 'fedSGD'
    ]:
        return False
    if dic['dataset'] not in [
        'MNIST', 'CIFAR'
    ]:
        return False
    if type(dic['para']['epoch']) is not int:
        return False
    return True


def loadConfig(path, check):
    try:
        with open(path, 'r') as f:
            s = f.read()
        config = load(s, Loader)
    except FileNotFoundError:
        raise LoadError("Config file not found")
    except yaml.YAMLError as exc:
        if hasattr(exc, 'problem_mark'):
            mark = exc.problem_mark
            print("Error position:(%s:%s)" % (mark.line+1, mark.column+1))
        raise LoadError("Config file wrong in YAML format, check info above")
    
    # Check
    if config is None:
        raise LoadError("Config file is empty")
    if check and not formatCheck(config):
        raise LoadError("Config file is not in right format")
    return config

if __name__ == '__main__':
    config = loadConfig("C:\\Users\\luoshenseeker\\home\\work\\科研\\new\\AIJack\\config.yaml", True)
    print(config)