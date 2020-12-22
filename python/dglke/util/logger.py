import os
import json
class Logger(object):
    log_path = None
    result_path = None

    def __init__(self):
        pass

    @classmethod
    def print(cls, content):
        with open(cls.log_path, 'a+') as f:
            f.write(content + '\n')
        print(content)

    @classmethod
    def save_result(cls, result: dict):
        with open(cls.result_path, 'w') as f:
            json.dump(result, f, indent=4)





