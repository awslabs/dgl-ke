import json
import os

class Logger(object):
    def __init__(self, base_path, log_file_name='log.txt', result_file_name='result.json', save_log=True):
        self.log_path = os.path.join(base_path, log_file_name)
        self.result_path = os.path.join(base_path, result_file_name)
        self.save_log = save_log

    def print_log(self, content):
        if self.save_log:
            with open(self.log_path, 'a+') as f:
                f.write(content + '\n')
        print(content)

    def save_result(self, mode, result: dict):
        json_file = {mode: result}
        with open(self.result_path, 'w') as f:
            json.dump(json_file, f, indent=4)





