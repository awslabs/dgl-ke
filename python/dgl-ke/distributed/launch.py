# -*- coding: utf-8 -*-
#
# setup.py
#
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import argparse

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--path', type=str, default='~/dgl-ke/python/dgl-ke/distributed',
                          help='launch script path.')
        self.add_argument('--task_file', type=str, default='fb15k_transe_l2.sh',
                          help='name of task file.')
        self.add_argument('--user_name', type=str, default='ubuntu',
                          help='user name for ssh.')
        self.add_argument('--ssh_key', type=str, default='my_key.pem',
                          help='ssh key.')


def get_server_count(ip_config):
    return

def get_machine_count(ip_config):
    return

def launch(args):
    return


if __name__ == '__main__':
    args = ArgParser().parse_args()
    launch(args)