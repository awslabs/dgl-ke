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
import sys
import subprocess
import argparse
import socket
if os.name != 'nt':
    import fcntl
    import struct

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--path', type=str, help='path of distributed script.')
        self.add_argument('--script', type=str, help='distributed script file.')
        self.add_argument('--ip_config', type=str, default='ip_config.txt', 
                          help='path of ip_config file.')
        self.add_argument('--user_name', type=str, help='user name for ssh.')
        self.add_argument('--ssh_key', type=str, help='ssh private key.')


def get_server_count(ip_config):
    """Get total server count from ip_config file
    """
    with open(ip_config) as f:
        _, _, server_count = f.readline().strip().split(' ')
        server_count = int(server_count)

    return server_count


def get_machine_count(ip_config):
    """Get total machine count from ip_config file
    """
    with open(ip_config) as f:
        machine_count = len(f.readlines())

    return machine_count


def local_ip4_addr_list():
    """Return a set of IPv4 address
    """
    nic = set()

    for ix in socket.if_nameindex():
        name = ix[1]
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', name[:15].encode("UTF-8")))[20:24])
        nic.add(ip)

    return nic


def is_local(ip_addr):
    """If ip_addr is a local ip
    """
    if ip_addr in local_ip4_addr_list():
        return True
    else:
        return False


def run_cmd(cmd_str):
    """run command
    """
    os.environ['PATH'] = '/usr/local/bin:/bin:/usr/bin:/sbin/'
    process = subprocess.Popen(cmd_str, shell=True, env=os.environ)
    return process


def wait_job(process, cmd_str):
    """Wait process finish its job
    """
    retcode = process.wait()
    mesg = 'Fail with retcode(%s): %s' %(retcode, cmd_str)
    if retcode != 0:
        raise RuntimeError(mesg)


def ssh_cmd(cmd_str, ip, user_name, ssh_key=None):
    """construct an ssh command
    """
    if ssh_key is None:
        ssh_cmd_str = 'ssh %s@%s \'%s\'' %(user_name, ip, cmd_str)
    else:
        ssh_cmd_str = 'ssh -i %s %s@%s \'%s & exit\'' %(ssh_key, user_name, ip, cmd_str)

    return ssh_cmd_str


def launch(args):
    """launch distributed jobs in cluster
    """
    job_list = []
    cmd_list = []
    server_count = get_server_count(args.ip_config)
    machine_count = get_machine_count(args.ip_config)
    with open(args.ip_config) as f:
        machine_id = 0
        for line in f:
            ip, port, count = line.strip().split(' ')
            server_id_low = machine_id * int(count)
            server_id_high = (machine_id+1) * int(count)
            cmd_str = 'cd %s; rm *-shape; %s %d %d' % (args.path, args.script, server_id_low, server_id_high)
            if is_local(ip) == False: # remote command
                cmd_str = ssh_cmd(cmd_str, ip, args.user_name, args.ssh_key)
            job_list.append(run_cmd(cmd_str))
            cmd_list.append(cmd_str)
            machine_id += 1
    # wait job finish
    for i in range(len(job_list)):
        wait_job(job_list[i], cmd_list[i])


def main():
    args = ArgParser().parse_args()
    launch(args)

if __name__ == '__main__':
    main()
