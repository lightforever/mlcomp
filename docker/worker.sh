#!/usr/bin/env bash

python worker.py
PYTHONPATH=../ supervisord -c supervisord.conf
#PYTHONPATH=../ python worker/__main__.py worker 0