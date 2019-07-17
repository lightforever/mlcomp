#!/usr/bin/env bash

python worker.py
PYTHONPATH=../ supervisord -c supervisord.conf