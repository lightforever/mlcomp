#!/usr/bin/env bash

pushd /app/mlcomp/migration
PYTHONPATH=../../ python manage.py version_control
PYTHONPATH=../../ python manage.py upgrade
popd

PYTHONPATH=../ python server/__main__.py start