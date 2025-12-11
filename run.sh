#!/bin/bash
. ./.venv/bin/activate

export PYTHONPATH=$PWD/src

streamlit run src/job_dash.py --server.headless true --server.port 8507
