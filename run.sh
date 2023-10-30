#!/bin/bash

# Get the directory of the current script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the Django project directory
cd "$DIR/emotion_detector_project"

# Create a virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    python3 -m venv env
fi

# Activate the virtual environment
source env/bin/activate

# Install requirements using pip
pip install -r requirements.txt

# Run the Django manage.py script
python manage.py runserver
