#!/bin/bash
echo "Starting application..."
gunicorn backend:app --timeout 180 --workers 1
