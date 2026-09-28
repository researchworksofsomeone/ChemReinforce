#!/bin/bash
echo "Setting up Pinggy SSH tunnel..."
echo "Use this script manually if Kaggle cells are unavailable."
echo "ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:22 <PINGGY_TOKEN>@a.pinggy.io"
