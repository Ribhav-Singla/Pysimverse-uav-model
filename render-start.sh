#!/bin/bash
pm2 start app.py --interpreter python3
pm2 start worker.js
pm2 logs
