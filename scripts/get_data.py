#!/usr/bin/env python
import kagglehub

path = kagglehub.dataset_download("dgawlik/nyse", path="data/raw")
print("Path to dataset files:", path)
