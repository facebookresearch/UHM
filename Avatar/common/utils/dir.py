# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import os
import sys

def make_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

