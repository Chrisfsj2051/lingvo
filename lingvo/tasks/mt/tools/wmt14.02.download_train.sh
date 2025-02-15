#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -eu

. wmt14_lib.sh

mkdir -p "${ROOT}/raw"
# ============================================================================
# Download WMT data.
# From the WMT14 website (En-De):
echo "
http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz" \
  | aria2c --all-proxy = 'http://127.0.0.1:1087' -x16 -s1 -j1 --dir="${ROOT}/raw" -i -
