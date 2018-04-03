#!/bin/bash

# Run feature extractor
feature_extractor \
    --database_path `pwd`/database.db \
    --image_path `pwd`/max \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 1 \
    --SiftExtraction.gpu_index 0 \
    --SiftExtraction.use_gpu 1

# Match features
exhaustive_matcher \
    --database_path `pwd`/database.db \
    --SiftMatching.use_gpu 1 \
    --ExhaustiveMatching.block_size 100

# Run sparse reconstruction
mkdir -p `pwd`/sparse
mapper \
    --database_path `pwd`/database.db \
    --image_path `pwd`/max \
    --export_path `pwd`/sparse \
    --Mapper.multiple_models 1

# Change ownership of files to belong to user
sudo chown -R "${SUDO_USER:-$USER}:${SUDO_USER:-$USER}" `pwd`
