#!/bin/bash

prior_mapper \
    --database_path database.db \
    --import_path model/0 \
    --export_path model/0 \
    --min_set_path min \
    --max_set_path max \
    --min_pose_path min/image_poses.txt \
    --max_pose_path max/image_poses.txt \
    --image_path max \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 1 \
    --SiftExtraction.gpu_index 0 \
    --SiftExtraction.use_gpu 1 \
    --SiftMatching.use_gpu 1 \
    --ExhaustiveMatching.block_size 250
