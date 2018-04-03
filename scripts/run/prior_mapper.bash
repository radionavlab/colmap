#!/bin/bash

prior_mapper \
    --import_path `pwd`/sparse/0 \
    --export_path `pwd`/sparse/0 \
    --min_set_path `pwd`/min \
    --max_set_path `pwd`/max \
    --min_pose_path `pwd`/min/image_poses.txt \
    --max_pose_path `pwd`/max/image_poses.txt
