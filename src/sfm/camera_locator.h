// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Tucker Haydon

#ifndef COLMAP_SRC_SFM_BATCH_MAPPER_H_
#define COLMAP_SRC_SFM_BATCH_MAPPER_H_

#include "base/database.h"
#include "base/database_cache.h"
#include "base/reconstruction.h"
#include "optim/bundle_adjustment.h"
#include "sfm/incremental_triangulator.h"
#include "util/alignment.h"

namespace colmap {

class BatchMapper {
 public:
  struct Options {
    // Minimum number of inliers for initial image pair.
    int init_min_num_inliers = 100;

    // Maximum error in pixels for two-view geometry estimation for initial
    // image pair.
    double init_max_error = 4.0;

    // Maximum forward motion for initial image pair.
    double init_max_forward_motion = 0.95;

    // Minimum triangulation angle for initial image pair.
    double init_min_tri_angle = 16.0;

    // Maximum number of trials to use an image for initialization.
    int init_max_reg_trials = 2;

    // Maximum reprojection error in absolute pose estimation.
    double abs_pose_max_error = 12.0;

    // Minimum number of inliers in absolute pose estimation.
    int abs_pose_min_num_inliers = 30;

    // Minimum inlier ratio in absolute pose estimation.
    double abs_pose_min_inlier_ratio = 0.25;

    // Whether to estimate the focal length in absolute pose estimation.
    bool abs_pose_refine_focal_length = true;

    // Whether to estimate the extra parameters in absolute pose estimation.
    bool abs_pose_refine_extra_params = true;

    // Thresholds for bogus camera parameters. Images with bogus camera
    // parameters are filtered and ignored in triangulation.
    double min_focal_length_ratio = 0.1;  // Opening angle of ~130deg
    double max_focal_length_ratio = 10;   // Opening angle of ~5deg
    double max_extra_param = 1;

    // Maximum reprojection error in pixels for observations.
    double filter_max_reproj_error = 4.0;

    // Minimum triangulation angle in degrees for stable 3D points.
    double filter_min_tri_angle = 1.5;

    // Maximum number of trials to register an image.
    int max_reg_trials = 3;

    // Number of threads.
    int num_threads = -1;

    bool Check() const;
  };

  // Create batch mapper. The database cache must live for the entire
  // life-time of the incremental mapper.
  explicit BatchMapper(const DatabaseCache* database_cache);

  // Prepare the mapper for a new reconstruction, which might have existing
  // registered images (in which case `RegisterNextImage` must be called)
  void BeginReconstruction(Reconstruction* reconstruction);

  // Cleanup the mapper after the current reconstruction is done. If the
  // model is discarded, the number of total and shared registered images will
  // be updated accordingly.
  void EndReconstruction(const bool discard);

  // Attempt to register image to the existing model.
  bool RegisterNextImage(const Options& options, const image_t image_id);

  // Triangulate observations of image.
  size_t TriangulateImage(const IncrementalTriangulator::Options& tri_options,
                          const image_t image_id);

  // Retriangulate image pairs that should have common observations according to
  // the scene graph but don't due to drift, etc. To handle drift, the employed
  // reprojection error thresholds should be relatively large. If the thresholds
  // are too large, non-robust bundle adjustment will break down; if the
  // thresholds are too small, we cannot fix drift effectively.
  size_t Retriangulate(const IncrementalTriangulator::Options& tri_options);

  // Complete tracks by transitively following the scene graph correspondences.
  // This is especially effective after bundle adjustment, since many cameras
  // and point locations might have improved. Completion of tracks enables
  // better subsequent registration of new images.
  size_t CompleteTracks(const IncrementalTriangulator::Options& tri_options);

  // Merge tracks by using scene graph correspondences. Similar to
  // `CompleteTracks`, this is effective after bundle adjustment and improves
  // the redundancy in subsequent bundle adjustments.
  size_t MergeTracks(const IncrementalTriangulator::Options& tri_options);

  // Global bundle adjustment using Ceres Solver.
  bool AdjustGlobalBundle(const BundleAdjustmentOptions& ba_options);

  // Filter images and point observations.
  size_t FilterImages(const Options& options);
  size_t FilterPoints(const Options& options);

  const Reconstruction& GetReconstruction() const;

  // Get changed 3D points, since the last call to `ClearModifiedPoints3D`.
  const std::unordered_set<point3D_t>& GetModifiedPoints3D();

  // Clear the collection of changed 3D points.
  void ClearModifiedPoints3D();

 private:
  // Class that holds all necessary data from database in memory.
  const DatabaseCache* database_cache_;

  // Class that holds data of the reconstruction.
  Reconstruction* reconstruction_;

  // Class that is responsible for incremental triangulation.
  std::unique_ptr<IncrementalTriangulator> triangulator_;

  // Images that have been filtered in current reconstruction.
  std::unordered_set<image_t> filtered_images_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_SFM_BATCH_MAPPER_H_
