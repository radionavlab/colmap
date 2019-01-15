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

#ifndef COLMAP_SRC_CONTROLLERS_NEXT_BEST_VIEW_H_
#define COLMAP_SRC_CONTROLLERS_NEXT_BEST_VIEW_H_

#include "base/reconstruction.h"
#include "base/roi.h"
#include "base/database_cache.h"
#include "util/threading.h"
#include "sfm/batch_mapper.h"

namespace colmap {

struct NextBestViewOptions {
 public:
  // Polyhedron specifying the region of interest in which keypoints reside. If
  // polyhedron is not specified, all points are considered keypoints.
  Polyhedron roi;

  // Which images are already incorporated
  std::set<std::string> incorporated_image_names;

  // Which images are being considered for incorporation
  std::set<std::string> candidate_image_names;

  // Uncertainty size in meters below which a keypoint is considered
  // 'sufficiently-contrained'
  double keypoint_uncertainty_bound = 0.01;

  // Ratio of 'well-constrained' keypoints to toal keypoints above which the
  // reconstruction is considered 'well-contrained'.  Functions as a terminal
  // condition for the NBV solver.
  double keypoint_uncertainty_ratio = 0.95;

  // The minimum score a candidate image can have to be considered a candidate
  double min_candidate_score = 100.00;

  // The number of cameras in a camera network selection
  size_t camera_network_size = 10;

  // The minimum number of matches for inlier matches to be considered.
  int min_num_matches = 15;

  // Whether to ignore the inlier matches of watermark image pairs.
  bool ignore_watermarks = false;

  // Whether to extract colors for reconstructed points.
  bool extract_colors = true;

  // The number of threads to use during reconstruction.
  int num_threads = -1;

  // Thresholds for filtering images with degenerate intrinsics.
  double min_focal_length_ratio = 0.1;
  double max_focal_length_ratio = 10.0;
  double max_extra_param = 1.0;

  // Which intrinsic parameters to optimize during the reconstruction.
  bool ba_refine_focal_length = false;
  bool ba_refine_principal_point = false;
  bool ba_refine_extra_params = false;

  // The maximum number of global bundle adjustment iterations.
  int ba_global_max_num_iterations = 50;

  // The thresholds for iterative bundle adjustment refinements.
  int ba_global_max_refinements = 5;
  double ba_global_max_refinement_change = 0.0005;

  BatchMapper::Options BatchMapperOptions() const;
  IncrementalTriangulator::Options TriangulatorOptions() const;
  BundleAdjustmentOptions BAOptions() const;

  bool Check() const;

 private:
  friend class OptionManager;
  BatchMapper::Options batch_mapper_options;
  IncrementalTriangulator::Options triangulator_options;
};


// Class that selects a camera network by greedily choosing and incorporating
// images that will reduce a covariance metric.
class NextBestViewController : public Thread {
 public:
  NextBestViewController(const NextBestViewOptions* options,
                         const std::string& image_path,
                         const std::string& database_path,
                         Reconstruction* reconstruction);

 private:
  void Run();
  bool LoadDatabase();

  /* Marks points within a specified region of interest as keypoints. These
   * keypoints are tracked as the reconstruction size grows and only their
   * covariance is evaluated. 
   */
  void MarkKeypoints();

  /* Determines if the terminal conditions are met */
  bool Finished();

  /* Ensures that the keypoint list contains only valid keypoints. Sometimes the
   * reconstruction determines that keypoints are bad and tosses them.
   */
  void ValidateKeypoints();

  /* Evaluates the covariance of keypoints */
  void EvaluateCovariance();

  void Reconstruct();

  const NextBestViewOptions* options_;
  std::set<point3D_t> keypoints_;
  std::set<std::string> candidate_image_names_;
  std::set<std::string> incorporated_image_names_;
  const std::string image_path_;
  const std::string database_path_;
  Reconstruction* reconstruction_;
  DatabaseCache database_cache_;

};

}  // namespace colmap

#endif  // COLMAP_SRC_CONTROLLERS_NEXT_BEST_VIEW_H_
