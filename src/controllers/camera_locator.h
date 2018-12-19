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

#ifndef COLMAP_SRC_CONTROLLERS_CAMERA_LOCATOR_H_
#define COLMAP_SRC_CONTROLLERS_CAMERA_LOCATOR_H_

#include "base/reconstruction_manager.h"
// #include "sfm/batch_mapper.h"
#include "util/threading.h"
#include <boost/lexical_cast.hpp>

namespace colmap {

struct CameraLocatorOptions {
 public:
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
  bool ba_refine_focal_length = true;
  bool ba_refine_principal_point = true;
  bool ba_refine_extra_params = true;

  // The maximum number of global bundle adjustment iterations.
  int ba_global_max_num_iterations = 100;

  // The thresholds for iterative bundle adjustment refinements.
  int ba_global_max_refinements = 5;
  double ba_global_max_refinement_change = 0.0005;

  // BatchMapper::Options Mapper() const;
  IncrementalTriangulator::Options Triangulation() const;
  BundleAdjustmentOptions GlobalBundleAdjustment() const;

  bool Check() const;

 private:
  friend class OptionManager;
  // BatchMapper::Options mapper;
  IncrementalTriangulator::Options triangulation;
};

// Class that controls the incremental mapping procedure by iteratively
// initializing reconstructions from the same scene graph.
class CameraLocatorController : public Thread {
 public:

  CameraLocatorController(const CameraLocatorOptions* options,
                          const std::string& image_list_path,
                          const std::string& database_path,
                          ReconstructionManager* reconstruction_manager);

 private:
  void Run();
  bool LoadDatabase();
  void Reconstruct(const BatchMapper::Options& init_mapper_options);
  void Locate();

  const CameraLocatorOptions* options_;
  const std::string image_list_path_;
  const std::string database_path_;
  ReconstructionManager* reconstruction_manager_;
  DatabaseCache database_cache_;
};

// Globally filter points and images in mapper.
size_t FilterPoints(const CameraLocatorOptions& options,
                    BatchMapper* mapper);
size_t FilterImages(const CameraLocatorOptions& options,
                    BatchMapper* mapper);

// Globally complete and merge tracks in mapper.
size_t CompleteAndMergeTracks(
    const CameraLocatorOptions& options,
    BatchMapper* mapper);

}  // namespace colmap

#endif  // COLMAP_SRC_CONTROLLERS_CAMERA_LOCATOR_H_
