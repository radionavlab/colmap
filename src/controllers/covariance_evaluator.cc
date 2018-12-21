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

#include "controllers/covariance_evaluator.h"

#include <ceres/ceres.h>

#include "optim/bundle_adjustment.h"
#include "base/roi.h"
#include "util/misc.h"

namespace colmap {
namespace {

// Callback functor called after each bundle adjustment iteration.
class BundleAdjustmentIterationCallback : public ceres::IterationCallback {
 public:
  explicit BundleAdjustmentIterationCallback(Thread* thread)
      : thread_(thread) {}

  virtual ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) {
    CHECK_NOTNULL(thread_);
    thread_->BlockIfPaused();
    if (thread_->IsStopped()) {
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    } else {
      return ceres::SOLVER_CONTINUE;
    }
  }

 private:
  Thread* thread_;
};
}  // namespace

CovarianceEvaluatorController::CovarianceEvaluatorController(
    const OptionManager& options, Reconstruction* reconstruction)
    : options_(options), reconstruction_(reconstruction) {}

void CovarianceEvaluatorController::Run() {
  CHECK_NOTNULL(reconstruction_);

  PrintHeading1("Global bundle adjustment");

  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  if (reg_image_ids.size() < 2) {
    std::cout << "ERROR: Need at least two views." << std::endl;
    return;
  }

  // Avoid degeneracies in bundle adjustment.
  reconstruction_->FilterObservationsWithNegativeDepth();

  BundleAdjustmentOptions ba_options = *options_.bundle_adjustment;
  ba_options.solver_options.minimizer_progress_to_stdout = true;
  ba_options.cov.compute = true;
  ba_options.using_priors = true;

  // TODO HACKING
  double MIN_X=-1, MAX_X=2;
  double MIN_Y=0,  MAX_Y=2;
  double MIN_Z=0,  MAX_Z=2;

  PolyhedronFace f1; f1.points_ = {
    Eigen::Vector3d(MIN_X, MIN_Y, MIN_Z),
    Eigen::Vector3d(MAX_X, MIN_Y, MIN_Z),
    Eigen::Vector3d(MAX_X, MIN_Y, MAX_Z),
    Eigen::Vector3d(MIN_X, MIN_Y, MAX_Z),
  };

  PolyhedronFace f2; f2.points_ = {
    Eigen::Vector3d(MAX_X, MAX_Y, MIN_Z),
    Eigen::Vector3d(MIN_X, MAX_Y, MIN_Z),
    Eigen::Vector3d(MIN_X, MAX_Y, MAX_Z),
    Eigen::Vector3d(MAX_X, MAX_Y, MAX_Z),
  };

  PolyhedronFace f3; f3.points_ = {
    Eigen::Vector3d(MAX_X, MIN_Y, MIN_Z),
    Eigen::Vector3d(MAX_X, MAX_Y, MIN_Z),
    Eigen::Vector3d(MAX_X, MAX_Y, MAX_Z),
    Eigen::Vector3d(MAX_X, MIN_Y, MAX_Z),
  };

  PolyhedronFace f4; f4.points_ = {
    Eigen::Vector3d(MIN_X, MAX_Y, MIN_Z),
    Eigen::Vector3d(MIN_X, MIN_Y, MIN_Z),
    Eigen::Vector3d(MIN_X, MIN_Y, MAX_Z),
    Eigen::Vector3d(MIN_X, MAX_Y, MAX_Z),
  };


  ba_options.cov.ROI.faces_ = {f1, f2, f3, f4};
  // END TODO HACKING

  BundleAdjustmentIterationCallback iteration_callback(this);
  ba_options.solver_options.callbacks.push_back(&iteration_callback);

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  // Run bundle adjustment.
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  bundle_adjuster.Solve(reconstruction_);

  GetTimer().PrintMinutes();
}

}  // namespace colmap