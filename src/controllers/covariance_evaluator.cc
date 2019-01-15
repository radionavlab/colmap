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

bool CovarianceEvaluatorOptions::Check() const {
  ba_options.Check();
  return true;
}

CovarianceEvaluatorController::CovarianceEvaluatorController(
    const CovarianceEvaluatorOptions* options, 
    ReconstructionManager* reconstruction_manager)
    : options_(options), 
      reconstruction_manager_(reconstruction_manager) 
    {
      options_->Check();   
    }


void CovarianceEvaluatorController::Run() {
  CHECK_NOTNULL(reconstruction_manager_);

  const bool initial_reconstruction_given = reconstruction_manager_->Size() > 0;
  CHECK_EQ(reconstruction_manager_->Size(), 1) << "CovarianceEvaluator must resume "
                                                  "a previous reconstruction.";
  size_t reconstruction_idx;
  if (!initial_reconstruction_given) {
    std::cerr << "Cannot run CovarianceEvaluator with an empty reconstruction." << std::endl;
    return;    
  } else {
    reconstruction_idx = 0;
  }

  Reconstruction& reconstruction =
      reconstruction_manager_->Get(reconstruction_idx);

  PrintHeading1("Global bundle adjustment");

  const std::vector<image_t>& reg_image_ids = reconstruction.RegImageIds();

  if (reg_image_ids.size() < 2) {
    std::cout << "ERROR: Need at least two views." << std::endl;
    return;
  }

  // Avoid degeneracies in bundle adjustment.
  reconstruction.FilterObservationsWithNegativeDepth();

  BundleAdjustmentOptions ba_options = options_->ba_options;
  ba_options.solver_options.minimizer_progress_to_stdout = true;
  ba_options.cov.compute = true;
  ba_options.using_priors = true;

  BundleAdjustmentIterationCallback iteration_callback(this);
  ba_options.solver_options.callbacks.push_back(&iteration_callback);

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  // Run bundle adjustment.
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  bundle_adjuster.Solve(&reconstruction);

  GetTimer().PrintMinutes();
}

}  // namespace colmap
