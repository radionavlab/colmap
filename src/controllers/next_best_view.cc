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

#include "controllers/next_best_view.h"
#include "controllers/covariance_evaluator.h"
#include "sfm/batch_mapper.h"

#include "util/misc.h"

namespace colmap {

namespace {
  size_t FilterPoints(const BatchMapper::Options& options,
                      BatchMapper* mapper) {
    const size_t num_filtered_observations = mapper->FilterPoints(options);
    std::cout << "  => Filtered observations: " << num_filtered_observations
              << std::endl;
    return num_filtered_observations;
  }
  
  size_t FilterImages(const BatchMapper::Options& options,
                      BatchMapper* mapper) {
    const size_t num_filtered_images = mapper->FilterImages(options);
    std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
    return num_filtered_images;
  }
  
  size_t CompleteAndMergeTracks(const IncrementalTriangulator::Options& options,
                                BatchMapper* mapper) {
    const size_t num_completed_observations =
        mapper->CompleteTracks(options);
    std::cout << "  => Merged observations: " << num_completed_observations
              << std::endl;
    const size_t num_merged_observations =
        mapper->MergeTracks(options);
    std::cout << "  => Completed observations: " << num_merged_observations
              << std::endl;
    return num_completed_observations + num_merged_observations;
  }

  size_t TriangulateImage(const IncrementalTriangulator::Options& options,
                          const Image& image, 
                          BatchMapper* mapper) {
    std::cout << "  => Continued observations: " << image.NumPoints3D()
              << std::endl;
    const size_t num_tris =
        mapper->TriangulateImage(options, image.ImageId());
    std::cout << "  => Added observations: " << num_tris << std::endl;
    return num_tris;
  }
  
  void IterativeGlobalRefinement(const NextBestViewOptions& options,
                                 BatchMapper* mapper) {
    PrintHeading1("Retriangulation");
    CompleteAndMergeTracks(options.TriangulatorOptions(), mapper);
    std::cout << "  => Retriangulated observations: "
              << mapper->Retriangulate(options.TriangulatorOptions()) << std::endl;
   
    for (int i = 0; i < options.ba_global_max_refinements; ++i) {
      const size_t num_observations =
          mapper->GetReconstruction().ComputeNumObservations();
      size_t num_changed_observations = 0;
  
      // Configure options
      BundleAdjustmentOptions custom_options = options.BAOptions();
      custom_options.using_priors = true;
      if(i < 2) {
        custom_options.loss_function_type = 
          BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
      } else {
        custom_options.loss_function_type = 
          BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
      }
  
      PrintHeading1("Global bundle adjustment");
      mapper->AdjustGlobalBundle(custom_options);
  
      num_changed_observations += CompleteAndMergeTracks(options.TriangulatorOptions(), mapper);
      num_changed_observations += FilterPoints(options.BatchMapperOptions(), mapper);
      const double changed =
          static_cast<double>(num_changed_observations) / num_observations;
      std::cout << StringPrintf("  => Changed observations: %.6f", changed)
                << std::endl;
      if (changed < options.ba_global_max_refinement_change) {
        break;
      }
    }
  
    FilterImages(options.BatchMapperOptions(), mapper);
  }
  
  void ExtractColors(const std::string& image_path, 
                     const image_t image_id,
                     Reconstruction* reconstruction) {
    if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
      std::cout << StringPrintf("WARNING: Could not read image %s at path %s.",
                                reconstruction->Image(image_id).Name().c_str(),
                                image_path.c_str())
                << std::endl;
    }
  }


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

  // Class encapsulating data about NBV keypoints
  class Keypoint {
    private:
      point3D_t point3D_id_;
      double score_;

    public: 
      Keypoint(point3D_t point3D_id, double score)
        : point3D_id_(point3D_id),
          score_(score) 
      {}

      Keypoint() : Keypoint(0, -1) {}

      double Score() const {
        return score_;
      }
  
      void Adjust(const Keypoint& other) {
        score_ = std::max(0.0, score_ - other.score_);
      }
  };
  
  // Class encapsulating data about NBV images
  class CandidateImage {
    private:
      image_t image_id_;
      double score_;
      std::map<point3D_t, Keypoint> keypoints_;
  
    public:
      CandidateImage(const image_t image_id)
        : image_id_(image_id)
      {};

      CandidateImage() : CandidateImage(0) {};

      void RemoveKeypoint(const point3D_t point3D_id) {
       keypoints_.erase(point3D_id) ;
      }

      void AddKeypoint(const point3D_t point3D_id, 
                       const Point3D& point3D, 
                       const Image& image) {
        if(!point3D.HasCovariance()) {
          return;
        }
  
        const Eigen::Matrix3d& covariance = point3D.Covariance();
        Eigen::EigenSolver<Eigen::Matrix3d> es;
        es.compute(covariance,  true);
        auto eigenvalues = es.eigenvalues();
        auto eigenvectors = es.eigenvectors();
  
        double score = 0;
        for(size_t idx = 0; idx < 3; ++idx) {
          score += (eigenvalues[idx] * (eigenvectors.col(idx).cross(image.ViewingDirection()))).squaredNorm();
        } 
  
        keypoints_[point3D_id] = Keypoint(point3D_id, score);
      }
  
      void CalcScore() {
        score_ = std::max(
          0.0, 
          std::accumulate(
            std::begin(keypoints_), 
            std::end(keypoints_), 
            0.0,
            [] (double score, const std::map<point3D_t, Keypoint>::value_type& kv)
              { return score + kv.second.Score(); }
          )
        );

      };

      double Score() const {
        return score_;
      }
  
      // Given that another image has been selected for the camera network,
      // adjust the value of this image. Functions as a dispatcher that reduces
      // the value of keypoints already seen by the other image.
      void Adjust(const CandidateImage& other) {
        // Adjust the value of keypoints seen by other image
        for (auto it = other.keypoints_.begin(); it != other.keypoints_.end(); ++it) {
          try { 
            Keypoint this_kp = this->keypoints_.at(it->first);
            Keypoint other_kp = it->second;
            this_kp.Adjust(other_kp);
          } catch(const std::out_of_range& e) {
            // Do nothing.
          }
        }
  
        // Rescore this image
        this->CalcScore();
      };
  
      // std::greater operator for sorting 
      bool operator>(const CandidateImage& rhs) const {
        return this->score_ > rhs.score_;
      }; 

      image_t ImageId() const {
        return image_id_;
      };
  };
}; // namespace


BatchMapper::Options NextBestViewOptions::BatchMapperOptions() const {
  BatchMapper::Options options = batch_mapper_options;
  options.abs_pose_refine_focal_length = ba_refine_focal_length;
  options.abs_pose_refine_extra_params = ba_refine_extra_params;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  options.num_threads = num_threads;
  return options;
}

IncrementalTriangulator::Options NextBestViewOptions::TriangulatorOptions()
    const {
  IncrementalTriangulator::Options options = triangulator_options;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  return options;
}

BundleAdjustmentOptions NextBestViewOptions::BAOptions()
    const {
  BundleAdjustmentOptions options;
  options.solver_options.function_tolerance = 0.0;
  options.solver_options.gradient_tolerance = 1.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_global_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.minimizer_progress_to_stdout = true;
  options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
  options.print_summary = true;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  return options;
}

bool NextBestViewOptions::Check() const {
  CHECK_OPTION_GT(min_num_matches, 0);
  CHECK_OPTION_GT(min_focal_length_ratio, 0);
  CHECK_OPTION_GT(max_focal_length_ratio, 0);
  CHECK_OPTION_GE(max_extra_param, 0);
  CHECK_OPTION_GT(ba_global_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_global_max_refinements, 0);
  CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
  CHECK_OPTION_GT(keypoint_uncertainty_bound, 0);
  CHECK_OPTION_GT(camera_network_size, 0);
  CHECK_OPTION_GT(keypoint_uncertainty_ratio, 0);
  CHECK_OPTION(BatchMapperOptions().Check());
  CHECK_OPTION(TriangulatorOptions().Check());
  CHECK_OPTION(BAOptions().Check());
  return true;
}

NextBestViewController::NextBestViewController(
    const NextBestViewOptions* options, 
    const std::string& image_path,
    const std::string& database_path,
    Reconstruction* reconstruction)
    : options_(options),
      image_path_(image_path),
      database_path_(database_path),
      reconstruction_(reconstruction) {
  CHECK(!(nullptr == reconstruction));
  CHECK(options_->Check());

  candidate_image_names_ = options_->candidate_image_names;
  incorporated_image_names_ = options_->incorporated_image_names;

  // Ensure that the two image sets are disjoint
  for(const std::string& image_name: incorporated_image_names_) {
    candidate_image_names_.erase(image_name);
  }

}

void NextBestViewController::EvaluateCovariance() {

  PrintHeading1("Evaluating Covariance");

  const std::vector<image_t>& reg_image_ids = this->reconstruction_->RegImageIds();

  CHECK_GE(reg_image_ids.size(), 2) << "ERROR: Need at least two views.";

  // Avoid degeneracies in bundle adjustment.
  this->reconstruction_->FilterObservationsWithNegativeDepth();

  BundleAdjustmentOptions ba_options;
  ba_options.solver_options.minimizer_progress_to_stdout = true;
  ba_options.cov.compute = true;
  ba_options.cov.keypoints = this->keypoints_;
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
  bundle_adjuster.Solve(this->reconstruction_);

  GetTimer().PrintMinutes();
  return;
}

void NextBestViewController::Reconstruct() {

  //////////////////////////////////////////////////////////////////////////////
  // Setup
  //////////////////////////////////////////////////////////////////////////////

  BatchMapper mapper(&this->database_cache_);
  mapper.BeginReconstruction(this->reconstruction_);

  //////////////////////////////////////////////////////////////////////////////
  // Register Images
  //////////////////////////////////////////////////////////////////////////////

  std::set<image_t> image_ids;
  for(const std::string& image_name: this->incorporated_image_names_) {
    const Image* image = this->reconstruction_->FindImageWithName(image_name);
    CHECK(!(nullptr == image)) << "Image does not exist!";
    image_t image_id = image->ImageId();

    if(this->reconstruction_->IsImageRegistered(image_id)) {
      continue;
    }

    CHECK(
        image->HasTvecPrior() && 
        image->HasQvecPrior() && 
        image->HasCovariancePrior()) << "Batch mapper requires priors "
                                       "for all images.";


    PrintHeading1(StringPrintf("Registering image #%d (%d)", image_id,
        this->reconstruction_->NumRegImages() + 1));

    std::cout << StringPrintf("  => Image sees %d / %d points",
                              image->NumVisiblePoints3D(),
                              image->NumObservations())
              << std::endl;

    bool reg_next_success =
        mapper.RegisterNextImage(this->options_->BatchMapperOptions(), image_id);

    if (reg_next_success) {
      TriangulateImage(this->options_->TriangulatorOptions(), *image, &mapper);

      if (options_->extract_colors) {
        ExtractColors(image_path_, image_id, this->reconstruction_);
      }

      image_ids.insert(image_id);

    } else {
      std::cout << "  => Could not register, trying another image."
                << std::endl;
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // Global Refinement
  ////////////////////////////////////////////////////////////////////////////

  IterativeGlobalRefinement(*options_, &mapper);
}

void NextBestViewController::Run() {

  //////////////////////////////////////////////////////////////////////////////
  // Setup
  //////////////////////////////////////////////////////////////////////////////
 
  if (!LoadDatabase()) {
    return;
  }
  this->reconstruction_->Load(database_cache_);

  MarkKeypoints();

  // Build a map between candidate images and 3D keypoints
  std::map<image_t, CandidateImage> candidate_image_map;
  const CorrespondenceGraph& cg = this->database_cache_.CorrespondenceGraph();
  for(point3D_t point3D_id: this->keypoints_) {
    const Point3D& point3D = this->reconstruction_->Point3D(point3D_id);
    const Track& track = point3D.Track();
    for(const TrackElement& te: track.Elements()) {
      const std::vector<CorrespondenceGraph::Correspondence>& correspondences = 
        cg.FindCorrespondences(te.image_id, te.point2D_idx);
      for(const auto& correspondence: correspondences) {

        // Get a reference to the candidate image. If none exists, create one
        image_t image_id = correspondence.image_id;
        try {
          candidate_image_map.at(image_id);
        } catch(const std::out_of_range& e) {
          candidate_image_map[image_id] = CandidateImage(image_id);
        }
        
        // Associate a keypoint with the candidate image 
        candidate_image_map[image_id].AddKeypoint(point3D_id, point3D, this->reconstruction_->Image(image_id));
      }
    }
  }
 
  //////////////////////////////////////////////////////////////////////////////
  // Main Loop
  //////////////////////////////////////////////////////////////////////////////
  bool terminate = false;
  while(true) {
    // Perform reconstruction with incorportaed images 
    Reconstruct();

    // Evaluate the covariance
    EvaluateCovariance();

    // Cleanup
    ValidateKeypoints();
 
    // No more images to process
    if(0 == candidate_image_names_.size()) {
      std::cout << "No more images to process. Terminating." << std::endl;
      break;
    } 

    // Terminal conditions met
    if(Finished()) {
      break;
    }

    // Terminate requested
    if(terminate) {
      break;
    }
    
    PrintHeading1("Selecting Camera Network");


    // Retrieve candidate images
    std::vector<CandidateImage> candidate_images;
    for(const std::string& image_name: this->candidate_image_names_) {
      const Image* image = this->reconstruction_->FindImageWithName(image_name);
      CHECK_NOTNULL(image);

      auto kv = candidate_image_map.find(image->ImageId());
      if(kv == candidate_image_map.end()) { continue; }

      // Copy the image into a vector.
      candidate_images.push_back(kv->second);

      // Score the image in the vector.
      candidate_images.back().CalcScore();
    }

    // Minimum number of images is greater than number of candidates
    if(candidate_image_names_.size() <= options_->camera_network_size) {
        incorporated_image_names_.insert(candidate_image_names_.begin(), candidate_image_names_.end());
        candidate_image_names_.clear();
        continue;
    }
 
    // Select the camera network
    for(size_t idx = 0; idx < options_->camera_network_size; ++idx) {
      std::sort(candidate_images.begin(), candidate_images.end(), std::greater<CandidateImage>());

      const CandidateImage selected_image = candidate_images.front();
      candidate_images.erase(candidate_images.begin());

      if(selected_image.Score() < this->options_->min_candidate_score) {
        if(idx == 0) {
          terminate = true;
          break;
        } else {
          continue;
        }
      }

      const Image& image = this->reconstruction_->Image(selected_image.ImageId());
      const std::string& image_name = image.Name();

      std::cout << "Adding " << image_name << " with score " 
        << selected_image.Score() << " to camera network." << std::endl;

      incorporated_image_names_.insert(image_name);
      candidate_image_names_.erase(image_name);

      std::for_each(
          candidate_images.begin(), 
          candidate_images.end(), 
          [selected_image](CandidateImage &ci){ ci.Adjust(selected_image); });
    }
  } 

  GetTimer().PrintMinutes();
  this->reconstruction_->TearDown();
}

bool NextBestViewController::Finished() {
  const size_t num_keypoints = this->keypoints_.size();
  size_t num_well_constrained_keypoints = 0;

  CHECK_GT(num_keypoints, 0) << "No keypoints marked!";

  const auto points3D = this->reconstruction_->Points3D();
  for(point3D_t point3D_id: this->keypoints_) {
    // For a symmetric matrix (covariance), the matrix 2-norm 
    // is equal to the largest eigenvalue
    if(std::sqrt(points3D.at(point3D_id).Covariance().norm())
        < options_->keypoint_uncertainty_bound) {
      ++num_well_constrained_keypoints;
    }
  }

  std::cout << 
    num_well_constrained_keypoints 
    << " / " 
    << num_keypoints 
    << " keypoints are contrained below " 
    << this->options_->keypoint_uncertainty_bound 
    << " meters"
    << std::endl;

  return (
      static_cast<double>(num_well_constrained_keypoints) /
      static_cast<double>(num_keypoints)
    ) > this->options_->keypoint_uncertainty_ratio;
}

void NextBestViewController::MarkKeypoints() {
  this->keypoints_.clear();
  for(const auto& point3D_kv: this->reconstruction_->Points3D()) {
    if(options_->roi.Contains(point3D_kv.second.XYZ())) {
      this->keypoints_.insert(point3D_kv.first);
    }
  }
}

bool NextBestViewController::LoadDatabase() {
  PrintHeading1("Loading database");

  Database database(database_path_);
  Timer timer;
  timer.Start();
  const size_t min_num_matches = static_cast<size_t>(options_->min_num_matches);
  database_cache_.Load(database, min_num_matches, options_->ignore_watermarks, {});
  std::cout << std::endl;
  timer.PrintMinutes();

  std::cout << std::endl;

  if (database_cache_.NumImages() == 0) {
    std::cout << "WARNING: No images with matches found in the database."
              << std::endl
              << std::endl;
    return false;
  }

  return true;
}

void NextBestViewController::ValidateKeypoints() {
  auto point_map = reconstruction_->Points3D();
  std::set<point3D_t> to_remove;
  for(point3D_t point_id: this->keypoints_) {
    try {
      point_map.at(point_id);
    } catch(const std::out_of_range& e) {
      // For some reason or another, the reconstruction has determined that this
      // point is invalid and has removed it from the reconstruction. Remove it
      // from the keypoint list as it no longer exists.
      to_remove.insert(point_id);
    }
  }

  for(point3D_t point_id: to_remove) {
    this->keypoints_.erase(point_id);
  }

}

}  // namespace colmap
