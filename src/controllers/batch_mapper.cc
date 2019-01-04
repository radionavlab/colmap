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

#include "controllers/batch_mapper.h"

#include "util/misc.h"

namespace colmap {
namespace {

size_t TriangulateImage(const BatchMapperOptions& options,
                        const Image& image, BatchMapper* mapper) {
  std::cout << "  => Continued observations: " << image.NumPoints3D()
            << std::endl;
  const size_t num_tris =
      mapper->TriangulateImage(options.Triangulation(), image.ImageId());
  std::cout << "  => Added observations: " << num_tris << std::endl;
  return num_tris;
}

void IterativeGlobalRefinement(const BatchMapperOptions& options,
                               BatchMapper* mapper) {
  PrintHeading1("Retriangulation");
  CompleteAndMergeTracks(options, mapper);
  std::cout << "  => Retriangulated observations: "
            << mapper->Retriangulate(options.Triangulation()) << std::endl;


  for (int i = 0; i < options.ba_global_max_refinements; ++i) {
    const size_t num_observations =
        mapper->GetReconstruction().ComputeNumObservations();
    size_t num_changed_observations = 0;
    BundleAdjustmentOptions custom_options = options.GlobalBundleAdjustment();

    // Configure options
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

    num_changed_observations += CompleteAndMergeTracks(options, mapper);
    num_changed_observations += FilterPoints(options, mapper);
    const double changed =
        static_cast<double>(num_changed_observations) / num_observations;
    std::cout << StringPrintf("  => Changed observations: %.6f", changed)
              << std::endl;
    if (changed < options.ba_global_max_refinement_change) {
      break;
    }
  }

  FilterImages(options, mapper);
}

void ExtractColors(const std::string& image_path, const image_t image_id,
                   Reconstruction* reconstruction) {
  if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
    std::cout << StringPrintf("WARNING: Could not read image %s at path %s.",
                              reconstruction->Image(image_id).Name().c_str(),
                              image_path.c_str())
              << std::endl;
  }
}

}  // namespace

size_t FilterPoints(const BatchMapperOptions& options,
                    BatchMapper* mapper) {
  const size_t num_filtered_observations =
      mapper->FilterPoints(options.Mapper());
  std::cout << "  => Filtered observations: " << num_filtered_observations
            << std::endl;
  return num_filtered_observations;
}

size_t FilterImages(const BatchMapperOptions& options,
                    BatchMapper* mapper) {
  const size_t num_filtered_images = mapper->FilterImages(options.Mapper());
  std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
  return num_filtered_images;
}

size_t CompleteAndMergeTracks(const BatchMapperOptions& options,
                              BatchMapper* mapper) {
  const size_t num_completed_observations =
      mapper->CompleteTracks(options.Triangulation());
  std::cout << "  => Merged observations: " << num_completed_observations
            << std::endl;
  const size_t num_merged_observations =
      mapper->MergeTracks(options.Triangulation());
  std::cout << "  => Completed observations: " << num_merged_observations
            << std::endl;
  return num_completed_observations + num_merged_observations;
}

BatchMapper::Options BatchMapperOptions::Mapper() const {
  BatchMapper::Options options = mapper;
  options.abs_pose_refine_focal_length = ba_refine_focal_length;
  options.abs_pose_refine_extra_params = ba_refine_extra_params;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  options.num_threads = num_threads;
  return options;
}

IncrementalTriangulator::Options BatchMapperOptions::Triangulation()
    const {
  IncrementalTriangulator::Options options = triangulation;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  return options;
}

BundleAdjustmentOptions BatchMapperOptions::GlobalBundleAdjustment()
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

bool BatchMapperOptions::Check() const {
  CHECK_OPTION_GT(min_num_matches, 0);
  CHECK_OPTION_GT(min_focal_length_ratio, 0);
  CHECK_OPTION_GT(max_focal_length_ratio, 0);
  CHECK_OPTION_GE(max_extra_param, 0);
  CHECK_OPTION_GT(ba_global_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_global_max_refinements, 0);
  CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
  CHECK_OPTION(Mapper().Check());
  CHECK_OPTION(Triangulation().Check());
  return true;
}

BatchMapperController::BatchMapperController(
    const BatchMapperOptions* options, const std::string& image_path,
    const std::string& database_path,
    ReconstructionManager* reconstruction_manager)
    : options_(options),
      image_path_(image_path),
      database_path_(database_path),
      reconstruction_manager_(reconstruction_manager) {
  CHECK(options_->Check());
  RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
  RegisterCallback(LAST_IMAGE_REG_CALLBACK);
}

void BatchMapperController::Run() {
  if (!LoadDatabase()) {
    return;
  }
  BatchMapper::Options init_mapper_options = options_->Mapper();
  Reconstruct(init_mapper_options);
  GetTimer().PrintMinutes();
}

bool BatchMapperController::LoadDatabase() {
  PrintHeading1("Loading database");

  Database database(database_path_);
  Timer timer;
  timer.Start();
  const size_t min_num_matches = static_cast<size_t>(options_->min_num_matches);
  database_cache_.Load(database, min_num_matches, options_->ignore_watermarks,
                       options_->image_names);
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

void BatchMapperController::Reconstruct(
    const BatchMapper::Options& init_mapper_options) {

  //////////////////////////////////////////////////////////////////////////////
  // Setup
  //////////////////////////////////////////////////////////////////////////////

  BatchMapper mapper(&database_cache_);

  // Is there a sub-model before we start the reconstruction? I.e. the user
  // has imported an existing reconstruction.
  const bool initial_reconstruction_given = reconstruction_manager_->Size() > 0;
  CHECK_LE(reconstruction_manager_->Size(), 1) << "Can only resume from a "
                                                  "single reconstruction, but "
                                                  "multiple are given.";
  size_t reconstruction_idx;
  if (!initial_reconstruction_given) {
    reconstruction_idx = reconstruction_manager_->Add();
  } else {
    reconstruction_idx = 0;
  }

  Reconstruction& reconstruction =
      reconstruction_manager_->Get(reconstruction_idx);

  mapper.BeginReconstruction(&reconstruction);

  ///////////////////////////////////////////////////////////////////////////
  // Register all images
  ////////////////////////////////////////////////////////////////////////////

  const EIGEN_STL_UMAP(image_t, class Image) images_map = reconstruction.Images();
  std::vector<image_t> image_ids;
  image_ids.reserve(images_map.size());
  for(const auto& key_val: images_map) { image_ids.push_back(key_val.first); }

  for(image_t& next_image_id: image_ids) {
    const Image& next_image = reconstruction.Image(next_image_id);

    if(next_image.IsRegistered()) {
      continue;
    }

    CHECK(
        next_image.HasTvecPrior() && 
        next_image.HasQvecPrior() && 
        next_image.HasCovariancePrior()) << "Batch mapper requires priors "
                                            "for all images.";

    PrintHeading1(StringPrintf("Registering image #%d (%d)", next_image_id,
                               reconstruction.NumRegImages() + 1));

    std::cout << StringPrintf("  => Image sees %d / %d points",
                              next_image.NumVisiblePoints3D(),
                              next_image.NumObservations())
              << std::endl;

    bool reg_next_success =
        mapper.RegisterNextImage(options_->Mapper(), next_image_id);

    if (reg_next_success) {
      TriangulateImage(*options_, next_image, &mapper);

      if (options_->extract_colors) {
        ExtractColors(image_path_, next_image_id, &reconstruction);
      }

      Callback(NEXT_IMAGE_REG_CALLBACK);
    } else {
      std::cout << "  => Could not register, trying another image."
                << std::endl;
    }
  }

  Callback(LAST_IMAGE_REG_CALLBACK);

  ///////////////////////////////////////////////////////////////////////////
  // Global Refinement
  ////////////////////////////////////////////////////////////////////////////

  IterativeGlobalRefinement(*options_, &mapper);

  const bool kDiscardReconstruction = false;
  mapper.EndReconstruction(kDiscardReconstruction); 
}

}  // namespace colmap
