// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "controllers/incremental_mapper.h"
#include "controllers/bundle_adjustment.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"
#include "base/reconstruction_manager.h"
#include "sfm/incremental_triangulator.h"
#include <Eigen/Geometry>

using namespace colmap;

namespace {
    typedef Eigen::Matrix3d R_t;
    typedef Eigen::Vector3d tvec_t;
    typedef Eigen::Vector4d qvec_t;
    typedef Eigen::Matrix<double, 9, 9> cov_t;
}

void ReadCameraMeasurements(const std::string& path,
                         std::vector<std::string>* image_names,
                         std::vector<tvec_t>* ricI_vec,
                         std::vector<tvec_t>* rciC_vec,
                         std::vector<R_t>* RIC_vec,
                         std::vector<cov_t>* cov_vec) {
  std::vector<std::string> lines = ReadTextFileLines(path);
  for (const auto line : lines) {
    std::stringstream line_parser(line);

    // Buffers to store data in
    std::string image_name = "";
    tvec_t ricI;
    tvec_t rciC;
    R_t RIC;
    cov_t cov;

    // Read data into buffers
    // Translation
    line_parser >> 
        image_name >> 
        ricI(0) >> 
        ricI(1) >> 
        ricI(2) >> 
        rciC(0) >>
        rciC(1) >>
        rciC(2);

    // Rotation
    for(int i = 0; i < 3; i ++) {
        for(int j = 0; j < 3; j++) {
            line_parser >> RIC(j, i);
        }
    }

    // Covariance
    for(int i = 0; i < 9; i++) {
        for(int j = 0; j < 9; j++) {
            line_parser >> cov(j, i);
        }
    }

    // Push data into vectors
    image_names->push_back(image_name);
    ricI_vec->push_back(ricI);
    rciC_vec->push_back(rciC);
    RIC_vec->push_back(RIC);
    cov_vec->push_back(cov);
  }
}

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string export_path;
  std::string metadata_path;

  // ssba --database_path database.db --image_path images --metadata_path
  // images/image_metadata_local.txt --export_path sparse
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("metadata_path", &metadata_path);
  options.AddRequiredOption("export_path", &export_path);
  options.AddBundleAdjustmentOptions();
  options.Parse(argc, argv);

  /* GAME PLAN
   * 1) Create new reconstruction
   * 2) Load all images into reconstruction
   * 3) Load camera params
   * 3) Load priors into reconstruction
   * 4) Tringulate all 3D points based on priors
   * 5) Run global bundle adjustment
   */


  // Create a new reconstruction
  ReconstructionManager reconstruction_manager;
  const size_t reconstruction_idx = reconstruction_manager.Add();
  Reconstruction& reconstruction = reconstruction_manager.Get(reconstruction_idx);

  // Load images and features from database
  Database db(*options.database_path);
  DatabaseCache db_cache;
  const size_t min_num_matches = 8;
  const bool ignore_watermarks = true;
  const std::set<std::string> image_cache_names;
  db_cache.Load(db, min_num_matches, ignore_watermarks, image_cache_names);

  // Set up reconstruction
  std::cout << "Setting up reconstruction..." << std::endl;
  {
    reconstruction.Load(db_cache);
    reconstruction.SetUp(&db_cache.CorrespondenceGraph());
    const EIGEN_STL_UMAP(image_t, class Image)& images = db_cache.Images();

    for(auto it = images.begin(); it != images.end(); ++it) {
      reconstruction.RegisterImage(it->first);
    }
  }

  // Add pre-computed params to camera
  std::cout << "Loading camera params..." << std::endl;

  const camera_t camera_id = 1;
  const size_t width = 3840;
  const size_t height = 2160;
  const std::string params = "1662.07,1920,1080,-0.021983";

  Camera& cam = reconstruction.Camera(camera_id);
  cam.SetWidth(width);
  cam.SetHeight(height);
  cam.SetParamsFromString(params);

  if(!cam.VerifyParams()) {
    std::cerr << "Camera params wrong!" << std::endl;
    return EXIT_FAILURE;
  }

  // Load the priors.
  std::cout << "Loading priors..." << std::endl;

  std::vector<std::string> image_names;
  std::vector<tvec_t> ricI_vec;
  std::vector<tvec_t> rciC_vec;
  std::vector<R_t> RIC_vec;
  std::vector<cov_t> cov_vec;
  ReadCameraMeasurements(
          metadata_path, 
          &image_names, 
          &ricI_vec, 
          &rciC_vec, 
          &RIC_vec, 
          &cov_vec);

  std::unordered_map< std::string, std::tuple<tvec_t, tvec_t, qvec_t, Eigen::Matrix<double, 6, 6> > > image_priors;
  for(size_t i = 0; i < rciC_vec.size(); i++) {
    image_priors.insert({
                image_names[i], 
                std::make_tuple(ricI_vec[i],
                                rciC_vec[i],
                                RotationMatrixToQuaternion(RIC_vec[i]), 
                                cov_vec[i].bottomRightCorner<6,6>())      // Bottom right corner is [eIC, ricI]
    });
  }

  // Sets both prior and initial guess
  reconstruction.AddPriors(image_priors);

  // Triangulate the 3D position of all points
  std::cout << "Triangulating..." << std::endl;

  IncrementalTriangulator triangulator(&db_cache.CorrespondenceGraph(), &reconstruction);
  const std::vector<image_t>& image_ids = reconstruction.RegImageIds();
  const IncrementalTriangulator::Options triangulator_options;
  for(const image_t& image_id: image_ids) {
    triangulator.TriangulateImage(triangulator_options, image_id);
    triangulator.CompleteImage(triangulator_options, image_id);
  }
  triangulator.CompleteAllTracks(triangulator_options);
  triangulator.MergeAllTracks(triangulator_options);

  for(size_t i = 0; i < 2; i++) {
    if(i != 0) {
      // Retriangulate
      std::cout << "Retriangulating..." << std::endl;
      {
        triangulator.Retriangulate(triangulator_options);
        triangulator.CompleteAllTracks(triangulator_options);
        triangulator.MergeAllTracks(triangulator_options);
      }
    }

    // Run robust global BA
    std::cout << "Running robust global BA..." << std::endl;
    {
      OptionManager options_(options);
      options_.bundle_adjustment->priors = true;
      options_.bundle_adjustment->loss_function_type = 
          BundleAdjustmentOptions::LossFunctionType::CAUCHY;
      options_.bundle_adjustment->loss_function_scale = 1;
      options_.bundle_adjustment->solver_options.max_num_iterations = 100;
      options_.bundle_adjustment->refine_focal_length = false;
      options_.bundle_adjustment->refine_extra_params = false;

      BundleAdjustmentController ba_controller(options_, &reconstruction);
      ba_controller.Start();
      ba_controller.Wait();
    }
  }

  // Filter points
  std::cout << "Filtering 3D points..." << std::endl;
  {
    const double max_reproj_error = 4.0;
    const double min_tri_angle = 10; // deg
    reconstruction.FilterAllPoints3D(max_reproj_error, min_tri_angle);
  }

  // Iteratively run global BA
  for(size_t i = 0; i < 5; i++) {

    // Run strict global BA
    std::cout << "Running strict global BA..." << std::endl;
    {
      OptionManager options_(options);
      options_.bundle_adjustment->priors = true;
      options_.bundle_adjustment->loss_function_type = 
          BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
      options_.bundle_adjustment->loss_function_scale = 1;
      options_.bundle_adjustment->solver_options.max_num_iterations = 100;
      options_.bundle_adjustment->refine_focal_length = true;
      options_.bundle_adjustment->refine_extra_params = true;

      BundleAdjustmentController ba_controller(options_, &reconstruction);
      ba_controller.Start();
      ba_controller.Wait();
    }

    // Complete and merge 3D point tracks
    std::cout << "Completing and merging tracks..." << std::endl;
    {
      triangulator.CompleteAllTracks(triangulator_options);
      triangulator.MergeAllTracks(triangulator_options);
    }

    // Filter points
    std::cout << "Filtering 3D points..." << std::endl;
    {
      const double max_reproj_error = 4.0;
      const double min_tri_angle = 10; // deg
      reconstruction.FilterAllPoints3D(max_reproj_error, min_tri_angle);
    }
  }

  // Calculate covariance
  {
    
    BundleAdjustmentOptions::CovarianceOptions cov_options;
    cov_options.compute = true;
    cov_options.center_point = Eigen::Vector3d(-1.226307, -0.113879, 0.667136);
    cov_options.radius = 1.0;

    OptionManager options_(options);
    options_.bundle_adjustment->priors = true;
    options_.bundle_adjustment->cov = cov_options;
    options_.bundle_adjustment->loss_function_type = 
        BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
    options_.bundle_adjustment->loss_function_scale = 1;
    options_.bundle_adjustment->solver_options.max_num_iterations = 100;
    options_.bundle_adjustment->refine_focal_length = false;
    options_.bundle_adjustment->refine_extra_params = false;

    BundleAdjustmentController ba_controller(options_, &reconstruction);
    ba_controller.Start();
    ba_controller.Wait(); 
  }

  // Save output
  std::cout << "Saving output..." << std::endl;
  reconstruction.Write(export_path);
  reconstruction.WriteText(export_path);
  // reconstruction.ExportPLY(export_path + "/model.ply");
  
  std::cout << "Success!" << std::endl;

  return EXIT_SUCCESS;
}
