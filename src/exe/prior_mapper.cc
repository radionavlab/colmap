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

  std::string import_path;
  std::string export_path;
  std::string metadata_path;

  OptionManager options;
  options.AddRequiredOption("import_path", &import_path);
  options.AddRequiredOption("export_path", &export_path);
  options.AddRequiredOption("metadata_path", &metadata_path);
  options.AddBundleAdjustmentOptions();
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(import_path);

  /* GAME PLAN
  * 1) Read in the measurements
  * 2) Apply similarity transform to data. Bring reconstruction into global frame.
  * 3) Insert the measurements into Colmap
  * 4) Run global BA
  */

  /* 1) Read in the camera measurements */
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

  /* 2) Align reconstruction with ECEF measurements to bring it into global frame. */
  RANSACOptions ransac_options;
  ransac_options.max_error = 0.01;
  bool alignment_success = reconstruction.AlignRobust(
          image_names,
          ricI_vec,
          3,
          ransac_options);

  if (alignment_success) {
    std::cout << " => Alignment succeeded" << std::endl;
    reconstruction.Write("sparse/aligned");
    reconstruction.WriteText("sparse/aligned");
  } else {
    std::cout << " => Alignment failed" << std::endl;
    return EXIT_FAILURE;
  }
 

  /* 3) Insert the measurements into reconstruction */
  std::unordered_map< std::string, std::tuple<tvec_t, qvec_t, Eigen::Matrix<double, 6, 6> > > image_priors;
  for(size_t i = 0; i < rciC_vec.size(); i++) {
    image_priors.insert({
                image_names[i], 
                std::make_tuple(ricI_vec[i], 
                                RotationMatrixToQuaternion(RIC_vec[i]), 
                                cov_vec[i].bottomRightCorner<6,6>())      // Bottom right corner is [eIC, ricI]
    });
  }

  reconstruction.AddPriors(image_priors);

  /* 4) Run global BA */
  // Configure BA
  options.bundle_adjustment->cov.compute = false;
  options.bundle_adjustment->priors = true;
  // options.bundle_adjustment->refine_focal_length = true;
  // options.bundle_adjustment->refine_extra_params = true;
  // options.bundle_adjustment->refine_principal_point = true;
  // options.bundle_adjustment->loss_function_type = 
  //     BundleAdjustmentOptions::LossFunctionType::CAUCHY;
  // options.bundle_adjustment->loss_function_scale = 5;
  options.bundle_adjustment->solver_options.max_num_iterations = 100;

  // Run BA
  BundleAdjustmentController ba_controller(options, &reconstruction);
  ba_controller.Start();
  ba_controller.Wait();

  // Save output
  reconstruction.Write("sparse/priors");
  reconstruction.WriteText("sparse/priors");
  
  std::cout << "Success!" << std::endl;

  return EXIT_SUCCESS;
}
