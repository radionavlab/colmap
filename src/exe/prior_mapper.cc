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

void ReadCameraMeasurements(const std::string& path,
                         std::vector<std::string>* image_names,
                         std::vector<Eigen::Vector3d>* TvecPriorsGlobal,
                         std::vector<Eigen::Vector3d>* TvecPriorsCamera,
                         std::vector<Eigen::Vector3d>* EvecPriorsCamera,
                         std::vector< Eigen::Matrix<double, 6, 6> >* PriorsCovariance) {
  std::vector<std::string> lines = ReadTextFileLines(path);
  for (const auto line : lines) {
    std::stringstream line_parser(line);

    // Buffers to store data in
    std::string image_name = "";
    Eigen::Vector3d TvecPriorGlobal;
    Eigen::Vector3d TvecPriorCamera;
    Eigen::Vector3d EvecPriorCamera;
    Eigen::Matrix<double, 6, 6> PriorCovariance;

    // Read data into buffers
    line_parser >> 
        image_name >> 
        TvecPriorGlobal(0) >> 
        TvecPriorGlobal(1) >> 
        TvecPriorGlobal(2) >> 
        TvecPriorCamera(0) >>
        TvecPriorCamera(1) >>
        TvecPriorCamera(2) >>
        EvecPriorCamera(0) >>
        EvecPriorCamera(1) >>
        EvecPriorCamera(2);

    for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 6; j++) {
            line_parser >> PriorCovariance(j, i);
        }
    }

    // Push data into vectors
    image_names->push_back(image_name);
    TvecPriorsGlobal->push_back(TvecPriorGlobal);
    TvecPriorsCamera->push_back(TvecPriorCamera);
    EvecPriorsCamera->push_back(EvecPriorCamera);
    PriorsCovariance->push_back(PriorCovariance);
  }
}

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string import_path;
  std::string export_path;
  std::string image_pose_path;

  OptionManager options;
  options.AddRequiredOption("import_path", &import_path);
  options.AddRequiredOption("export_path", &export_path);
  options.AddRequiredOption("image_pose_path", &image_pose_path);
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
  std::vector<Eigen::Vector3d> TvecPriorsGlobal;
  std::vector<Eigen::Vector3d> TvecPriorsCamera;
  std::vector<Eigen::Vector3d> EvecPriorsCamera;
  std::vector< Eigen::Matrix<double, 6, 6> > PriorsCovariance;
  ReadCameraMeasurements(
          image_pose_path, 
          &image_names, 
          &TvecPriorsGlobal, 
          &TvecPriorsCamera, 
          &EvecPriorsCamera, 
          &PriorsCovariance);

  /* 2) Align reconstruction with ECEF measurements to bring it into global frame. */
  bool alignment_success = reconstruction.Align(
          image_names,
          TvecPriorsGlobal,
          3);

  if (alignment_success) {
    std::cout << " => Alignment succeeded" << std::endl;
  } else {
    std::cout << " => Alignment failed" << std::endl;
    return EXIT_FAILURE;
  }
 

  /* 3) Insert the measurements into reconstruction */
  std::unordered_map< std::string, std::tuple<Eigen::Vector3d, Eigen::Vector4d, Eigen::Matrix<double, 6, 6> > > image_priors;
  for(size_t i = 0; i < TvecPriorsCamera.size(); i++) {
    // Transform 3-2-1 euler angle to quaternion
    const Eigen::Vector4d QvecPriorCamera = 
        RotationMatrixToQuaternion(
                EulerAnglesToRotationMatrix(
                    EvecPriorsCamera[i](0), 
                    EvecPriorsCamera[i](1), 
                    EvecPriorsCamera[i](2)));

    image_priors.insert({
                image_names[i], 
                std::make_tuple(TvecPriorsCamera[i], QvecPriorCamera, PriorsCovariance[i])
    });
  }

  reconstruction.AddPriors(image_priors);

  /* 4) Run global BA */
  options.bundle_adjustment->compute_covariance = false;
  options.bundle_adjustment->normalize = false;
  options.bundle_adjustment->solver_options.max_num_iterations = 1000;
  BundleAdjustmentController ba_controller(options, &reconstruction);
  ba_controller.Start();
  ba_controller.Wait();
 
  // reconstruction.Write(export_path);
  reconstruction.WriteText(export_path);
  
  /* 
   * Game Plan 
   * 1) Iterate through all the 3D points and extract those that are within the ROI
   * 2) Iterate through all of the extra images and score them
   * 3) Select the image with the greatest score
   * 4) Add image to problem. Use priors as initial estimate
   * 5) Run bundle adjustment
   * 6) Evaluate constraints
   * 7) GOTO 2
   */

//     const Eigen::Vector3d centerECEF(-742015.201696, -5462219.446174, 3198014.314080);
//     const Eigen::Vector3d upperECEF = centerECEF + Eigen::Vector3d(2,2,2); 
//     const Eigen::Vector3d lowerECEF = centerECEF + Eigen::Vector3d(-2,-2,-2); 
//     
//     /* 1) Filter points out that are not within the ROI */
//     const auto points3D = FilterPointsROI(reconstruction.Points3D());
// 
//     /* 2) Score candidate images */
//     const auto candidateImages = getCandidateImages();
// 
//     auto scoreImage = [&](img& img, const point p) -> double {};
//     for(img& img : candidateImages) {
//         for(point p : points3D) {
//             img.score += scoreImage(img, p);
//         }
//     }
// 
//     /* 3) Select image with the greatest score */
//     auto img_compare = [&](const img& a, const img& b) -> bool {a.score > b.score};
//     const img img = std::max_element(img.begin(), img.end(), img_compare);
// 
//     /* 4) Add image to problem */
// 
//     /* 5) Run BA */
// 
//     /* 6) Evaluate constraints */
// 
//     /* 7) GOTO 2 */
//     
//   std::cout << "Success!" << std::endl;

  return EXIT_SUCCESS;
}
