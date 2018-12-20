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

#ifndef COLMAP_SRC_BASE_ROI_H_
#define COLMAP_SRC_BASE_ROI_H_

#include <Eigen/Core>
#include <vector>

#include "util/logging.h"

namespace colmap {

  struct PolyhedronFace {
    // POD
    std::vector<Eigen::Vector3d> points_;

    // Normal vector
    Eigen::Vector3d Normal() const {
      CHECK_GE(points_.size(), 3);
      Eigen::Vector3d v1 = points_[1] - points_[0];
      Eigen::Vector3d v2 = points_[2] - points_[0];
      Eigen::Vector3d n = v1.cross(v2);
      n = n / n.norm();
      return n;
    };

  };

  struct Polyhedron {
    // POD
    std::vector<PolyhedronFace> faces_;
    static constexpr double BOUND = -1e-15;

    // Is point contained within polyhedron
    bool Contains(const Eigen::Vector3d& point) const {
      for(const auto& face: faces_) {
        Eigen::Vector3d p2f = face.points_[0] - point;
        double d = p2f.dot(face.Normal()) / p2f.norm();
        if(d < BOUND) { return false; }
      }
      return true;
    };
  };

}  // namespace colmap
#endif  // COLMAP_SRC_BASE_ROI_H_
