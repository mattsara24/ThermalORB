/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef SUPERPOINTEXTRACTOR_H
#define SUPERPOINTEXTRACTOR_H
#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>
#if (CV_MAJOR_VERSION > 3)
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#define CV_LOAD_IMAGE_UNCHANGED cv::IMREAD_UNCHANGED
#endif

#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>

#include <iostream>
#include <memory>


namespace ORB_SLAM3
{

/*
REF CLASS
class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
}; */

class SuperPointExtractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    SuperPointExtractor(int nfeatures, std::string filePath);

    ~SuperPointExtractor(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    int operator()( cv::InputArray _image,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea);


    at::Tensor nms_fast(at::Tensor in_corners, int h, int w, int dist_thresh);
    at::Tensor getPtsFromHeatmap(at::Tensor heatmap, float conf_thresh);
    at::Tensor depth2space(at::Tensor nodust, int blocks);
    at::Tensor flattenDetections(at::Tensor semi);
    at::Tensor matToTensor(cv::Mat frame, torch::Device device);



protected:

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);

    int nfeatures;
    std::vector<int> umax;
    torch::jit::script::Module module;
    torch::Device device = torch::kCPU;

};

} //namespace ORB_SLAM

#endif

