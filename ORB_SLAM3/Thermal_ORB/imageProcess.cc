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



#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include <opencv2/core/core.hpp>
#if (CV_MAJOR_VERSION > 3)
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#define CV_LOAD_IMAGE_UNCHANGED cv::IMREAD_UNCHANGED
#endif

//#include<System.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>


#include <iostream>
#include <memory>

#define CONF_THRESH 0.000002


at::Tensor getPtsFromHeatmap(at::Tensor heatmap, float conf_thresh){
    heatmap = heatmap.squeeze();
    auto size = heatmap.sizes();
    auto comp = at::where(heatmap >= conf_thresh);
    auto sparseMap = (heatmap >= conf_thresh);
    if(comp[0].size(0) == 0)
        return at::zeros({3,0});   

    auto pts = at::zeros({3,comp[0].size(0)});
    pts[0].slice(0,0,pts.size(1)) = comp[1];
    pts[1].slice(0,0,pts.size(1)) = comp[0]; // TODO VERIFY 
    //pts[2].slice(0,0,pts.size(1)) = heatmap(comp[0],comp[1]); TODO FIX
    std::cout << pts << std::endl;
    /*
FUNCTIONS

    def getPtsFromHeatmap(self, heatmap):
        '''
        :param self:
        :param heatmap:
            np (H, W)
        :return:
        '''
        heatmap = heatmap.squeeze()
        # print("heatmap sq:", heatmap.shape)
        H, W = heatmap.shape[0], heatmap.shape[1]
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        self.sparsemap = (heatmap >= self.conf_thresh)
        if len(xs) == 0:
            return np.zeros((3, 0))
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys # abuse of ys, xs
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]  # check the (x, y) here
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        return pts
*/
    return heatmap;
}

at::Tensor depth2space(at::Tensor nodust, int blocks)
{
        int block_size = blocks;
        int block_size_sq = blocks*blocks;
        
        auto output = nodust.permute({0,2,3,1});
        
        auto size = output.sizes();
        int s_depth = size[3] / block_size_sq;
        int s_width = size[2] * block_size;
        int s_height = size[1] * block_size;

        auto t_1 = output.reshape({1,size[1], size[2], block_size_sq, s_depth});
        auto spl = t_1.split(block_size,3);
        std::vector<at::Tensor> stack;
        for(int i = 0; i < spl.size(); i++)
            stack.push_back(spl[0].reshape({1, size[1] ,s_width, s_depth}));

        output = at::stack(stack,0).transpose(0,1).permute({0,2,1,3,4});
        output = output.reshape({1, s_height, s_width, s_depth});
        return output;
}

at::Tensor flattenDetections(at::Tensor semi){
    auto dense =  at::softmax(semi, 1, c10::optional<c10::ScalarType>()); // [1,65,30,40]
    auto nodust = dense.slice(1,0, dense.size(1) - 1); // [1,64,30,40]

    auto heatmap = depth2space(nodust, 8);
    return heatmap;
}

at::Tensor matToTensor(cv::Mat frame, torch::Device device) {
    cv::resize(frame, frame, cv::Size(240,320));
    auto input_tensor = torch::from_blob(frame.data, {1, 240, 320, 1}, torch::kByte);

    input_tensor = input_tensor.permute({0, 3, 1, 2});
    input_tensor = input_tensor.to(torch::kFloat);
    input_tensor = input_tensor.to(device);
    return input_tensor;
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: ThermalORB <path-to-exported-script-module>\n";
    return -1;
  }

  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Using GPU." << std::endl;
    //device = torch::kCUDA;
  } else {
      std::cout << "Using CPU.\n";
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
    module.to(device);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  std::cout << "SUCCESSFULLY LOADED MODEL" << argv[1] << std::endl;

  std::vector<torch::jit::IValue> inputs;
  cv::Mat img = cv::imread("../test.png");
  if(img.empty())
  {
    std::cout << "Could not read the image: " << std::endl;
    return 1;
  }

  auto tens = matToTensor(img, device);

  inputs.push_back(tens);
  
  auto output = module.forward(inputs).toGenericDict();

  auto semi = output.at("semi").toTensor();
  auto descriptors = output.at("desc").toTensor();



  auto heatmap = flattenDetections(semi);
  auto pts = getPtsFromHeatmap(heatmap.to(torch::kCPU), CONF_THRESH);
  /*
  for (auto & pt : pts) {
      std::cout << pt <<  std::endl;
  }
  */
}
