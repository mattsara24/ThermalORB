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

#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>


#include <iostream>
#include <memory>

#define CONF_THRESH 0.02

at::Tensor nms_fast(at::Tensor in_corners, int h, int w, int dist_thresh) {
    auto grid = at::zeros({h,w}).to(torch::kInt);
    auto inds = at::zeros({h,w}).to(torch::kInt);
    
    auto inds1 = (in_corners[2]).argsort(0);
    auto corners = at::zeros(in_corners.sizes());
    for(int i = 0; i < in_corners.size(0); i++) {
      for (int j = in_corners.size(1) - 1 ; j >= 0; j--) {
        corners[i][j] = in_corners[i][inds1[j]];
      }
    }

    auto rcorners = at::zeros({2, corners.size(1)});
    for(int i = 0; i < 2; i++) {
        rcorners[i] = corners[i];
    }
    
    if(rcorners.size(1) == 0){
        return at::zeros({3,0}).to(torch::kInt);
    } else if (rcorners.size(1) == 1) {
        return at::vstack({rcorners, in_corners[2]}).reshape({3,1});
    }
    
    auto rcornersT = rcorners.transpose(0,1).round().to(torch::kInt);
    for(int i = 0; i < rcornersT.size(0); i++){
       grid[rcorners[1][i].item<int>()][rcorners[0][i].item<int>()] = 1;
       inds[rcorners[1][i].item<int>()][rcorners[0][i].item<int>()] = i;
    }

    int pad = dist_thresh;
    grid = torch::nn::functional::pad(grid, torch::nn::functional::PadFuncOptions({pad,pad,pad,pad}).mode(torch::kConstant));
    std::cout << grid.sizes() << std::endl;
    for(int i = 0; i < rcornersT.size(0); i++){
      int pt[2] = {rcornersT[i][0].item<int>() + pad, rcornersT[i][1].item<int>() + pad};
      if (grid[pt[1]][pt[0]].item<int>() == 1){
          for(int i = - 1 * pad; i < pad; i++){
              for(int j = - 1 * pad; j < pad; j++){
                  grid[pt[1] + i][pt[0] + j] = 0;
              }
          }
          grid[pt[1]][pt[0]] = -1;
      }
    }

    auto keep = at::where(grid == -1);
    std::cout << "KEEPS" << keep[0].size(0) << std::endl;
    keep[0] -= pad;
    keep[1] -= pad;
    auto inds_keep = at::zeros(keep[0].size(0));

    for(int i = 0; i < keep[0].size(0); i++) {
      inds_keep[i] = inds[keep[0][i].item<int>()][keep[1][i].item<int>()];
    }
    std::cout << "test" << std::endl;

    auto out = at::zeros({3,keep[0].size(0)});
    for(int i = 0; i < inds_keep.size(0); i++) {
        out[0][i] = corners[0][inds_keep[i].item<int>()];
        out[1][i] = corners[1][inds_keep[i].item<int>()];
        out[2][i] = corners[2][inds_keep[i].item<int>()];

    }

    std::cout << "test" << std::endl;
    auto values = out[2];
    auto inds2 = (-1* values).argsort();
    for(int i = 0; i < inds2.size(0); i++) {
        out[0][i] = out[0][inds2[i].item<int>()];
        out[1][i] = out[1][inds2[i].item<int>()];
        out[2][i] = out[2][inds2[i].item<int>()];
    }

    return out;
}

at::Tensor getPtsFromHeatmap(at::Tensor heatmap, float conf_thresh){
    heatmap = heatmap.squeeze();
    auto comp = at::where(heatmap >= conf_thresh);
    std::cout << "PASSING POINTS:" << comp[0].size(0) << std::endl;
    std::cout << "FAILING POINTS:" << comp[1].size(0) << std::endl;

    if(comp[0].size(0) == 0)
        return at::zeros({3,0});   

    auto pts = at::zeros({3,comp[0].size(0)});
    for(int i = 0; i < comp[0].size(0) && i < comp[1].size(0); i++){
      pts[1][i] = comp[0][i];
      if(i < comp[1].size(0)){
        pts[0][i] = comp[1][i];
        pts[2][i] = heatmap[comp[0][i].item()][comp[1][i].item()];
      }
    }
    std::cout << "NUMBER OF POINTS BEFORE NMS:" << heatmap.sizes() << std::endl;
    pts = nms_fast(pts, heatmap.size(0), heatmap.size(1), 3); // TODO -> Parametrize
    std::cout << "NUMBER OF POINTS AFTER NMS:" << pts.sizes() << std::endl;
    auto inds = pts[2].argsort(0);
    for(int i = 0; i < pts.size(0); i++) {
      for (int j = pts.size(1) - 1 ; j >= 0; j--) {
        pts[i][j] = pts[i][inds[j]];
      }
    }  
    std::cout << "FINAL POINTS:" << pts.sizes() << std::endl;
    for(int i =0; i < pts.size(1); i++){
        std::cout<< pts[0][i].item<int>() << " " << pts[1][i].item<int>() << " " << pts[2][i].item<float>() << " " << std::endl;
    }

    //TODO -> Remove points around the border of the image
    return pts;
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
        std::cout << "FINAL OUTPUT" << output.sizes() << at::mean(output)<<std::endl;
        return output;
        
}

at::Tensor flattenDetections(at::Tensor semi){
    auto dense =  torch::softmax(semi,1); // [1,65,30,40]
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

  cv::Mat img = cv::imread("../test.png");
  if(img.empty())
  {
    std::cout << "Could not read the image: " << std::endl;
    return 1;
  }

  std::vector<torch::jit::IValue> inputs;
  auto tens = matToTensor(img, device);

  inputs.push_back(tens);
  auto output = module.forward(inputs).toGenericDict();

  auto semi = output.at("semi").toTensor();
  auto descriptors = output.at("desc").toTensor();

  auto heatmap = flattenDetections(semi);
  auto pts = getPtsFromHeatmap(heatmap, CONF_THRESH).to(torch::kCPU);


  for(int i = 0; i < pts.size(1); i++){
    int x = pts[0][i].item<int>();
    int y = pts[1][i].item<int>();
    cv::circle(img,cv::Point(x,y), 5, (0,0,255), -1);
  } 


  //cv::imshow("Display window", img);
  //int j = cv::waitKey(0); // Wait for a keystroke in the window
 
  //TODO -> Translate to C++  --- > dense_desc = nn.functional.interpolate(coarse_desc, scale_factor=(self.cell, self.cell), mode='bilinear')
  auto dn = torch::norm(descriptors, 2, 1);
  auto desc = descriptors.div(torch::unsqueeze(dn, 1));

  desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]
  desc = desc.to(torch::kCPU);

  /*

    # norm the descriptor
    def norm_desc(desc):
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        return desc
    dense_desc = norm_desc(dense_desc)

    # extract descriptors
    dense_desc_cpu = dense_desc.cpu().detach().numpy()

  */
}
