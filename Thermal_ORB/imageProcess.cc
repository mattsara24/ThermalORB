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

#define CONF_THRESH 0.06

at::Tensor nms_fast(at::Tensor in_corners, int h, int w, int dist_thresh) {
    auto grid = at::zeros({h,w}).to(torch::kInt);
    auto inds = at::zeros({h,w}).to(torch::kInt);
    
    auto inds1 = (in_corners[2]).argsort(0,true);
    auto corners = in_corners.index({torch::indexing::Slice(0, in_corners.size(0), 1), inds1.slice(0,0, in_corners.size(1))});

    auto rcorners = corners.index({torch::indexing::Slice(0,2,1)});
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
    for(int i = 0; i < rcornersT.size(0); i++){
      int pt[2] = {rcornersT[i][0].item<int>() + pad, rcornersT[i][1].item<int>() + pad};
      if (grid[pt[1]][pt[0]].item<int>() == 1){
          grid.index_put_({torch::indexing::Slice(pt[1] - pad, pt[1] + pad + 1, 1), torch::indexing::Slice(pt[0] - pad, pt[0] + pad + 1, 1)},0);
          grid[pt[1]][pt[0]] = -1;
      }
    }

    auto keep = at::where(grid == -1);
    keep[0] -= pad;
    keep[1] -= pad;
    auto inds_keep = inds.index({keep[0].slice(0,0,keep[0].size(0)), keep[1].slice(0,0,keep[0].size(0))}).to(torch::kLong);

    auto out = corners.index({torch::indexing::Slice(0,3,1), inds_keep.slice(0,0,inds_keep.size(0))  });


    auto values = out[2];
    auto inds2 = values.argsort(0, true);
    out = out.index({torch::indexing::Slice(0,3,1), inds2.slice(0,0,inds2.size(0))});

    return out;
}

at::Tensor getPtsFromHeatmap(at::Tensor heatmap, float conf_thresh){
    heatmap = heatmap.squeeze();
    auto comp = at::where(heatmap >= conf_thresh);
    //std::cout << "PASSING POINTS:" << comp[0].size(0) << std::endl;
    //std::cout << "FAILING POINTS:" << comp[1].size(0) << std::endl;

    if(comp[0].size(0) == 0)
        return at::zeros({3,0});   

    auto pts = at::zeros({3,comp[0].size(0)});

    size_t size = comp[0].size(0);
    size_t size2 = comp[1].size(0);
    
    //pts = pts.index({comp[1].slice(0,0,size2), comp[0].slice(0,0,size), heatmap.index({comp[0].slice(0,0,size), comp[1].slice(0,0,size2)})});
    for(int i = 0; i < size && i < size2; i++){
      pts[1][i] = comp[0][i];
      if(i < comp[1].size(0)){
        pts[0][i] = comp[1][i];
        pts[2][i] = heatmap[comp[0][i].item()][comp[1][i].item()];
      }
    }
    pts = nms_fast(pts, heatmap.size(0), heatmap.size(1), 3); // TODO -> Parametrize
    auto inds = pts[2].argsort(0, true);

    pts = pts.index({torch::indexing::Slice(0, 3, 1), inds.slice(0,0,pts.size(1))});


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
    device = torch::kCUDA;
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

    for (int i = 0; i < 10; i++){
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    std::vector<torch::jit::IValue> inputs;
    auto tens = matToTensor(img, device);

    inputs.push_back(tens);
    auto output = module.forward(inputs).toGenericDict();

    auto semi = output.at("semi").toTensor().to(torch::kCPU);
    auto coarse_desc = output.at("desc").toTensor().to(torch::kCPU);
    
    auto heatmap = flattenDetections(semi);
    auto pts = getPtsFromHeatmap(heatmap, CONF_THRESH);

/*
    for(int i = 0; i < pts.size(1); i++){
      int x = pts[0][i].item<int>();
      int y = pts[1][i].item<int>();
      cv::circle(img,cv::Point(x,y), 5, (0,0,255), -1);
    } 
  */

    auto dense_desc = torch::nn::functional::interpolate(coarse_desc, torch::nn::functional::InterpolateFuncOptions().scale_factor(std::vector<double>({8,8})).mode(torch::kBilinear).align_corners(false));
    auto dn = torch::norm(dense_desc, 2, 1);
    auto desc = dense_desc.div(torch::unsqueeze(dn, 1));

    int monoIndex = 0;
    auto desc_per = desc.permute({0,2,3,1});
    for(int i = 0; i < pts.size(1); i++) {
        //TODO -> might need to make this dynamic
        cv::KeyPoint keypoint = cv::KeyPoint(pts[0][i].item<int>(), pts[1][i].item<int>(),1); // TODO -> what is size param?
        //_keypoints.at(monoIndex) = (keypoint);
        //pts_desc = [dense_desc_cpu[i, :, pts[i][1,:].astype(int), pts[i][0, :].astype(int)].transpose() for i in range(len(pts))]
        auto tensor = desc_per[0][pts[1][i].item<int>()][pts[0][i].item<int>()].contiguous();
        std::vector<float> vecVal(tensor.data_ptr<float>(), tensor.data_ptr<float>()+tensor.numel());
         //std::cout << descriptors.size() << " " << descriptors.channels() << std::endl;
        //cv::InputArray(vecVal).copyTo(descriptors.row(monoIndex));

        //descriptors.row(monoIndex).setTo(cv::InputArray(vecVal));
        monoIndex++;
    }
        std::cout << "TIME: " <<  std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - t1).count() << std::endl;

    }
  //  cv::imshow("display window", img);
  //  cv::waitKey(0);


}
