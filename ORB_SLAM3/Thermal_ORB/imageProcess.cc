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

#define CONF_THRESH 0.002

at::Tensor nms_fast(at::Tensor in_corners, int h, int w, int dist_thresh) {
    auto grid = at::zeros({h,w}).to(torch::kInt);
    auto inds = at::zeros({h,w}).to(torch::kInt);

    auto inds1 = (-1*in_corners[2]).argsort(0);
    auto corners = at::zeros((in_corners.sizes()));

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


    int pad = 2;//dist_thresh;

    //grid = torch::nn::functional::pad(grid, torch::nn::functional::PadFuncOptions({pad}).mode(torch::kConstant));
    for(int i = 2; i < rcornersT.size(0) - 2; i++){
      int pt[2] = {rcornersT[i][0].item<int>() + pad, rcornersT[i][1].item<int>() + pad};
      if ( pt[1] < 235 && pt[0] < 315 && grid[pt[1]][pt[0]].item<float>() == 1){
          grid[pt[1] - pad][pt[0] - pad] = 0;
          grid[pt[1]][pt[0] - pad] = 0;
          grid[pt[1] - pad][pt[0]] = 0;
          grid[pt[1] + pad][pt[0] + pad] = 0;
          grid[pt[1] + pad][pt[0] + pad] = 0;

          grid[pt[1]][pt[0]] = -1;
      } 
    }

    auto keep = at::where(grid  == -1);
  
    auto inds_keep = at::zeros(keep[0].size(0));
    
    for(int i = 0; i < inds.size(0); i++) {    
      std::cout <<keep[1][i].item<int>()<< " " <<keep[0][i].item<int>()<< std::endl;
      inds_keep[i] = inds[keep[0][i].item<int>()][keep[1][i].item<int>()];
    }

    auto out = at::zeros({3,keep[0].size(0)});
    for(int i = 0; i < inds_keep.size(0); i++) {
        std::cout << inds_keep.sizes() << inds_keep[i].item<int>() << corners.sizes() << std::endl;
        out[0][i] = corners[0][inds_keep[i].item<int>()];
        out[1][i] = corners[1][inds_keep[i].item<int>()];
        out[2][i] = corners[2][inds_keep[i].item<int>()];

    }
    auto values = out[2];
    auto inds2 = (-1* values).argsort();
    for(int i = 0; i < inds2.size(0); i++) {
        out[0][i] = out[0][inds2[i].item<int>()];
        out[1][i] = out[1][inds2[i].item<int>()];
        out[2][i] = out[2][inds2[i].item<int>()];
    }

    return out;
}

/*
    def nms_fast(self, in_corners, H, W, dist_thresh):

        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds
        */


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

    pts = nms_fast(pts, heatmap.size(0), heatmap.size(1), 6); // TODO -> Parametrize
    auto inds = pts[2].argsort(0);
    for(int i = 0; i < pts.size(0); i++) {
      for (int j = pts.size(1) - 1 ; j >= 0; j--) {
        pts[i][j] = pts[i][inds[j]];
      }
    }  
    std::cout << "NUMBER OF POINTS:" << pts.sizes() << std::endl;


    /*
FUNCTIONS

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

  cv::Mat img = cv::imread("/Users/mattsaraceno/Desktop/EECS568/FINAL_RESOURCES/ThermalORB/ORB_SLAM3/Thermal_ORB/test.png");
  if(img.empty())
  {
    std::cout << "Could not read the image: " << std::endl;
    return 1;
  }
  for(int k = 0; k < 10; k++ ){
  std::vector<torch::jit::IValue> inputs;
  auto tens = matToTensor(img, device);

  inputs.push_back(tens);
  auto output = module.forward(inputs).toGenericDict();

  auto semi = output.at("semi").toTensor();
  auto descriptors = output.at("desc").toTensor();

  auto heatmap = flattenDetections(semi);
  auto pts = getPtsFromHeatmap(heatmap.to(torch::kCPU), CONF_THRESH).to(torch::kCPU);


  for(int i = 0; i < pts.size(1); i++){
    int x = pts[0][i].item<int>();
    int y = pts[1][i].item<int>();
    cv::circle(img,cv::Point(x,y), 5, (0,0,255), -1);
  } 


  cv::imshow("Display window", img);
  int j = cv::waitKey(0); // Wait for a keystroke in the window
  }
  /*

    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]
    desc = desc.to(torch::kCPU);

    dense_desc = nn.functional.interpolate(coarse_desc, scale_factor=(self.cell, self.cell), mode='bilinear')
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
