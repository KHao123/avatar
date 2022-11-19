#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "Avatar.h"
#include "AvatarOptimizer.h"
#include "AvatarRenderer.h"
#include "BGSubtractor.h"
#include "Calibration.h"
#include "RTree.h"
#include "Util.h"
#include "smplx.hpp"
#include "util_smplx.hpp"
#define BEGIN_PROFILE auto start = std::chrono::high_resolution_clock::now()
#define PROFILE(x)                                                    \
    do {                                                              \
        printf("%s: %f ms\n", #x,                                     \
               std::chrono::duration<double, std::milli>(             \
                   std::chrono::high_resolution_clock::now() - start) \
                   .count());                                         \
        start = std::chrono::high_resolution_clock::now();            \
    } while (false)

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    std::string datasetPath;
    int bgId, imId, padSize, nnStep, interval;
    int frameICPIters, reinitICPIters, reinitCnz, itersPerICP;
    float betaPose, betaShape;
    std::string rtreePath;
    bool rtreeOnly, disableOcclusion;

    po::options_description desc("Option arguments");
    po::options_description descPositional(
        "OpenARK Avatar Offline Demo [previously bgsubtract] (c) Alex Yu "
        "2019\nPosition arguments");
    po::options_description descCombined("");
    desc.add_options()("help", "produce help message")(
        "background,b", po::value<int>(&bgId)->default_value(9999),
        "Background image id")(
        "image,i", po::value<int>(&imId)->default_value(1), "Current image id")(
        "pad,p", po::value<int>(&padSize)->default_value(4),
        "Zero pad width for image names in this dataset")(
        "rtree-only,R", po::bool_switch(&rtreeOnly),
        "Show RTree part segmentation only and skip optimization")(
        "no-occlusion", po::bool_switch(&disableOcclusion),
        "Disable occlusion detection in avatar optimizer prior to NN matching")(
        "betapose", po::value<float>(&betaPose)->default_value(0.05),
        "Optimization loss function: pose prior term weight")(
        "betashape", po::value<float>(&betaShape)->default_value(0.12),
        "Optimization loss function: shape prior term weight")(
        "data-interval,I", po::value<int>(&interval)->default_value(12),
        "Only computes rtree weights and optimizes for pixels with x = y = 0 "
        "mod interval")(
        "nnstep", po::value<int>(&nnStep)->default_value(20),
        "Optimization nearest-neighbor step: only matches neighbors every x "
        "points; a heuristic to improve speed (currently, not used)")(
        "frame-icp-iters,t", po::value<int>(&frameICPIters)->default_value(3),
        "ICP iterations per frame")(
        "reinit-icp-iters,T", po::value<int>(&reinitICPIters)->default_value(6),
        "ICP iterations when reinitializing (at beginning/after tracking "
        "loss)")("inner-iters,p",
                 po::value<int>(&itersPerICP)->default_value(10),
                 "Maximum inner iterations per ICP step")(
        "min-points,M", po::value<int>(&reinitCnz)->default_value(1000),
        "Minimum number of detected body points to allow continued tracking; "
        "if it falls below this number, then the tracker reinitializes");
    descPositional.add_options()(
        "dataset_path", po::value<std::string>(&datasetPath)->required(),
        "Input dataset root directory, should contain depth_exr etc")(
        "rtree", po::value<std::string>(&rtreePath), "RTree model path");
    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;

    po::positional_options_description posopt;
    posopt.add("dataset_path", 1);
    posopt.add("rtree", 1);
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(descCombined)
                      .positional(posopt)
                      .run(),
                  vm);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << descPositional << "\n" << desc << "\n";
        return 1;
    }

    if (vm.count("help")) {
        std::cout << descPositional << "\n" << desc << "\n";
        return 0;
    }

    try {
        po::notify(vm);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << descPositional << "\n" << desc << "\n";
        return 1;
    }

    using boost::filesystem::exists;
    using boost::filesystem::path;
    std::string intrinPath = (path(datasetPath) / "intrin.txt").string();
    ark::CameraIntrin intrin;
    if (intrinPath.size()) {
        intrin.readFile(intrinPath);
    }

    // if (rtreeOnly) interval = 1;

    std::stringstream ss_bg_id;
    ss_bg_id << std::setw(padSize) << std::setfill('0') << std::to_string(bgId);
    std::string bgPath =
        (path(datasetPath) / "depth_exr" / ("depth_" + ss_bg_id.str() + ".exr"))
            .string();
    cv::Mat background;
    ark::util::readXYZ(bgPath, background, intrin);
    if (background.empty()) {
        std::cerr << "ERROR: empty background image. Incorrect path/ID out of "
                     "bounds/pad size incorrect (specify -p)?\n";
        return 1;
    }

    ark::RTree rtree(rtreePath);

    ark::AvatarModel avaModel;
    // std::cout<<"#####2\n";
    ark::Avatar ava(avaModel);
    // std::cout<<"#####3\n";
    ark::AvatarOptimizer avaOpt(ava, intrin, background.size(), rtree.numParts,
                                rtree.partMap);
    // std::cout<<"#####4\n";
    // std::cout << ava.model.numJoints() << std::endl;
    // std::cout << ava.model.numPoints() << std::endl;
    // std::cout << ava.model.parent << std::endl;
    // std::cout << ava.model.numShapeKeys() << std::endl;
    // std::cout << ava.model.useJointShapeRegressor << std::endl;
    // std::cout << ava.model.baseCloud << std::endl;
    // std::cout << ava.model.mesh << std::endl;
    // std::cout << ava.model.jointRegressor << std::endl;
    // std::cout << ava.model.assignedJoints.size() << std::endl;
    // std::cout << ava.model.keyClouds << std::endl;
    // getchar();
    
    // smplx::ModelX model(
    //     smplx::util::parse_gender("MALE"));
    // smplx::BodyX body(model);

    avaOpt.betaPose = betaPose;
    avaOpt.betaShape = betaShape;
    avaOpt.nnStep = nnStep;
    avaOpt.enableOcclusion = !disableOcclusion;
    avaOpt.maxItersPerICP = itersPerICP;
    ark::BGSubtractor bgsub(background);
    bgsub.numThreads = std::thread::hardware_concurrency();
    std::vector<std::array<int, 2>> compsBySize;

    // Previous centers of mass: required by RTree postprocessor
    Eigen::Matrix<double, 2, Eigen::Dynamic> comPre;

    bool reinit = true;

    while (true) {
        // body.update();
        // std::cout << body.pose();
        std::stringstream ss_img_id;
        ss_img_id << std::setw(padSize) << std::setfill('0')
                  << std::to_string(imId);

        std::string inPath = (path(datasetPath) / "depth_exr" /
                              ("depth_" + ss_img_id.str() + ".exr"))
                                 .string();
        //+DEBUG
        std::cout << "\n> LOAD" << inPath << "\n";
        //-DEBUG

        cv::Mat image;
        ark::util::readXYZ(inPath, image, intrin);
        std::string inPathRGB =
            (path(datasetPath) / "rgb" / ("rgb_" + ss_img_id.str() + ".jpg"))
                .string();
        cv::Mat imageRGB = cv::imread(inPathRGB);
        if (image.empty() || imageRGB.empty()) {
            std::cerr << "WARNING: no more images found, exiting\n";
            break;
        }
        cv::Mat depth;
        cv::extractChannel(image, depth, 2);
        auto ccstart = std::chrono::high_resolution_clock::now();
        BEGIN_PROFILE;
        cv::Mat sub =
            bgsub.run(image, rtreePath.empty() ? &compsBySize : nullptr);
        PROFILE(BG Subtraction);

        cv::Mat vis(sub.size(), CV_8UC3);
        for (int r = bgsub.topLeft.y; r <= bgsub.botRight.y; ++r) {
            const auto* inptr = sub.ptr<uint8_t>(r);
            auto* dptr = depth.ptr<float>(r);
            for (int c = bgsub.topLeft.x; c <= bgsub.botRight.x; ++c) {
                if (inptr[c] >= 254) {
                    dptr[c] = 0.0f;
                }
            }
        }
        PROFILE(Apply foreground mask to depth);

        if (rtreePath.size()) {
            vis.setTo(cv::Vec3b(0, 0, 0));
            cv::Mat result =
                rtree.predictBest(depth, std::thread::hardware_concurrency(), 2,
                                  bgsub.topLeft, bgsub.botRight);
            PROFILE(RTree inference);
            rtree.postProcess(result, comPre, 2,
                              std::thread::hardware_concurrency(),
                              bgsub.topLeft, bgsub.botRight);
            PROFILE(RTree postproc);
            if (rtreeOnly) {
                for (int r = bgsub.topLeft.y; r <= bgsub.botRight.y; ++r) {
                    auto* inPtr = result.ptr<uint8_t>(r);
                    auto* visualPtr = vis.ptr<cv::Vec3b>(r);
                    for (int c = bgsub.topLeft.x; c <= bgsub.botRight.x; ++c) {
                        if (inPtr[c] == 255) continue;
                        visualPtr[c] = ark::util::paletteColor(inPtr[c], true);
                    }
                }
            } else {
                size_t cnz = 0;
                for (int r = bgsub.topLeft.y; r <= bgsub.botRight.y;
                     r += interval) {
                    auto* partptr = result.ptr<uint8_t>(r);
                    for (int c = bgsub.topLeft.x; c <= bgsub.botRight.x;
                         c += interval) {
                        if (partptr[c] == 255) continue;
                        ++cnz;
                    }
                }
                if (cnz >= reinitCnz / (interval * interval)) {
                    ark::CloudType dataCloud(3, cnz);
                    Eigen::VectorXi dataPartLabels(cnz);
                    size_t i = 0;
                    for (int r = bgsub.topLeft.y; r <= bgsub.botRight.y;
                         r += interval) {
                        auto* ptr = image.ptr<cv::Vec3f>(r);
                        auto* partptr = result.ptr<uint8_t>(r);
                        for (int c = bgsub.topLeft.x; c <= bgsub.botRight.x;
                             c += interval) {
                            if (partptr[c] == 255) continue;
                            if (partptr[c] >= rtree.numParts) {
                                std::cerr
                                    << "FATAL: RTree body part prediction "
                                    << (int)partptr[c]
                                    << " is invalid, since there are only "
                                    << rtree.numParts << " body parts\n";
                                std::exit(1);
                            }
                            dataCloud(0, i) = ptr[c][0];
                            dataCloud(1, i) = -ptr[c][1];
                            dataCloud(2, i) = ptr[c][2];
                            dataPartLabels(i) = partptr[c];
                            ++i;
                        }
                    }
                    int icpIters = frameICPIters;
                    if (reinit) {
                        Eigen::Vector3d cloudCen = dataCloud.rowwise().mean();
                        ava.p = cloudCen;
                        ava.w.setZero();
                        for (int i = 1; i < ava.model.numJoints(); ++i) {
                            ava.r[i].setIdentity();
                        }
                        ava.r[0] =
                            Eigen::AngleAxisd(M_PI, Eigen::Vector3d(0, 1, 0))
                                .toRotationMatrix();
                        reinit = false;
                        ava.update();
                        icpIters = reinitICPIters;
                        PROFILE(Prepare reinit);
                    }
                    avaOpt.optimize(dataCloud, dataPartLabels, icpIters,
                                    std::thread::hardware_concurrency());
                    PROFILE(Optimize(Total));
                    printf(
                        "Overall (excluding visualization): %f ms\n",
                        std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - ccstart)
                            .count());
                    ark::AvatarRenderer rend(ava, intrin);
                    // Draw avatar onto RGB using lambertian shading
                    cv::Mat modelMap = rend.renderLambert(depth.size());
                    for (int r = 0; r < vis.rows; ++r) {
                        auto* outptr = vis.ptr<cv::Vec3b>(r);
                        const auto* renderptr = modelMap.ptr<uint8_t>(r);
                        for (int c = 0; c < vis.cols; ++c) {
                            if (renderptr[c] > 0) {
                                outptr[c][0] = outptr[c][1] = outptr[c][2] =
                                    renderptr[c];
                            }
                        }
                    }
                    printf(
                        "Overall: %f ms\n",
                        std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - ccstart)
                            .count());
                }
            }
            for (int r = 0; r < image.rows; ++r) {
                auto* outptr = vis.ptr<cv::Vec3b>(r);
                const auto* rgbptr = imageRGB.ptr<cv::Vec3b>(r);
                for (int c = 0; c < image.cols; ++c) {
                    if (outptr[c][0] == 0 && outptr[c][1] == 0 &&
                        outptr[c][2] == 0)
                        outptr[c] = rgbptr[c];
                    else {
                        // Blend
                        outptr[c] = rgbptr[c] / 5 * 2 + outptr[c] / 5 * 3;
                    }
                }
            }
        } else {
            std::vector<int> colorid(256, 255);
            for (int r = 0; r < compsBySize.size(); ++r) {
                colorid[compsBySize[r][1]] = r;  // > 0 ? 255 : 0;
            }
            for (int r = 0; r < image.rows; ++r) {
                auto* outptr = vis.ptr<cv::Vec3b>(r);
                const auto* inptr = sub.ptr<uint8_t>(r);
                for (int c = 0; c < image.cols; ++c) {
                    int colorIdx = colorid[inptr[c]];
                    if (colorIdx >= 254) {
                        outptr[c] = 0;
                    } else {
                        outptr[c] = ark::util::paletteColor(colorIdx, true);
                    }
                }
            }
        }
        // cv::rectangle(vis, bgsub.topLeft, bgsub.botRight,
        // cv::Scalar(0,0,255));

        cv::imshow("Visual", vis);
        // cv::imshow("Depth", depth);
        ++imId;
        int k = cv::waitKey(1);
        if (k == 'q') break;
    }  // while(true)
    return 0;
}
