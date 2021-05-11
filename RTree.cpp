#include "RTree.h"

#include <fstream>
#include <chrono>
#include <cstdio>
#include <csignal>
#include <random>
#include <atomic>
#include <deque>
#include <mutex>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <boost/filesystem.hpp>
#include <Eigen/StdVector>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Util.h"
#include "AvatarRenderer.h"
#include "SparseImage.h"

#define BEGIN_PROFILE auto _start = std::chrono::high_resolution_clock::now()
#define PROFILE(x) do{printf("* P %s: %f ms\n", #x, std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - _start).count()); _start = std::chrono::high_resolution_clock::now(); }while(false)

namespace {
    // Compute Shannon entropy of a distribution (must be normalized)
    template<class Distribution>
    inline float entropy(const Distribution & distr) {
        // Can this be vectorized?
        float entropy = 0.f;
        for (int i = 0; i < distr.size(); ++i) {
            if (distr[i] < 1e-10) continue;
            entropy -= distr[i] * std::log2(distr[i]);
        }
        return entropy;
    }

    // Get depth at point in depth image, or return BACKGROUND_DEPTH
    // if in the background OR out of bounds
    template<class Image>
    inline float getDepth(const Image& depth_image, const Eigen::Vector2i& point) {
        if (point.y() < 0 || point.x() < 0 ||
                point.y() >= depth_image.rows || point.x() >= depth_image.cols)
            return ark::RTree::BACKGROUND_DEPTH;
        float depth = depth_image.template at<float>(point.y(), point.x());
        if (depth == 0.0) return ark::RTree::BACKGROUND_DEPTH;
        return depth;
    }

    /** Get score of single sample given by a feature */
    template <class Image>
    inline float scoreByFeature(const Image& depth_image,
            const ark::RTree::Vec2i& pix,
            const ark::RTree::Vec2& u,
            const ark::RTree::Vec2& v) {
            float sampleDepth = depth_image.template at<float>(pix.y(), pix.x());
            // Add feature u,v and round
            Eigen::Vector2f ut = u / sampleDepth, vt = v / sampleDepth;
            Eigen::Vector2i uti, vti;
            uti[0] = static_cast<int32_t>(std::round(ut.x()));
            uti[1] = static_cast<int32_t>(std::round(ut.y()));
            vti[0] = static_cast<int32_t>(std::round(vt.x()));
            vti[1] = static_cast<int32_t>(std::round(vt.y()));
            uti += pix.cast<int32_t>(); vti += pix.cast<int32_t>();

            return (getDepth(depth_image, uti) - getDepth(depth_image, vti));
    }

    void upscaleGrid(cv::Mat& image, int interval, int num_threads,
            const cv::Point& top_left, const cv::Point& bot_right) {
        {
            std::atomic<int> row(top_left.y);
            auto worker = [&] () {
                int rr = 0;
                while (true) {
                    rr = (row += interval);
                    if (rr > bot_right.y) break;

                    uint8_t* ptrRef = image.ptr<uint8_t>(rr);
                    for (int r = rr ; r < rr + interval; ++r) {
                        if (r > bot_right.y) break;
                        uint8_t* ptr = image.ptr<uint8_t>(r);
                        for (int cc = top_left.x ; cc <= bot_right.x; cc += interval) {
                            uint8_t val = ptrRef[cc];
                            memset(ptr + cc, val, interval);
                        }
                    }
                }
            };
            std::vector<std::thread> thds;
            for (int i = 0; i < num_threads; ++i) {
                thds.emplace_back(worker);
            }
            for (int i = 0; i < num_threads; ++i) {
                thds[i].join();
            }
        }
    }

    void majorityGrid(cv::Mat& image, int interval, int num_parts) {
        Eigen::VectorXi cnt(num_parts + 1);
        for (int rr = 0 ; rr < image.rows; rr += interval) {
            for (int cc = 0 ; cc < image.cols; cc += interval) {
                cnt.setZero();
                for (int r = rr ; r < rr + interval; ++r) {
                    uint8_t* ptr = image.ptr<uint8_t>(r);
                    for (int c = cc ; c < cc + interval; ++c) {
                        if (ptr[c] == 255) ++cnt[num_parts];
                        else ++cnt[ptr[c]];
                    }
                }
                int argmax;
                cnt.maxCoeff(&argmax);
                if (argmax == num_parts) argmax = 255;

                uint8_t bestPartId = argmax;
                for (int r = rr ; r < rr + interval; ++r) {
                    uint8_t* ptr = image.ptr<uint8_t>(r);
                    memset(ptr + cc, bestPartId, interval);
                }
            }
        }
    }

    void suppressPartNonMax(cv::Mat& image, int interval, int num_parts, int num_threads, const cv::Point& top_left, const cv::Point& bot_right,
            Eigen::Matrix<double, 2, Eigen::Dynamic>& com_pre,
            double dist_to_pre_weight) {
        const int VISITED_OFFSET = 128;

        std::vector<int> stk, curCompVis;
        std::vector<std::vector<int> > bestComp(num_parts);

        Eigen::VectorXd bestScore(num_parts);
        bestScore.setZero();

        Eigen::Matrix<double, 2, Eigen::Dynamic> comBest(2, num_parts);
        comBest.setZero();

        Eigen::Vector2d com(2);

        stk.reserve((bot_right.x - top_left.x + 1) *
                (bot_right.y - top_left.y + 1) / interval / interval);
        curCompVis.reserve(stk.capacity());
        int hi_bit = (1<<16);
        int lo_mask = hi_bit - 1;
        hi_bit *= interval;

        auto maybe_visit = [&](int curr_r, int curr_c, uint8_t curr_val, int new_r, int new_c, int new_id) {
            uint8_t& val = image.at<uint8_t>(new_r, new_c);
            if (curr_val == val) {
                val += VISITED_OFFSET;
                curCompVis.push_back(new_id);
                stk.push_back(new_id);
            }
        };

        for (int rr = top_left.y ; rr <= bot_right.y; rr += interval) {
            uint8_t* ptr = image.ptr<uint8_t>(rr);
            for (int cc = top_left.x; cc <= bot_right.x; cc += interval) {
                uint8_t val = ptr[cc];
                if (val >= VISITED_OFFSET) continue;

                ptr[cc] += VISITED_OFFSET;
                stk.push_back((rr << 16) + cc);
                curCompVis.clear();
                curCompVis.push_back(stk.back());
                com.setZero();
                Eigen::Vector2d pt;
                bool hasPrevCom = com_pre(0, val) >= 0.;
                while (stk.size()) {
                    int id = stk.back();
                    int cur_c = (id & lo_mask), cur_r = (id >> 16);
                    stk.pop_back();
                    if (cur_r >= top_left.y + interval) maybe_visit(cur_r, cur_c, val, cur_r - interval, cur_c, id - hi_bit);
                    if (cur_r <= bot_right.y- interval) maybe_visit(cur_r, cur_c, val, cur_r + 1, cur_c, id + hi_bit);
                    if (cur_c >= top_left.x + interval) maybe_visit(cur_r, cur_c, val, cur_r, cur_c - interval, id - interval);
                    if (cur_c <= bot_right.x - interval) maybe_visit(cur_r, cur_c, val, cur_r, cur_c + interval, id + interval);
                    pt << cur_c, cur_r;
                    com += pt;
                }
                double score = curCompVis.size();
                com /= curCompVis.size();
                if (hasPrevCom) {
                    score -= (com - com_pre.col(val)).squaredNorm() * dist_to_pre_weight;
                }
                if (score > bestScore(val)) {
                    bestScore[val] = score;
                    comBest.col(val) = com;
                    for (int id : bestComp[val]) {
                        const int cr = (id >> 16), cc = (id & lo_mask);
                        image.at<uint8_t>(cr, cc) = 255;
                    }
                    bestComp[val].swap(curCompVis);
                } else {
                    for (int id : curCompVis) {
                        const int cr = (id >> 16), cc = (id & lo_mask);
                        image.at<uint8_t>(cr, cc) = 255;
                    }
                }
            }
        }

        for (int i = 0; i < num_parts; ++i) {
            if (bestComp[i].empty()) {
                com_pre(0, i) = -1.;
            } else {
                com_pre.col(i) = comBest.col(i);
            }
        }
        // for (int rr = 0 ; rr < image.rows; ++rr) {
        //     uint8_t* ptr = image.ptr<uint8_t>(rr);
        //     for (int cc = 0 ; cc < image.cols; ++cc) {
        //         uint8_t& val = ptr[cc];
        //         if (val >= VISITED_OFFSET && val != 255)
        //             val -= VISITED_OFFSET;
        //     }
        // }
        //
        {
            std::atomic<int> row(top_left.y);
            auto worker = [&] () {
                int r = 0;
                while (true) {
                    r = row++;
                    if (r > bot_right.y) break;
                    uint8_t* ptr = image.ptr<uint8_t>(r);
                    for (int cc = top_left.x; cc <= bot_right.x; ++cc) {
                        uint8_t& val = ptr[cc];
                        if (val >= VISITED_OFFSET && val != 255)
                            val -= VISITED_OFFSET;
                    }
                }
            };
            std::vector<std::thread> thds;
            for (int i = 0; i < num_threads; ++i) {
                thds.emplace_back(worker);
            }
            for (int i = 0; i < num_threads; ++i) {
                thds[i].join();
            }
        }
    }

    void removeSmallPieces(cv::Mat& image, int interval, int num_parts, int num_threads, const cv::Point& top_left, const cv::Point& bot_right,
            double thresh = 0.0005) {
        const int VISITED_OFFSET = 128;

        std::vector<int> stk, curCompVis;

        size_t scaledThresh = image.rows * image.cols / (interval * interval) * thresh;

        stk.reserve((bot_right.x - top_left.x + 1) *
                (bot_right.y - top_left.y + 1) / interval / interval);
        curCompVis.reserve(stk.capacity());
        int hi_bit = (1<<16);
        int lo_mask = hi_bit - 1;
        hi_bit *= interval;

        auto maybe_visit = [&](int curr_r, int curr_c, uint8_t curr_val, int new_r, int new_c, int new_id) {
            uint8_t& val = image.at<uint8_t>(new_r, new_c);
            if (curr_val == val) {
                val += VISITED_OFFSET;
                curCompVis.push_back(new_id);
                stk.push_back(new_id);
            }
        };

        for (int rr = top_left.y ; rr <= bot_right.y; rr += interval) {
            uint8_t* ptr = image.ptr<uint8_t>(rr);
            for (int cc = top_left.x; cc <= bot_right.x; cc += interval) {
                uint8_t val = ptr[cc];
                if (val >= VISITED_OFFSET) continue;

                ptr[cc] += VISITED_OFFSET;
                stk.push_back((rr << 16) + cc);
                curCompVis.clear();
                curCompVis.push_back(stk.back());
                Eigen::Vector2d pt;
                while (stk.size()) {
                    int id = stk.back();
                    int cur_c = (id & lo_mask), cur_r = (id >> 16);
                    stk.pop_back();
                    if (cur_r >= top_left.y + interval) maybe_visit(cur_r, cur_c, val, cur_r - interval, cur_c, id - hi_bit);
                    if (cur_r <= bot_right.y- interval) maybe_visit(cur_r, cur_c, val, cur_r + 1, cur_c, id + hi_bit);
                    if (cur_c >= top_left.x + interval) maybe_visit(cur_r, cur_c, val, cur_r, cur_c - interval, id - interval);
                    if (cur_c <= bot_right.x - interval) maybe_visit(cur_r, cur_c, val, cur_r, cur_c + interval, id + interval);
                    pt << cur_c, cur_r;
                }
                if (curCompVis.size() < scaledThresh) {
                    for (int id : curCompVis) {
                        const int cr = (id >> 16), cc = (id & lo_mask);
                        image.at<uint8_t>(cr, cc) = 255;
                    }
                }
            }
        }
        {
            std::atomic<int> row(top_left.y);
            auto worker = [&] () {
                int r = 0;
                while (true) {
                    r = row++;
                    if (r > bot_right.y) break;
                    uint8_t* ptr = image.ptr<uint8_t>(r);
                    for (int cc = top_left.x; cc <= bot_right.x; ++cc) {
                        uint8_t& val = ptr[cc];
                        if (val >= VISITED_OFFSET && val != 255)
                            val -= VISITED_OFFSET;
                    }
                }
            };
            std::vector<std::thread> thds;
            for (int i = 0; i < num_threads; ++i) {
                thds.emplace_back(worker);
            }
            for (int i = 0; i < num_threads; ++i) {
                thds[i].join();
            }
        }
    }
}

namespace ark {
    const float RTree::BACKGROUND_DEPTH = 20.f;

    RTree::RNode::RNode() : leafid(-1), lnode(-1), rnode(-1) {};
    RTree::RNode::RNode(const Vec2& u, const Vec2& v, float thresh) :
                u(u), v(v), thresh(thresh), leafid(-1) {}

    enum {
        DATA_DEPTH,
        DATA_PART_MASK,
        _DATA_TYPE_COUNT
    };
    const int IMREAD_FLAGS[2] = { cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH, cv::IMREAD_GRAYSCALE };

    struct Sample {
        Sample () {}
        Sample(int index, const RTree::Vec2i& pix) : index(index), pix(pix) {};
        Sample(int index, int r, int c) : index(index), pix(c, r) {};

        // Image index
        int index;
        // Pixel position
        RTree::Vec2i pix;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
    using SampleVec = std::vector<Sample, Eigen::aligned_allocator<Sample> >;

    struct FileDataSource {
        FileDataSource(
                const std::string& depth_dir,
                const std::string& part_mask_dir)
       : depthDir(depth_dir), partMaskDir(part_mask_dir) {
           reload();
       }

        void reload() {
            _data_paths[DATA_DEPTH].clear();
            _data_paths[DATA_PART_MASK].clear();
            using boost::filesystem::directory_iterator;
            // List directories
            for (auto it = directory_iterator(depthDir); it != directory_iterator(); ++it) {
                _data_paths[DATA_DEPTH].push_back(it->path().string());
            }
            std::sort(_data_paths[DATA_DEPTH].begin(), _data_paths[DATA_DEPTH].end());
            for (auto it = directory_iterator(partMaskDir); it != directory_iterator(); ++it) {
                _data_paths[DATA_PART_MASK].push_back(it->path().string());
            }
            std::sort(_data_paths[DATA_PART_MASK].begin(), _data_paths[DATA_PART_MASK].end());
        }

        int size() const {
            return _data_paths[0].size();
        }

        const std::array<cv::Mat, 2>& load(int idx, int hint = -1) {
            thread_local std::array<cv::Mat, 2> arr;
            thread_local int last_idx = -1, last_hint = -1;
            if (idx != last_idx || hint != last_hint) {
                if (hint == 0 || hint == -1)
                    arr[0] = cv::imread(_data_paths[idx][0], IMREAD_FLAGS[0]);
                if (hint == 1 || hint == -1)
                    arr[1] = cv::imread(_data_paths[idx][1], IMREAD_FLAGS[1]);
                last_idx = idx;
                last_hint = hint;
            }
            return arr;
        }

        void serialize(std::ostream& os) {
            os.write("SRC_FILE", 8);
            util::write_bin(os, depthDir.size());
            os.write(depthDir.c_str(), depthDir.size());
            util::write_bin(os, partMaskDir.size());
            os.write(depthDir.c_str(), depthDir.size());
        }

        void deserialize(std::istream& is) {
            char marker[8];
            is.read(marker, 8);
            if (strncmp(marker, "SRC_FILE", 8)) {
                std::cerr << "ERROR: Invalid file data source specified in stored samples file\n";
                return;
            }
            size_t sz;
            util::read_bin(is, sz);
            depthDir.resize(sz);
            is.read(&depthDir[0], sz);
            util::read_bin(is, sz);
            partMaskDir.resize(sz);
            is.read(&partMaskDir[1], sz);
            reload();
        }

        std::vector<std::string> _data_paths[2];
        std::string depthDir, partMaskDir;
    };

    struct AvatarDataSource {
        AvatarDataSource(AvatarModel& ava_model,
                         AvatarPoseSequence& pose_seq,
                         CameraIntrin& intrin,
                         cv::Size& image_size,
                         int num_images,
                         const std::vector<int>& part_map)
            : numImages(num_images), imageSize(image_size),
              avaModel(ava_model), poseSequence(pose_seq), intrin(intrin),
              partMap(part_map) {
            seq.reserve(poseSequence.numFrames);
            const size_t INT32_MAXVAL = static_cast<size_t>(std::numeric_limits<int>::max());
            if (poseSequence.numFrames > INT32_MAXVAL) {
                std::cerr << "WARNING: Truncated pose sequence of length " <<
                    poseSequence.numFrames << " > 2^31-1 to int32 range, to prevent overflow. "
                    "May need to switch sequence type from int "
                    "to int64_t in RTree.cpp AvatarDataSource (didn't do this since wastes memory)\n";
            }
            for (size_t i = 0; i < std::min(INT32_MAXVAL, poseSequence.numFrames); ++i) {
                seq.push_back(static_cast<int>(i));
            }
            for (size_t i = 0; i < std::min(INT32_MAXVAL, poseSequence.numFrames); ++i) {
                size_t r = random_util::randint<size_t>(i, poseSequence.numFrames - 1);
                if (r != i) std::swap(seq[r], seq[i]);
            }
            seq.resize(num_images);
            xorKey = random_util::randint<uint32_t>(1, std::numeric_limits<uint32_t>::max());
        }

        int size() const {
            return numImages;
        }

        /** Implements DataSource interface */
        const std::array<cv::Mat, 2>& load(int idx, int hint = -1) {
            thread_local std::array<cv::Mat, 2> arr;
            thread_local int last_idx = -1, last_hint = -1;
            thread_local Avatar ava(avaModel);
            if (idx != last_idx || hint != last_hint) {
                last_idx = idx;
                last_hint = hint;
                if (poseSequence.numFrames) {
                    // random_util::randint<size_t>(0, poseSequence.numFrames - 1)
                    int seqid = seq[idx % seq.size()];
                    poseSequence.poseAvatar(ava, seqid);
                    ava.r[0].setIdentity();
                    ava.randomize(false, true, true, static_cast<uint32_t>(idx) ^ xorKey);
                } else {
                    ava.randomize(true, true, true, static_cast<uint32_t>(idx) ^ xorKey);
                }
                ava.update();
                AvatarRenderer renderer(ava, intrin);

                if (hint == 0 || hint == -1)
                    arr[0] = renderer.renderDepth(imageSize);
                if (hint == 1 || hint == -1)
                    arr[1] = renderer.renderPartMask(imageSize, partMap);
            }
            return arr;
        }

        /** Simple load: load the depth and part mask for image at idx */
        void loadSimple(int idx, cv::Mat& depth, cv::Mat& part_mask, bool skip_part_mask = false) {
            thread_local Avatar ava(avaModel);
            if (poseSequence.numFrames) {
                // random_util::randint<size_t>(0, poseSequence.numFrames - 1)
                int seqid = seq[idx % seq.size()];
                poseSequence.poseAvatar(ava, seqid);
                ava.r[0].setIdentity();
                ava.randomize(false, true, true, static_cast<uint32_t>(idx) ^ xorKey);
            } else {
                ava.randomize(true, true, true, static_cast<uint32_t>(idx) ^ xorKey);
            }
            ava.update();
            AvatarRenderer renderer(ava, intrin);
            depth = renderer.renderDepth(imageSize);
            if (!skip_part_mask)
                part_mask = renderer.renderPartMask(imageSize, partMap);
        }

        // Warning: serialization is incomplete, still need to load same avatar model, pose sequence, etc.
        void serialize(std::ostream& os) {
            os.write("SRC_AVATAR", 10);
            util::write_bin<size_t>(os, std::numeric_limits<size_t>::max());
            util::write_bin<uint32_t>(os, xorKey);
            util::write_bin(os, seq.size());
            for (int i : seq) {
                util::write_bin(os, i);
            }
        }

        void deserialize(std::istream& is) {
            char marker[10];
            is.read(marker, 10);
            if (strncmp(marker, "SRC_AVATAR", 10)) {
                std::cerr << "ERROR: Invalid avatar data source specified in stored samples file\n";
                return;
            }
            size_t sz;
            util::read_bin<size_t>(is, sz);
            if (sz == std::numeric_limits<size_t>::max()) {
                // New format with xor key
                util::read_bin<uint32_t>(is, xorKey);
                util::read_bin(is, sz);
            } else {
                // For compatibility
                xorKey = 0;
            }

            seq.clear();
            seq.reserve(sz);
            int x;
            for (size_t i = 0; i < sz; ++i) {
                util::read_bin(is, x);
                seq.push_back(x);
            }
            if (seq.size() > numImages) {
                seq.resize(numImages);
            }
        }

        int numImages; uint32_t xorKey;
        cv::Size imageSize;
        AvatarModel& avaModel;
        AvatarPoseSequence& poseSequence;
        CameraIntrin& intrin;
        std::vector<int> seq;
        const std::vector<int>& partMap;
    };

    template<class DataSource>
    /** Responsible for handling data loading from abstract data source
     *  Interface: dataLoader.get(sample): get (depth, part mask) images for a sample
     *             dataLoader.preload(samples, a, b, numThreads): preload images for samples if possible */
    struct DataLoader {
        DataSource& dataSource;
        DataLoader(DataSource& data_source, size_t max_images_loaded) : dataSource(data_source), maxImagesLoaded(max_images_loaded) {}

        /** Precondition: samples must be sorted by image index on [a, ..., b-1] */
        bool preload(const SampleVec& samples, int a, int b, int numThreads) {
            std::vector<int> images;
            images.push_back(samples[a].index);
            images.reserve((b-a - 1) / 2000 + 1); // HACK
            size_t loadSize = data.size();

            imageIdx.resize(dataSource.size(), -1);
            for (int i = a + 1; i < b; ++i) {
                if (samples[i].index != images.back()) {
                    images.push_back(samples[i].index);
                    if (imageIdx[samples[i].index] == -1) {
                        ++loadSize;
                    }
                    // if (samples[i].index < images.back()) {
                    //     std::cerr << "FATAL: Images not sorted at preload " <<a << "," << b << "\n";
                    //     std::exit(0);
                    // }
                }
            }
            if (loadSize - data.size() == 0) return true; // Already all loaded
            if (images.size() > maxImagesLoaded) {
                std::cout << "INFO: Number of images too large (" << images.size() <<
                    " > " << maxImagesLoaded << "), not preloading\n";
                return false;
            }

            if (loadSize > maxImagesLoaded) {
                clear();
            } else {
                std::vector<int> newImages;
                newImages.reserve(loadSize);
                for (int im : images) {
                    if (imageIdx[im] == -1) {
                        newImages.push_back(im);
                    }
                }
                images.swap(newImages);
            }
            if (images.empty()) return true;

            std::vector<std::thread> threads;
            std::mutex mtx;
            std::atomic<size_t> i(0);
            size_t basei = data.size();
            data.resize(basei + images.size());
            revImageIdx.resize(basei + images.size());;

            auto worker = [&]() {
                size_t thread_i;
                while (true) {
                    thread_i = i++;
                    if (thread_i >= images.size()) break;
                    data[basei + thread_i] = dataSource.load(images[thread_i]);
                    imageIdx[images[thread_i]] = basei + thread_i;
                    revImageIdx[basei + thread_i] = images[thread_i];
                }
            };
            for (int i = 0; i < numThreads; ++i) {
                threads.emplace_back(worker);
            }
            for (int i = 0; i < numThreads; ++i) {
                threads[i].join();
            }
            return true;
        }

        const std::array<cv::Mat, 2>& get(const Sample& sample, int hint = -1) const {
            int iidx = sample.index >= static_cast<int>(imageIdx.size()) ?
                         -1 : imageIdx[sample.index];
            if (iidx < 0) {
                return dataSource.load(sample.index, hint);
            }
            return data[iidx];
        }

        void clear() {
            data.clear();
            for (int i : revImageIdx) {
                imageIdx[i] = -1;
            }
            revImageIdx.clear();
        }

        std::vector<std::array<cv::Mat, 2> > data;
        std::vector<int> imageIdx, revImageIdx;
        size_t maxImagesLoaded;
    };

    /** Internal trainer implementation */
    template<class DataSource>
    class Trainer {
    public:
        struct Feature {
            RTree::Vec2 u, v;
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        };

        Trainer() =delete;
        Trainer(const Trainer&) =delete;
        Trainer(Trainer&&) =delete;

        Trainer(std::vector<RTree::RNode, Eigen::aligned_allocator<RTree::RNode> >& nodes,
                std::vector<RTree::Distribution>& leaf_data,
                DataSource& data_source,
                int num_parts,
                size_t max_images_loaded)
            : nodes(nodes), leafData(leaf_data), numParts(num_parts),
              dataLoader(data_source, max_images_loaded) {
        }

        void train(int num_images, int num_points_per_image, int num_features,
                   int max_probe_offset, int min_samples, int max_tree_depth,
                   int samples_per_feature, int threshes_per_feature, int num_threads,
                   bool verbose, bool skip_init = false, bool skip_train = false) {
            this->verbose = verbose;
            numThreads = num_threads;

            if (!skip_init) {
                // Initialize new samples
                initTraining(num_images, num_points_per_image, max_tree_depth);
            }

            if (skip_train) return;

            if (verbose) {
                std::cerr << "Init RTree training with maximum depth " << max_tree_depth << "\n";
            }

            // Train
            numFeatures = num_features;
            maxProbeOffset = max_probe_offset;
            minSamples = min_samples;
            numImages = num_images;
            samplesPerFeature = samples_per_feature;
            threshesPerFeature = threshes_per_feature;
            nodes.resize(1);
            trainFromNode(nodes[0], 0, static_cast<int>(samples.size()), max_tree_depth);
            if (verbose) {
                std::cerr << "Training finished\n";
            }
        }

        void writeSamples(const std::string & path) {
            std::ofstream ofs(path, std::ios::out | std::ios::binary);
            dataLoader.dataSource.serialize(ofs);
            reorderByImage(samples, 0, samples.size());
            ofs.write("S\n", 2);
            size_t last_idx = 0;
            util::write_bin(ofs, samples.size());
            for (size_t i = 0; i <= samples.size(); ++i) {
                if (i == samples.size() ||
                    samples[i].index != samples[last_idx].index) {
                    util::write_bin(ofs, samples[last_idx].index);
                    util::write_bin(ofs, int(i - last_idx));
                    for (size_t j = last_idx; j < i; ++j) {
                        util::write_bin(ofs, samples[j].pix[0]);
                        util::write_bin(ofs, samples[j].pix[1]);
                    }
                    last_idx = i;
                }
            }
            util::write_bin(ofs, -1);
            util::write_bin(ofs, -1);
            ofs.close();
        }

        void readSamples(const std::string & path, bool verbose = false, int max_num_images = -1) {
            std::ifstream ifs(path, std::ios::in | std::ios::binary);
            if (verbose) {
                std::cout << "Recovering data source from samples file\n";
            }
            dataLoader.dataSource.deserialize(ifs);
            if (verbose) {
                std::cout << "Reading samples from samples file\n";
            }
            char marker[2];
            ifs.read(marker, 2);
            if (strncmp(marker, "S\n", 2)) {
                std::cerr << "ERROR: Invalid or corrupted samples file at " << path << "\n";
                return;
            }
            size_t numSamplesTotal;
            util::read_bin(ifs, numSamplesTotal);
            samples.reserve(numSamplesTotal);
            while (ifs) {
                int imgIndex, imgSamps;
                util::read_bin(ifs, imgIndex);
                if (~max_num_images && imgIndex >= max_num_images) {
                    std::cerr << "Image index " << imgIndex << " out of bounds, invalid samples file?\n";
                    std::exit(0);
                }
                util::read_bin(ifs, imgSamps);
                if (verbose && imgIndex % 1000 == 0 && imgIndex >= 0) {
                    std::cout << "Reading samples for image #" << imgIndex << " with " << imgSamps << " sample pixels\n";
                }
                if (!ifs || imgSamps < 0) break;
                while (imgSamps--) {
                    samples.emplace_back();
                    Sample& sample = samples.back();
                    sample.index = imgIndex;
                    util::read_bin(ifs, sample.pix[0]);
                    util::read_bin(ifs, sample.pix[1]);
                }
            }
            ifs.close();
        }

    private:
        std::mutex trainMutex;
        void trainFromNode(RTree::RNode& node, size_t start, size_t end, uint32_t depth) {
            size_t mid;
            float bestThresh, bestInfoGain = -FLT_MAX;
            Feature bestFeature;
            {
                // Add a leaf leaf
                if (depth <= 1 || end - start <= static_cast<size_t>(minSamples)) {
                    node.leafid = static_cast<int>(leafData.size());
                    if (verbose) {
                        if (node.leafid % 500 == 0) {
                            std::cout << "Added leaf node: id=" << node.leafid << "\n";
                        }
                    }
                    leafData.emplace_back();
                    leafData.back().resize(numParts);
                    leafData.back().setZero();
                    for (size_t i = start; i < end; ++i) {
                        auto samplePart = dataLoader.get(samples[i], DATA_PART_MASK)[DATA_PART_MASK]
                            .template at<uint8_t>(samples[i].pix.y(), samples[i].pix.x());
                        leafData.back()(samplePart) += 1.f;
                    }
                    leafData.back() /= leafData.back().sum();
                    return;
                }
                if (verbose) {
                    std::cout << "Training internal node, remaining depth: " << depth <<
                        ". Current data interval: " << start << " to " << end << "\n" << std::flush;
                }

                using FeatureVec = std::vector<Feature, Eigen::aligned_allocator<Feature> >;
                FeatureVec candidateFeatures;
                candidateFeatures.resize(numFeatures);

                static const double MIN_PROBE = 0.1;
                // Create random features
                for (auto& feature : candidateFeatures) {
                    // Create random feature in-place
                    feature.u.x() =
                        random_util::uniform(-maxProbeOffset + MIN_PROBE, maxProbeOffset - MIN_PROBE);
                    feature.u.x() += (feature.u.x() >= 0.0 ? MIN_PROBE : -MIN_PROBE);
                    feature.u.y() = random_util::uniform(-maxProbeOffset + MIN_PROBE, maxProbeOffset - MIN_PROBE);
                    feature.u.y() += (feature.u.y() >= 0.0 ? MIN_PROBE : -MIN_PROBE);
                    feature.v.x() = random_util::uniform(-maxProbeOffset + MIN_PROBE, maxProbeOffset - MIN_PROBE);
                    feature.v.x() += (feature.v.x() >= 0.0 ? MIN_PROBE : -MIN_PROBE);
                    feature.v.y() = random_util::uniform(-maxProbeOffset + MIN_PROBE, maxProbeOffset - MIN_PROBE);
                    feature.v.y() += (feature.v.y() >= 0.0 ? MIN_PROBE : -MIN_PROBE);
                }

                // Precompute features scores
                if (verbose && end-start > 1000) {
                    std::cout << "Allocating memory and sampling...\n";
                }
                SampleVec subsamples;
                // TODO: Optimize this to avoid copy (not really bottleneck)
                subsamples.reserve(samplesPerFeature);
                if (end-start <= static_cast<size_t>(samplesPerFeature)) {
                    // Use all samples
                    std::copy(samples.begin() + start, samples.begin() + end, std::back_inserter(subsamples));
                } else {
                    SampleVec _tmp;
                    _tmp.reserve(end-start);
                    // Choose sparse subset of samples
                    // Copy then sample is less costly than sorting again
                    std::copy(samples.begin() + start, samples.begin() + end, std::back_inserter(_tmp));
                    subsamples = random_util::choose(_tmp, samplesPerFeature);
                    reorderByImage(subsamples, 0, subsamples.size());
                }
                Eigen::MatrixXd sampleFeatureScores(subsamples.size(), numFeatures);
                Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> sampleParts(subsamples.size());
                if (verbose && end-start > 500) {
                    std::cout << " > Preload " << subsamples.size() << " sparse samples\n" << std::flush;
                }
                dataLoader.preload(subsamples, 0, subsamples.size(), numThreads);

                // CONCURRENT OP 1:
                // This worker loads and computes the pixel scores and parts for sparse features
                std::vector<std::thread> threadMgr;
                if (verbose && end-start > 500) {
                    std::cout << "Computing features on sparse samples...\n" << std::flush;
                }
                auto sparseFeatureWorker = [&](size_t left, size_t right) {
                    for (size_t sampleId = left; sampleId < right; ++sampleId) {
                        auto& sample = subsamples[sampleId];
                        const auto& dataArr = dataLoader.get(sample);

                        sampleParts(sampleId) = dataArr[DATA_PART_MASK]
                            .template at<uint8_t>(sample.pix[1], sample.pix[0]);

                        for (int featureId = 0; featureId < numFeatures; ++featureId) {
                            auto& feature = candidateFeatures[featureId];
                            sampleFeatureScores(sampleId, featureId) =
                                scoreByFeature(dataArr[DATA_DEPTH],
                                        sample.pix, feature.u, feature.v);
                        }
                    }
                };

                if (subsamples.size() < 50) {
                    // Better to not create threads
                    sparseFeatureWorker(0, subsamples.size());
                } else {
                    size_t step = subsamples.size() / numThreads;
                    for (int i = 0; i < numThreads-1; ++i) {
                        threadMgr.emplace_back(sparseFeatureWorker, step * i, step * (i + 1));
                    }
                    threadMgr.emplace_back(sparseFeatureWorker, step * (numThreads-1), subsamples.size());
                    for (int i = 0; i < numThreads; ++i) {
                        threadMgr[i].join();
                    }
                    threadMgr.clear();
                }

                if (verbose && end-start > 500) {
                    std::cout << "Optimizing information gain on sparse samples...\n" << std::flush;
                }

                // Find best information gain
                std::vector<std::vector<std::array<float, 2> > > featureThreshes(numFeatures);

                // CONCURRENT OP 2:
                // This worker finds threshesPerSample optimal thresholds for each feature on the selected sparse features
                std::atomic<int> featureCount(numFeatures - 1);
                bool shortCircuitOnFeatureOpt = end-start == subsamples.size();

                auto threshWorker = [&](int thread_id) {
                    int featureId;
                    while (true) {
                        featureId = featureCount--;
                        if (featureId < 0) break;
                        if (verbose && end-start > 500 &&
                                featureId % 5000 == 0) {
                            std::cout << " Sparse features to evaluate: " << featureId << "\n";
                        }

                        // float infoGain =
                        computeOptimalThreshes(
                                subsamples, sampleFeatureScores, sampleParts,
                                featureId, featureThreshes[featureId], shortCircuitOnFeatureOpt);
                    }
                };
                for (int i = 0; i < numThreads; ++i) {
                    threadMgr.emplace_back(threshWorker, i);
                }
                for (int i = 0; i < numThreads; ++i) {
                    threadMgr[i].join();
                }

                // Depending on whether the current node is 'small'. we can either fast-forward
                // (use samples from above to determine feature/threshold) or compute complete
                // negative conditional entropy using all samples
                if (shortCircuitOnFeatureOpt) {
                    // Interval is very short (only has subsamples), reuse earlier computations
                    if (verbose) {
                        std::cout << "Fast-forward evaluation for small node\n" << std::flush;
                    }
                    for (int featureId = 0; featureId < numFeatures; ++featureId) {
                        if (featureThreshes[featureId].empty()) {
                            std::cerr<< "WARNING: Encountered feature with no canditates thresholds (skipped)\n";
                            continue;
                        }
                        auto& bestFeatureThresh = featureThreshes[featureId][0];
                        if (bestFeatureThresh[0] > bestInfoGain) {
                            bestInfoGain = bestFeatureThresh[0];
                            bestThresh = bestFeatureThresh[1];
                            bestFeature = candidateFeatures[featureId];
                        }
                    }
                } else {
                    // Interval is long
                    if (verbose && end - start > 500) {
                        std::cout << "Computing part distributions for each candidate feature/threshold pair...\n";
                    }
                    sampleFeatureScores.resize(0, 0);
                    sampleParts.resize(0);
                    {
                        SampleVec _;
                        subsamples.swap(_);
                    }
                    if (verbose && end - start > 500) {
                        std::cout << " > Maybe preload " << end-start << " samples\n";
                    }
                    bool preloaded = dataLoader.preload(samples, start, end, numThreads);
                    if (verbose && end - start > 500) {
                        std::cout << "  > Preload decision: " << preloaded << "\n" << std::flush;
                    }

                    // CONCURRENT OP 3:
                    std::vector<Eigen::MatrixXi, Eigen::aligned_allocator<Eigen::MatrixXi> > featureThreshDist(numFeatures);
                    for (int i = 0; i < numFeatures; ++i) {
                        featureThreshDist[i].resize(featureThreshes[i].size(), numParts * 2);
                        featureThreshDist[i].setZero();
                    }

                    // std::atomic<size_t> sampleCount(start);
                    size_t sampleCount = 0;
                    // Compute part distributions for each feature/threshold pair
                    auto featureDistributionWorker = [&](size_t left, size_t right) {
                        std::vector<Eigen::MatrixXi, Eigen::aligned_allocator<Eigen::MatrixXi> > threadFeatureDist(numFeatures);
                        for (int i = 0; i < numFeatures; ++i) {
                            threadFeatureDist[i].resize(featureThreshes[i].size(), numParts * 2);
                            threadFeatureDist[i].setZero();
                        }
                        if (right > end || left < start) {
                            std::cerr << "FATAL: Interval " << left << ", " << right << " is not valid\n";
                            std::exit(0);
                        }
                        if (right <= left) return;
                        for (size_t sampleId = left; sampleId < right; ++sampleId) {
                            // sampleId = sampleCount++;
                            // if (sampleId >= end) break;
                            if (verbose && end-start > 5000 &&
                                    sampleId > left &&
                                    (sampleId - left) % 10000 == 0) {
                                sampleCount += 10000; // This is not really safe but for displaying only anyway
                                std::cout << " Approx samples evaluated: " << sampleCount << " of " << end-start << "\n";
                                if (sampleCount % 1000000 == 0) std::cout << std::flush;
                            }
                            auto& sample = samples[sampleId];
                            const auto& dataArr = dataLoader.get(sample);

                            if (dataArr[DATA_PART_MASK].rows <= sample.pix[1]
                                    || dataArr[DATA_PART_MASK].cols <= sample.pix[0]
                                    || dataArr[DATA_DEPTH].rows <= sample.pix[1]
                                    || dataArr[DATA_DEPTH].cols <= sample.pix[0]) {
                                std::cerr << "FISHY\n";
                                std::exit(1);
                            }
                            uint32_t samplePart = dataArr[DATA_PART_MASK]
                                .template at<uint8_t>(sample.pix[1], sample.pix[0]);

                            for (int featureId = 0; featureId < numFeatures; ++featureId) {
                                auto& feature = candidateFeatures[featureId];
                                auto& distMat = threadFeatureDist[featureId];
                                float score = scoreByFeature(dataArr[DATA_DEPTH],
                                        sample.pix, feature.u, feature.v);
                                for (size_t threshId = 0; threshId < featureThreshes[featureId].size(); ++threshId) {
                                    uint32_t part = (score > featureThreshes[featureId][threshId][1]) ? samplePart : samplePart + numParts;
                                    ++distMat(threshId, part);
                                }
                            }
                        }
                        {
                            std::lock_guard<std::mutex> lock(trainMutex);
                            for (int featureId = 0; featureId < numFeatures; ++featureId) {
                                featureThreshDist[featureId].noalias() += threadFeatureDist[featureId];
                            }
                        }
                    };

                    // Probably better to use parallel for rather than atomic counter
                    // since will result in less loads (better thread-local image cache, ref. load())
                    size_t step = (end-start) / numThreads;
                    threadMgr.clear();
                    for (int i = 0; i < numThreads - 1; ++i) {
                        threadMgr.emplace_back(featureDistributionWorker,
                                               start + step * i, start + step * (i + 1));
                    }
                    threadMgr.emplace_back(featureDistributionWorker, start + step * (numThreads-1), end);
                    for (int i = 0; i < numThreads; ++i) {
                        threadMgr[i].join();
                    }
                    // CONCURRENT OP 4:
                    // finding optimal feature
                    if (verbose && end-start > 500) {
                        std::cout << "Finding optimal feature...\n" << std::flush;
                    }

                    featureCount = numFeatures - 1;
                    auto featureOptWorker = [&]() {
                        int featureId;
                        float threadBestInfoGain = -FLT_MAX, threadBestThresh = -1;
                        Feature threadBestFeature;
                        threadBestFeature.u.setZero();
                        threadBestFeature.v.setZero();
                        while (true) {
                            featureId = featureCount--;
                            if (featureId < 0) break;
                            if (verbose && end-start > 1000 &&
                                    featureId % 1000 == 0) {
                                std::cout << " Candidate features to evaluate: " << featureId << "\n";
                            }
                            auto& feature = candidateFeatures[featureId];
                            auto& distMat = featureThreshDist[featureId];

                            float featureBestInfoGain = -FLT_MAX;
                            float featureBestThresh = -1.;

                            for (size_t threshId = 0; threshId < featureThreshes[featureId].size(); ++threshId) {
                                auto distLeft = distMat.block<1, Eigen::Dynamic>(threshId, 0, 1, numParts);
                                auto distRight = distMat.block<1, Eigen::Dynamic>(threshId, numParts, 1, numParts);

                                float lsum = distLeft.sum();
                                float rsum = distRight.sum();
                                if (lsum == 0 || rsum == 0) continue;

                                float leftEntropy = entropy(distLeft.template cast<float>() / lsum);
                                float rightEntropy = entropy(distRight.template cast<float>() / rsum);
                                // Compute the information gain
                                float infoGain = - lsum * leftEntropy - rsum * rightEntropy;
                                // if (infoGain > 0) {
                                //     std::cerr << "FATAL: Possibly overflow detected during training, exiting. Internal data: left entropy "
                                //         << leftEntropy << " right entropy "
                                //         << rightEntropy << " information gain "
                                //         << infoGain<< "\n";
                                //     std::exit(2);
                                // }
                                if (infoGain > featureBestInfoGain) {
                                    featureBestInfoGain = infoGain;
                                    featureBestThresh = featureThreshes[featureId][threshId][1];
                                }
                            }
                            if (featureBestInfoGain > threadBestInfoGain) {
                                threadBestInfoGain = featureBestInfoGain;
                                threadBestFeature = feature;
                                threadBestThresh = featureBestThresh;
                            }
                        }
                        {
                            std::lock_guard<std::mutex> lock(trainMutex);
                            if (threadBestInfoGain > bestInfoGain) {
                                bestInfoGain = threadBestInfoGain;
                                bestFeature = threadBestFeature;
                                bestThresh = threadBestThresh;
                            }
                        }
                    };
                    threadMgr.clear();
                    for (int i = 0; i < numThreads; ++i) {
                        threadMgr.emplace_back(featureOptWorker);
                    }
                    for (int i = 0; i < numThreads; ++i) {
                        threadMgr[i].join();
                    }
                }

                if (verbose && end-start > 1000) {
                    std::cout << "Splitting data interval for child nodes.." << std::endl;
                }
                mid = split(start, end, bestFeature, bestThresh);

                if (verbose) {
                    std::cout << "> Best info gain " << bestInfoGain << ", thresh " << bestThresh << ", feature.u " << bestFeature.v.x() << "," << bestFeature.v.y() <<", features.v " << bestFeature.u.x() << "," << bestFeature.u.y() << std::endl; // flush to make sure this is logged
                }
            } // scope to manage memory use
            if (mid == end || mid == start) {
                // force leaf
                trainFromNode(node, start, end, 0);
                /*
                std::cerr << bestFeatures[bestThreadId].u << " U\n";
                std::cerr << bestFeatures[bestThreadId].v << " V\n";
                std::cerr << bestThreshs << " Threshs\n";
                std::cerr << bestInfoGains << " InfoGain\n";
                std::cerr << bestThreadId << " thead\n\n";

                for (int i = start; i < end; ++i) {
                    std::cerr << " " <<
                        scoreByFeature(
                                data[DATA_DEPTH][samples[i].index],
                                samples[i].pix,
                                bestFeatures[bestThreadId].u,
                                bestFeatures[bestThreadId].v);
                }
                std::cerr << "\n";
                std::exit(1);
                */
                return;
            }
            node.thresh = bestThresh;
            node.u = bestFeature.u;
            node.v = bestFeature.v;

            // If the 'info gain' [actually is -(expected new entropy)] was zero then
            // it means all of children have same class, so we should stop
            node.lnode = static_cast<int>(nodes.size());
            nodes.emplace_back();
            if (bestInfoGain == 0.0) {
                trainFromNode(nodes.back(), start, mid, 0);
            } else {
                trainFromNode(nodes.back(), start, mid, depth - 1);
            }

            node.rnode = static_cast<int>(nodes.size());
            nodes.emplace_back();
            if (bestInfoGain == 0.0) {
                trainFromNode(nodes.back(), mid, end, 0);
            } else {
                trainFromNode(nodes.back(), mid, end, depth - 1);
            }
        }

        // Compute information gain (expected decrease in entropy; also corresponds to mutual information of parameter/part distribution)
        // by choosing optimal threshold
        // output into optimal_thresh.col(feature_id)
        // best <= threshesPerSample thresholds are found and returned in arbitrary order
        // if place_best_thresh_first then puts the absolute best threshold first, rest still arbitrary order
        void computeOptimalThreshes(
            const SampleVec& samples,
            const Eigen::MatrixXd& sample_feature_scores,
            const Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>& sample_parts,
            int feature_id, std::vector<std::array<float, 2> >& optimal_threshes,
            bool place_best_thresh_first = false) {

            // Initially everything is in left set
            RTree::Distribution distLeft(numParts), distRight(numParts);
            distLeft.setZero();
            distRight.setZero();

            // Compute scores
            std::vector<std::pair<float, int> > samplesByScore;
            samplesByScore.reserve(samples.size());
            for (size_t i = 0; i < samples.size(); ++i) {
                // const Sample& sample = samples[i];
                samplesByScore.emplace_back(
                        sample_feature_scores(i, feature_id), i);
                uint8_t samplePart = sample_parts(i);
                if (samplePart >= distLeft.size()) {
                    std::cerr << "FATAL: Invalid sample " << int(samplePart) << " detected during RTree training, "
                                 "please check the randomization code\n";
                    std::exit(0);
                }
                distLeft[samplePart] += 1.f;
            }
            static auto scoreComp = [](const std::pair<float, int> & a, const std::pair<float, int> & b) {
                return a.first < b.first;
            };
            std::sort(samplesByScore.begin(), samplesByScore.end(), scoreComp);
            // std::cerr << samplesByScore.size() << "\n";

            // Start everything in the left set ...
            float lastScore = -FLT_MAX;
            for (size_t i = 0; i < samplesByScore.size()-1; ++i) {
                // Update distributions for left, right sets
                int idx = samplesByScore[i].second;
                uint8_t samplePart = sample_parts(idx);
                distLeft[samplePart] -= 1.f;
                distRight[samplePart] += 1.f;
                if (lastScore == samplesByScore[i].first) continue;
                lastScore = samplesByScore[i].first;

                float left_entropy = entropy(distLeft / distLeft.sum());
                float right_entropy = entropy(distRight / distRight.sum());
                // Compute the information gain
                float infoGain = - ((samples.size() - i - 1) * left_entropy
                                 + (i+1)                     * right_entropy);
                if (infoGain > 0) {
                    std::cerr << "FATAL: Possibly overflow detected during training, exiting. Internal data: left entropy "
                        << left_entropy << " right entropy "
                        << right_entropy << " information gain "
                        << infoGain<< "\n";
                    std::exit(2);
                }
                // Add to candidate threshes
                optimal_threshes.push_back({infoGain, random_util::uniform(samplesByScore[i].first, samplesByScore[i+1].first)});
            }
            if (static_cast<size_t>(threshesPerFeature) < optimal_threshes.size()) {
                std::nth_element(optimal_threshes.begin(), optimal_threshes.begin() + threshesPerFeature,
                        optimal_threshes.end(), std::greater<std::array<float, 2> >());
                optimal_threshes.resize(threshesPerFeature);
            }
            if (place_best_thresh_first) {
                std::nth_element(optimal_threshes.begin(), optimal_threshes.begin() + 1, optimal_threshes.end(), std::greater<std::array<float, 2> >());
            }
        }

        void initTraining(int num_images, int num_points_per_image, int max_tree_depth) {
            // 1. Choose num_images random images u.a.r. from given image list
            std::vector<int> allImages(dataLoader.dataSource.size());
            std::iota(allImages.begin(), allImages.end(), 0);
            chosenImages = allImages.size() > static_cast<size_t>(num_images) ?
                random_util::choose(allImages, num_images) : std::move(allImages);

            // 2. Choose num_points_per_image random foreground pixels from each image,
            std::atomic<size_t> imageIndex(0);
            samples.reserve(num_points_per_image * num_images);
            auto worker = [&]() {
                size_t i;
                SampleVec threadSamples;
                threadSamples.reserve(samples.size() / numThreads + 1);
                while (true) {
                    i = imageIndex++;
                    if (i >= chosenImages.size()) break;
                    if (verbose && i % 1000 == 999) {
                        std::cerr << "Preprocessing data: " << i+1 << " of " << num_images << "\n";
                    }
                    cv::Mat mask = dataLoader.get(Sample(chosenImages[i], 0, 0), DATA_PART_MASK)[DATA_PART_MASK];
                    // cv::Mat mask2 = dataLoader.get(Sample(chosenImages[i], 0, 0))[DATA_PART_MASK];
                    // cv::hconcat(mask, mask2, mask);
                    // cv::resize(mask, mask, mask.size() / 2);
                    // cv::imshow("MASKCat", mask);
                    // cv::waitKey(0);
                    std::vector<RTree::Vec2i, Eigen::aligned_allocator<RTree::Vec2i> > candidates;
                    for (int r = 0; r < mask.rows; ++r) {
                        auto* ptr = mask.ptr<uint8_t>(r);
                        for (int c = 0; c < mask.cols; ++c) {
                            if (ptr[c] != 255) {
                                candidates.emplace_back();
                                candidates.back() << c, r;
                            }
                        }
                    }
                    std::vector<RTree::Vec2i, Eigen::aligned_allocator<RTree::Vec2i> > chosenCandidates =
                        (candidates.size() > static_cast<size_t>(num_points_per_image)) ?
                        random_util::choose(candidates, num_points_per_image) : std::move(candidates);
                    for (auto& v : chosenCandidates) {
                        threadSamples.emplace_back(chosenImages[i], v);
                    }
                }
                std::lock_guard<std::mutex> lock(trainMutex);
                std::move(threadSamples.begin(), threadSamples.end(), std::back_inserter(samples));
            };

            {
                std::vector<std::thread> threads;
                for (int i = 0; i < numThreads; ++i) {
                    threads.emplace_back(worker);
                }
                for (int i = 0; i < numThreads; ++i) {
                    threads[i].join();
                }
            }

            if(verbose) {
                std::cerr << "Preprocessing done, sparsely verifying data validity before training...\n";
            }
            for (size_t i = 0; i < samples.size(); i += std::max<size_t>(samples.size() / 100, 1)) {
                auto& sample = samples[i];
                cv::Mat mask = dataLoader.get(sample, DATA_PART_MASK)[DATA_PART_MASK];
                if (mask.at<uint8_t>(sample.pix[1], sample.pix[0]) == 255) {
                    std::cerr << "FATAL: Invalid data detected during verification: background pixels were included in samples.\n";
                    std::exit(0);
                }
            }
            if(verbose) {
                std::cerr << "Result: data is valid\n" << std::flush;
            }
        }

        // Split samples {start ... end-1} by feature+thresh in-place and return the dividing index
        // left (less) set will be {start ... idx-1}, right (greater) set is {idx ... end-1}
        size_t split(size_t start, size_t end, const Feature& feature, float thresh) {
            size_t nextIndex = start;
            // SampleVec temp;
            // temp.reserve(end-start / 2);
            // More concurrency (LOL)
            std::vector<SampleVec> workerLefts(numThreads),
                                   workerRights(numThreads);
            auto worker = [&](int tid, size_t left, size_t right) {
                auto& workerLeft = workerLefts[tid];
                auto& workerRight = workerRights[tid];
                workerLeft.reserve((right - left) / 2);
                workerRight.reserve((right - left) / 2);
                for (size_t i = left; i < right; ++i) {
                    const Sample& sample = samples[i];
                    if (scoreByFeature(dataLoader.get(sample, DATA_DEPTH)[DATA_DEPTH],
                                sample.pix, feature.u, feature.v) < thresh) {
                        workerLeft.push_back(samples[i]);
                    } else {
                        workerRight.push_back(samples[i]);
                    }
                }
            };
            size_t step = (end-start) / numThreads;
            std::vector<std::thread> threadMgr;
            for (int i = 0; i < numThreads - 1; ++i) {
                threadMgr.emplace_back(worker, i,
                        start + step * i, start + step * (i + 1));
            }
            threadMgr.emplace_back(worker, numThreads - 1, start + step * (numThreads-1), end);
            for (int i = 0; i < numThreads; ++i) {
                threadMgr[i].join();
                std::copy(workerLefts[i].begin(), workerLefts[i].end(), samples.begin() + nextIndex);
                nextIndex += workerLefts[i].size();
            }
            size_t splitIndex = nextIndex;
            for (int i = 0; i < numThreads; ++i) {
                std::copy(workerRights[i].begin(), workerRights[i].end(), samples.begin() + nextIndex);
                nextIndex += workerRights[i].size();
            }
            if (nextIndex != end) {
                std::cerr << "FATAL: Tree internal node splitting failed, "
                    "next index mismatch " << nextIndex << " != " << end << ", something is fishy\n";
                std::exit(0);
            }
            return splitIndex;
            /*
            size_t nextIndex = start;
            for (size_t i = start; i < end; ++i) {
                const Sample& sample = samples[i];
                if (scoreByFeature(dataLoader.get(sample)[DATA_DEPTH],
                            sample.pix, feature.u, feature.v) < thresh) {
                    if (nextIndex != i) {
                        std::swap(samples[nextIndex], samples[i]);
                    }
                    ++nextIndex;
                }
            }
            reorderByImage(samples, start, nextIndex);
            reorderByImage(samples, nextIndex, end);
            return nextIndex;
            */
        }

        // Reorder samples in [start, ..., end-1] by image index to improve cache performance
        void reorderByImage(SampleVec& samples, size_t start, size_t end) {
            static auto sampleComp = [](const Sample & a, const Sample & b) {
                // if (a.index == b.index) {
                //     if (a.pix[1] == b.pix[1]) return a.pix[0] < b.pix[0];
                //     return a.pix[1] < b.pix[1];
                // }
                return a.index < b.index;
            };
            sort(samples.begin() + start, samples.begin() + end, sampleComp);
        }

        std::vector<RTree::RNode, Eigen::aligned_allocator<RTree::RNode> >& nodes;
        std::vector<RTree::Distribution>& leafData;
        const int numParts;
        DataLoader<DataSource> dataLoader;

        int numImages, numFeatures, maxProbeOffset, minSamples, numThreads, samplesPerFeature, threshesPerFeature;
        bool verbose;

        // const int SAMPLES_PER_FEATURE = 60;
        std::vector<int> chosenImages;
        SampleVec samples;
    };

    template<class DataSource>
    class TrainerV2 {
    public:
        struct Feature {
            RTree::Vec2 u, v;
            std::vector<float> threshes;
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        };

        TrainerV2() =delete;
        TrainerV2(const TrainerV2&) =delete;
        TrainerV2(TrainerV2&&) =delete;

        TrainerV2(std::vector<RTree::RNode, Eigen::aligned_allocator<RTree::RNode> >& nodes,
                std::vector<RTree::Distribution>& leaf_data,
                DataSource& data_source,
                int num_parts,
                size_t max_images_loaded)
            : nodes(nodes), leafData(leaf_data), numParts(num_parts),
              dataLoader(data_source, max_images_loaded), needInitTraining(true) {
        }

        void train(int num_images, int num_points_per_image, int num_features,
                   int num_features_filtered,
                   int max_probe_offset, int min_samples, int max_tree_depth,
                   int min_samples_per_feature,
                   float frac_samples_per_feature,
                   int threshes_per_feature, int num_threads,
                   const std::string& save_path,
                   int mem_limit_mb,
                   bool verbose) {
            if (!save_path.empty()) readSamples(save_path, verbose, num_images);
            if (needInitTraining) {
                std::cerr << "Init RTree training (v2) with maximum depth " << max_tree_depth << "\n";

                // Initialize new samples
                initTraining(num_images, num_points_per_image, max_tree_depth, num_threads, verbose);
                needInitTraining = false;
                std::cerr << "Init complete\n";
                if (!save_path.empty()) {
                    std::cerr << "Saving to " << save_path << "\n";
                    writeSamples(save_path);
                }
            } else {
                std::cerr << "Resuming RTree training at depth " << depth << " of " << max_tree_depth << "\n";
            }


            // Train

            /* Current features for each node */
            std::vector<std::vector<Feature, Eigen::aligned_allocator<Feature> > > feats;
            /* Num samples for node, feature, (greater than) thresh, part */
            Eigen::Tensor<float, 4> featureThreshCount;
            /* Num samples for node, feature, part */
            Eigen::MatrixXf nodeCount;

            /* Keeps track of currently running threads */
            std::vector<std::thread> threads;
            /** Precomputed vector of part indices (correct label)
             *  for each sample */
            std::cout << "Computing and caching part label of each sample...\n" << std::flush;
            Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> samplesParts(samples.size());
            size_t partCountDisplay = 0;
            auto partWorker = [&](size_t left, size_t right){
                for (size_t i = left; i < right; ++i) {
                    if ((i - left) % 100000 == 99999) {
                        partCountDisplay += 100000;
                        std::cout << partCountDisplay << " of " << samples.size() << " sample part labels computed\n";
                    }
                    const auto& dataArr = dataLoader.get(samples[i], DATA_PART_MASK);
                    samplesParts(i) = dataArr[DATA_PART_MASK]
                        .template at<uint8_t>(samples[i].pix[1], samples[i].pix[0]);
                }
            };
            size_t step = samples.size() / num_threads;
            for (int i = 0; i < num_threads-1; ++i) {
                threads.emplace_back(partWorker, step * i, step * (i + 1));
            }
            threads.emplace_back(partWorker, step * (num_threads-1), samples.size());
            for (int i = 0; i < num_threads; ++i) {
                threads[i].join();
            }
            threads.clear();

            size_t memAvailable = size_t(mem_limit_mb) * 1024ULL * 1024ULL;
            size_t memPerNode = num_features_filtered *
                (threshes_per_feature + 1) * numParts * (num_threads + 1) * 4;
            size_t nodesPerBatch = std::max<size_t>(memAvailable / memPerNode, 1ULL);
            /* Main loop, exactly once per tree level */
            for (; depth <= max_tree_depth; ++depth) {
                int numNodes = nodes.size() - currStartNode;
                if (numNodes == 0) {
                    std::cout << "Note: quitting early at depth " << depth << " because no nodes in next layer (maybe already separates data perfectly)\n";
                    break;
                }
                std::cout << "\nRTree training (v2) at depth " << depth << ", " << numNodes << " node(s) remaining at this depth\n" << std::flush;

                if (sparse.size() != numNodes) {
                    std::cerr << "FATAL: the size of the sparse samples vector " << sparse.size() << " is different from the number of nodes " << numNodes << "\n";
                    std::exit(1);
                }

                /** STEP 0 Compute random reatures and sparse samples */
                feats.resize(numNodes);
                {
                    static const double MIN_PROBE = 0.1;
                    std::atomic<int> randomGenIndex(0);
                    auto randomGenWorker = [&]() {
                        int i;
                        while (true) {
                            i = randomGenIndex++;
                            if (i >= numNodes) {
                                break;
                            }
                            // Create random features
                            feats[i].resize(num_features);
                            for (auto& feature : feats[i]) {
                                feature.u.x() =
                                    random_util::uniform(-max_probe_offset + MIN_PROBE, max_probe_offset - MIN_PROBE);
                                feature.u.x() += (feature.u.x() >= 0.0 ? MIN_PROBE : -MIN_PROBE);
                                feature.u.y() = random_util::uniform(-max_probe_offset + MIN_PROBE, max_probe_offset - MIN_PROBE);
                                feature.u.y() += (feature.u.y() >= 0.0 ? MIN_PROBE : -MIN_PROBE);
                                feature.v.x() = random_util::uniform(-max_probe_offset + MIN_PROBE, max_probe_offset - MIN_PROBE);
                                feature.v.x() += (feature.v.x() >= 0.0 ? MIN_PROBE : -MIN_PROBE);
                                feature.v.y() = random_util::uniform(-max_probe_offset + MIN_PROBE, max_probe_offset - MIN_PROBE);
                                feature.v.y() += (feature.v.y() >= 0.0 ? MIN_PROBE : -MIN_PROBE);
                            }
                            size_t numSparseSamples = std::max<size_t>(sparse[i].size() * frac_samples_per_feature, min_samples_per_feature);
                            if (sparse[i].size() > numSparseSamples) {
                                for (int j = 0; j < numSparseSamples; ++j) {
                                    size_t r = random_util::randint<size_t>(j, sparse[i].size()-1);
                                    if (r != j) std::swap(sparse[i][j], sparse[i][r]);
                                }
                                sparse[i].resize(numSparseSamples);
                                sparse[i].shrink_to_fit();
                            }
                            sort(sparse[i].begin(), sparse[i].end());
                        }
                    };

                    for (int i = 0; i < std::min(numNodes, num_threads); ++i) {
                        threads.emplace_back(randomGenWorker);
                    }
                    for (int i = 0; i < static_cast<int>(threads.size()); ++i) {
                        threads[i].join();
                    }
                    threads.clear();
                }

                if (!save_path.empty()) {
                    std::cout << "Saving to " << save_path << "\n";
                    writeSamples(save_path);
                }

                /* STEP 1 Choose features and threshes based on sparse samples */
                std::cout << "Computing features on sparse samples...\n" << std::flush;
                for (int nodeid = 0; nodeid < numNodes; ++nodeid) {
                    if (nodeid % 100 == 99) {
                        std::cout << nodeid + 1 << " of " << numNodes << " nodes processed\n";
                        if (nodeid % 1000 == 999) std::cout << std::flush;
                    }
                    // Skip leaves
                    if (~nodes[currStartNode + nodeid].leafid) continue;
                    auto& subsamples = sparse[nodeid];
                    Eigen::MatrixXd sampleFeatureScores(subsamples.size(), num_features);
                    // This worker loads and computes the pixel scores and
                    // parts for sparse features
                    auto sparseFeatureWorker = [&](size_t left, size_t right) {
                        for (size_t sampleId = left; sampleId < right; ++sampleId) {
                            auto& sample = samples[subsamples[sampleId]];
                            const auto& dataArr = dataLoader.get(sample, DATA_DEPTH);

                            for (int featureId = 0; featureId < num_features; ++featureId) {
                                auto& feature = feats[nodeid][featureId];
                                sampleFeatureScores(sampleId, featureId) =
                                    scoreByFeature(dataArr[DATA_DEPTH],
                                            sample.pix, feature.u, feature.v);
                            }
                        }
                    };

                    if (subsamples.size() < 10) {
                        // Better to not create threads
                        sparseFeatureWorker(0, subsamples.size());
                    } else {
                        size_t step = subsamples.size() / num_threads;
                        for (int i = 0; i < num_threads-1; ++i) {
                            threads.emplace_back(sparseFeatureWorker, step * i, step * (i + 1));
                        }
                        threads.emplace_back(sparseFeatureWorker, step * (num_threads-1), subsamples.size());
                        for (int i = 0; i < num_threads; ++i) {
                            threads[i].join();
                        }
                        threads.clear();
                    }

                    // Find best information gain (expected entropy decrease)
                    // This worker finds threshesPerSample optimal thresholds for each feature on the selected sparse features
                    std::atomic<int> featuresLeft(feats[nodeid].size()  - 1);
                    std::vector<std::pair<float, int> > rankedFeats;
                    rankedFeats.reserve(num_features);
                    auto threshWorker = [&](int thread_id) {
                        std::vector<std::pair<float, int> > threadRankedFeats;
                        threadRankedFeats.reserve(num_features / num_threads);
                        int featureId;
                        while (true) {
                            featureId = featuresLeft--;
                            if (featureId < 0) break;
                            float infoGain = computeOptimalThreshes2(
                                    subsamples, sampleFeatureScores, samplesParts,
                                    featureId, feats[nodeid][featureId].threshes,
                                    threshes_per_feature);
                            threadRankedFeats.emplace_back(infoGain, featureId);
                        }
                        std::lock_guard<std::mutex> lock(trainMutex);
                        std::move(threadRankedFeats.begin(), threadRankedFeats.end(), std::back_inserter(rankedFeats));
                    };
                    for (int i = 0; i < num_threads; ++i) {
                        threads.emplace_back(threshWorker, i);
                    }
                    for (int i = 0; i < num_threads; ++i) {
                        threads[i].join();
                    }
                    threads.clear();

                    // Pick best features
                    if (rankedFeats.size() > num_features_filtered) {
                        std::nth_element(rankedFeats.begin(), rankedFeats.begin() + num_features_filtered,
                                         rankedFeats.end(), std::greater<std::pair<float, int> >());
                        for (int j = 0; j < num_features_filtered; ++j) {
                            int featureId = rankedFeats[j].second;
                            if (j != featureId) {
                                std::swap(feats[nodeid][j], feats[nodeid][featureId]);
                            }
                        }
                        feats[nodeid].resize(num_features_filtered);
                    }
                }

                size_t numBatches = (numNodes - 1ULL) / nodesPerBatch + 1ULL;
                std::cout << "Finding best (feature, thresh) for each node...\n" << std::flush;

                /* STEP 2 Concurrently count the number of samples exceeding each feature/thresh pair */
                std::vector<uint8_t> isLeaf(numNodes);
                {
                    for (size_t batchid = 0; batchid < numBatches; ++batchid)
                    {
                        std::cout << "Counting total samples matching each (feature, thresh) pair, batch " << batchid+1 << " of " << numBatches << "...\n" << std::flush;
                        size_t batchBegin = currStartNode + batchid * nodesPerBatch;
                        size_t batchEnd = std::min(currStartNode + (batchid + 1) * nodesPerBatch, nodes.size());
                        featureThreshCount.resize(batchEnd-batchBegin, num_features_filtered, threshes_per_feature, numParts);
                        nodeCount.resize(batchEnd-batchBegin, numParts);
                        featureThreshCount.setZero();
                        nodeCount.setZero();
                        size_t sampCount = 0;
                        auto countWorker = [&](size_t samp_left, size_t samp_right) {
                            Eigen::Tensor<float, 4> threadFeatureThreshCount;
                            Eigen::MatrixXf threadNodeCount(batchEnd-batchBegin, numParts);
                            threadFeatureThreshCount.resize(batchEnd-batchBegin, num_features_filtered, threshes_per_feature, numParts);
                            threadFeatureThreshCount.setZero();
                            threadNodeCount.setZero();
                            for (size_t sampid = samp_left; sampid < samp_right; ++sampid) {
                                if ((sampid - samp_left) % 200000 == 0 && sampid > samp_left) {

                                    sampCount += 200000;
                                    std::cout << "Approximately " << sampCount << " of " << samples.size() << " samples processed\n";
                                    if (sampCount % 5000000 == 0) std::cout << std::flush;
                                }
                                Sample& sample = samples[sampid];
                                int nodeid = assignedNode[sampid];
                                if (nodeid < batchBegin || nodeid >= batchEnd) continue;
                                auto& node = nodes[nodeid];
                                std::vector<Feature, Eigen::aligned_allocator<Feature> > & nodeFeatures = feats[nodeid - currStartNode];
                                if (num_features_filtered < static_cast<int>(nodeFeatures.size())) {
                                    std::cerr << "FATAL: more features generated than allowed " << nodeFeatures.size() << " > " << num_features_filtered <<", terminated\n";
                                    std::exit(1);
                                }
                                uint8_t partid = 0;
                                for (int featid = 0; featid < static_cast<int>(nodeFeatures.size()); ++featid) {
                                    Feature& feature = nodeFeatures[featid];
                                    const auto& dataArr = dataLoader.get(sample, DATA_DEPTH);
                                    partid = samplesParts[sampid];
                                    if (partid >= numParts) {
                                        std::cerr << "FATAL: Invalid part id " << partid << " detected, possibly samples are corrupted\n";
                                        std::exit(1);
                                    }
                                    float score = scoreByFeature(dataArr[DATA_DEPTH],
                                            sample.pix, feature.u, feature.v);

                                    if (threshes_per_feature < static_cast<int>(feature.threshes.size())) {
                                        std::cerr << "FATAL: more threshes generated than allowed for feature " << featid << ": " << feature.threshes.size() << " > " << threshes_per_feature <<", terminated\n";
                                        std::exit(1);
                                    }

                                    for(int threshid = 0; threshid < static_cast<int>(feature.threshes.size()); ++ threshid) {
                                        if (score < feature.threshes[threshid]) {
                                            threadFeatureThreshCount(nodeid - batchBegin, featid, threshid, partid) += 1.0f;
                                        }
                                    }
                                }
                                threadNodeCount(nodeid - batchBegin, partid) += 1.0f;
                            }

                            {
                                std::lock_guard<std::mutex> lock(trainMutex);
                                featureThreshCount += threadFeatureThreshCount;
                                nodeCount.noalias() += threadNodeCount;
                            }
                        };
                        size_t step = samples.size() / num_threads;
                        for (int i = 0; i < num_threads - 1; ++i) {
                            threads.emplace_back(countWorker, step * i, step * (i + 1));
                        }
                        threads.emplace_back(countWorker, step * (num_threads - 1), samples.size());
                        for (int i = 0; i < num_threads; ++i) {
                            threads[i].join();
                        }
                        threads.clear();
                        // Done counting

                        std::cout << "Finding optimal features, batch " << batchid+1 << " of " << numBatches << "...\n" << std::flush;
                        /** STEP 3 finding optimal feature */
                        {
                            Eigen::VectorXf featureCountTotal = nodeCount.rowwise().sum();
                            Eigen::array<float, 1> _dims({3});
                            Eigen::Tensor<float, 3> featureThreshCountTotal = featureThreshCount.sum(_dims);
                            // Compute optimal feature/thresh pairs from counts
                            std::atomic<size_t> nodeOptIndex(batchBegin);
                            auto nodeOptWorker = [&]() {
                                size_t nodeid = 0;
                                uint8_t threadIsLeaf;
                                while (true) {
                                    threadIsLeaf = 0;
                                    nodeid = nodeOptIndex++;
                                    if (nodeid >= batchEnd) break;
                                    if (~nodes[nodeid].leafid) continue;
                                    float bestEntropy = FLT_MAX, bestThresh;
                                    Feature bestFeature;
                                    std::vector<Feature, Eigen::aligned_allocator<Feature> > & nodeFeatures = feats[nodeid - currStartNode];
                                    float total = featureCountTotal(nodeid - batchBegin);
                                    float bestPartTotal = -1.; // DEBUG
                                    for (int featid = 0; featid < static_cast<int>(nodeFeatures.size()); ++featid) {
                                        Feature& feature = nodeFeatures[featid];
                                        for(int threshid = 0; threshid < static_cast<int>(feature.threshes.size()); ++ threshid) {
                                            float entropy = 0.f;
                                            float partTotal = featureThreshCountTotal(nodeid - batchBegin, featid, threshid);
                                            float otherPartTotal = total - partTotal;
                                            float partFrac = partTotal / total, otherFrac = 1.f - partFrac;
                                            for (int partid = 0; partid < numParts; ++partid) {
                                                float p = featureThreshCount(nodeid - batchBegin, featid, threshid, partid) / partTotal;
                                                float q = (nodeCount(nodeid - batchBegin, partid) -
                                                        featureThreshCount(nodeid - batchBegin, featid, threshid, partid)) / otherPartTotal;
                                                if (p > 1e-12) entropy -= p * log2(p) * partFrac;
                                                if (q > 1e-12) entropy -= q * log2(q) * otherFrac;
                                            }
                                            if (entropy < bestEntropy) {
                                                bestEntropy = entropy;
                                                bestFeature = feature;
                                                bestThresh = feature.threshes[threshid];
                                                // Leaf detection
                                                if (partTotal == 0.0 || otherPartTotal == 0.0) {
                                                    threadIsLeaf = 4; // self is leaf
                                                } else if (entropy <= 1e-12f) {
                                                    threadIsLeaf = 3; // both children are leaves
                                                } else {
                                                    threadIsLeaf = 0; // not leaf
                                                    if (partTotal <= 1.0) {
                                                        threadIsLeaf |= 1; // left child is leaf
                                                    }
                                                    if (otherPartTotal <= 1.0) {
                                                        threadIsLeaf |= 2; // right child is leaf
                                                    }
                                                }
                                                bestPartTotal = partTotal;
                                            }
                                        }
                                    }
                                    if (depth >= max_tree_depth) {
                                        // Max depth reached, force this to be a leaf
                                        threadIsLeaf = 4;
                                    }
                                    // Only display for first one (else gets too messy)
                                    if (verbose && nodeid == batchBegin) {
                                        std::cout<< "[First node in batch] Min expected entropy: " << bestEntropy << " thresh: " << bestThresh <<" u:" << bestFeature.u.transpose() << " v:" << bestFeature.v.transpose() << " split:" << bestPartTotal << "," << total - bestPartTotal << " detect leaf? ";

                                        switch(threadIsLeaf) {
                                            case 0: std::cout << "NO"; break;
                                            case 1: std::cout << "Left child"; break;
                                            case 2: std::cout << "Right child"; break;
                                            case 3: std::cout << "Both children"; break;
                                            default: std::cout << "YES";
                                        }
                                        std::cout  << "\n";
                                    }
                                    // best feature/thresh will be the ones at index 0
                                    bestFeature.threshes.resize(1);
                                    nodeFeatures.resize(1);
                                    bestFeature.threshes[0] = bestThresh;
                                    nodeFeatures[0] = bestFeature;
                                    bestFeature.threshes.shrink_to_fit();
                                    nodeFeatures.shrink_to_fit();
                                    if (threadIsLeaf) {
                                        isLeaf[nodeid - currStartNode] = threadIsLeaf;
                                    }
                                }
                            };
                            for (size_t i = 0; i < std::min<size_t>(batchEnd - batchBegin, num_threads); ++i) {
                                threads.emplace_back(nodeOptWorker);
                            }
                            for (int i = 0; i < static_cast<int>(threads.size()); ++i) {
                                threads[i].join();
                            }
                            threads.clear();
                        }
                    }
                }

                std::cout << "Creating new nodes and leaves\n" << std::flush;
                /** STEP 4 making leaves and children */
                int oldStartNode = currStartNode;
                currStartNode = static_cast<int>(nodes.size());
                {

                    size_t oldNumLeaves = leafData.size();
                    // Helper for making a node a leaf (creates leaf data)
                    auto addLeaf = [&](RTree::RNode& node) {
                        node.leafid = static_cast<int>(leafData.size());
                        leafData.emplace_back();
                        leafData.back().resize(numParts);
                        leafData.back().setZero();
                    };

                    // Set threshes, make children nodes for all non-leaf nodes, and add leaf data for leaf nodes
                    for (int nodeid = oldStartNode; nodeid < currStartNode; ++nodeid) {
                        auto& node = nodes[nodeid];
                        if (~node.leafid) continue;
                        if (isLeaf[nodeid - oldStartNode] != 4) {
                            node.u = feats[nodeid - oldStartNode][0].u;
                            node.v = feats[nodeid - oldStartNode][0].v;
                            node.thresh = feats[nodeid - oldStartNode][0].threshes[0];
                            node.lnode = nodes.size();
                            nodes.emplace_back();
                            node.rnode = nodes.size();
                            nodes.emplace_back();
                            if (isLeaf[nodeid - oldStartNode] & uint8_t(1)) {
                                addLeaf(nodes[node.lnode]);
                            }
                            if (isLeaf[nodeid - oldStartNode] & uint8_t(2)) {
                                addLeaf(nodes[node.rnode]);
                            }
                        } else {
                            addLeaf(node);
                        }
                    }
                    if (verbose) {
                        std::cout << "Added " << leafData.size() - oldNumLeaves << " leaf nodes, there are now " << leafData.size() << " leaves in the tree\n";
                    }
                }
                int newNumNodes = static_cast<int>(nodes.size()) - currStartNode;
                sparse.clear();
                sparse.resize(newNumNodes);
                std::cout << "Splitting interval for next tree level...\n" << std::flush;

                /** STEP 5 splitting nodes and preparing for next level */
                // Split non-leaf nodes and add leaves
                auto splitWorker = [&](size_t samp_left, size_t samp_right) {
                    for (size_t sampid = samp_left; sampid < samp_right; ++sampid) {
                        Sample& sample = samples[sampid];
                        int nodeid = assignedNode[sampid];
                        if (nodeid < 0) continue;
                        auto& node = nodes[nodeid];

                        std::vector<Feature, Eigen::aligned_allocator<Feature> > & nodeFeatures = feats[nodeid - oldStartNode];
                        Feature& feature = nodeFeatures[0];
                        const auto& dataArr = dataLoader.get(sample, DATA_DEPTH);
                        int8_t partid = samplesParts[sampid];
                        auto nodeIsLeaf = isLeaf[nodeid - oldStartNode];
                        if (nodeIsLeaf == 4) {
                            // TODO: try to get rid of these locks, they are evil
                            // Create leaf and make distribution
                            assignedNode[sampid] = -1;
                            std::lock_guard<std::mutex> lock(trainMutex);
                            if (node.leafid < 0) {
                                std::cerr << "WHAT\n";
                                std::exit(1);
                            }
                            ++leafData[node.leafid][partid];
                            continue;
                        }
                        if (~node.leafid) {
                            std::cerr << "FATAL: Node should not be leaf!\n";
                            std::exit(1);
                        }
                        if (nodeid < oldStartNode || nodeid >= currStartNode) {
                            std::cerr << "FATAL: Node in wrong interval!\n";
                            std::exit(1);
                        }

                        float score = scoreByFeature(dataArr[DATA_DEPTH],
                                sample.pix, feature.u, feature.v);

                        if (score < feature.threshes[0]) {
                            std::lock_guard<std::mutex> lock(trainMutex);
                            if (nodeIsLeaf & 1) {
                                assignedNode[sampid] = -1;
                                ++leafData[nodes[node.lnode].leafid][partid];
                            } else {
                                sparse[node.lnode - currStartNode].push_back(sampid);
                                assignedNode[sampid] = node.lnode;
                            }
                        } else {
                            std::lock_guard<std::mutex> lock(trainMutex);
                            if (nodeIsLeaf & 2) {
                                assignedNode[sampid] = -1;
                                ++leafData[nodes[node.rnode].leafid][partid];
                            } else {
                                sparse[node.rnode - currStartNode].push_back(sampid);
                                assignedNode[sampid] = node.rnode;
                            }
                        }
                    }
                };
                step = samples.size() / num_threads;
                for (int i = 0; i < num_threads - 1; ++i) {
                    threads.emplace_back(splitWorker, step * i, step * (i + 1));
                }
                threads.emplace_back(splitWorker, step * (num_threads - 1), samples.size());
                for (int i = 0; i < num_threads; ++i) {
                    threads[i].join();
                }
                threads.clear();

                // Normalize leaves
                for (int nodeid = oldStartNode; nodeid < currStartNode; ++nodeid) {
                    auto& node = nodes[nodeid];
                    bool bad = false;
                    if (isLeaf[nodeid - oldStartNode] == 4) {
                        float sum = leafData[node.leafid].sum();
                        if (sum <= 0.f) bad = true;
                        leafData[node.leafid] /= sum;
                    } else {
                        if (isLeaf[nodeid - oldStartNode] & 1) {
                            float sum = leafData[nodes[node.lnode].leafid].sum();
                            if (sum <= 0.f) bad = true;
                            leafData[nodes[node.lnode].leafid] /= sum;
                        }
                        if (isLeaf[nodeid - oldStartNode] & 2) {
                            float sum = leafData[nodes[node.rnode].leafid].sum();
                            if (sum <= 0.f) bad = true;
                            leafData[nodes[node.rnode].leafid] /= sum;
                        }
                    }
                    if (bad) {
                        std::cerr << "FATAL: Empty node (sum 0) detected\n";
                        std::exit(1);
                    }
                }
            }
            needInitTraining = true;
            if (!save_path.empty()) {
                std::cout << "[Almost done] Saving to " << save_path << "\n";
                writeSamples(save_path);
            }

            std::cout << "Training finished :)\n" << std::flush;
        }

        void writeSamples(const std::string & path) {
            std::ofstream ofs(path, std::ios::out | std::ios::binary);
            ofs.write("RTREE_V2 ", 9);
            util::write_bin(ofs, numParts);
            dataLoader.dataSource.serialize(ofs);
            util::write_bin(ofs, needInitTraining);
            util::write_bin(ofs, depth);
            util::write_bin(ofs, currStartNode);

            util::write_bin(ofs, sparse.size());
            for (auto& spc : sparse) {
                util::write_bin(ofs, spc.size());
                for (auto& sz: spc) {
                    util::write_bin(ofs, sz);
                }
            }

            util::write_bin(ofs, size_t(assignedNode.rows()));
            for (int i = 0; i < assignedNode.rows(); ++i) {
                util::write_bin(ofs, int(assignedNode[i]));
            }

            util::write_bin(ofs, size_t(nodes.size()));
            for (int i = 0; i < nodes.size(); ++i) {
                util::write_bin(ofs, nodes[i].u[0]);
                util::write_bin(ofs, nodes[i].u[1]);
                util::write_bin(ofs, nodes[i].v[0]);
                util::write_bin(ofs, nodes[i].v[1]);
                util::write_bin(ofs, nodes[i].thresh);
                util::write_bin(ofs, nodes[i].lnode);
                util::write_bin(ofs, nodes[i].rnode);
                util::write_bin(ofs, nodes[i].leafid);
            }

            util::write_bin(ofs, size_t(leafData.size()));
            for (int i = 0; i < leafData.size(); ++i) {
                for (int j = 0; j < numParts; ++j) {
                    util::write_bin(ofs, leafData[i][j]);
                }
            }

            reorderByImage(samples, 0, samples.size());
            ofs.write("S\n", 2);
            size_t last_idx = 0;
            util::write_bin(ofs, samples.size());
            for (size_t i = 0; i <= samples.size(); ++i) {
                if (i == samples.size() ||
                    samples[i].index != samples[last_idx].index) {
                    util::write_bin(ofs, samples[last_idx].index);
                    util::write_bin(ofs, int(i - last_idx));
                    for (size_t j = last_idx; j < i; ++j) {
                        util::write_bin(ofs, samples[j].pix[0]);
                        util::write_bin(ofs, samples[j].pix[1]);
                    }
                    last_idx = i;
                }
            }
            ofs.close();
        }

        void readSamples(const std::string & path, bool verbose = false, int max_num_images = -1) {
            std::ifstream ifs(path, std::ios::in | std::ios::binary);
            if (!ifs) {
                if (verbose) {
                    std::cerr << "Could not open " << path << ", assuming new file\n";
                }
                return;
            }
            if (verbose) {
                std::cout << "Recovering data source from samples file\n";
            }
            char rtreev2marker[9];
            ifs.read(rtreev2marker, 9);
            if (strncmp(rtreev2marker, "RTREE_V2 ", 9)) {
                std::cerr << "ERROR: Invalid or corrupted trainer V2 samples file at " << path << "\n";
                return;
            }
            if (verbose) {
                std::cout << "Reading samples from samples file\n";
            }
            int numPartsCheck;
            util::read_bin(ifs, numPartsCheck);
            if(numPartsCheck != numParts) {
                std::cerr << "ERROR: Trainer V2 samples file at " << path << " has differerent number of parts " <<
                    numParts << ", perhaps you changed the part map or model during training?\n";
                return;
            }
            dataLoader.dataSource.deserialize(ifs);
            util::read_bin(ifs, needInitTraining);
            util::read_bin(ifs, depth);
            util::read_bin(ifs, currStartNode);

            size_t spsz;
            util::read_bin(ifs, spsz);

            sparse.resize(spsz);
            for (auto& spc : sparse) {
                size_t subsz;
                util::read_bin(ifs, subsz);
                spc.resize(subsz);
                for (size_t& sz: spc) {
                    util::read_bin(ifs, sz);
                }
            }

            size_t assignNodeSz;
            util::read_bin(ifs, assignNodeSz);
            assignedNode.resize(assignNodeSz, 1);
            for (int i = 0; i < assignedNode.rows(); ++i) {
                util::read_bin(ifs, assignedNode[i]);
            }

            size_t nodesz;
            util::read_bin(ifs, nodesz);
            nodes.resize(nodesz);
            for (int i = 0; i < nodes.size(); ++i) {
                util::read_bin(ifs, nodes[i].u[0]);
                util::read_bin(ifs, nodes[i].u[1]);
                util::read_bin(ifs, nodes[i].v[0]);
                util::read_bin(ifs, nodes[i].v[1]);
                util::read_bin(ifs, nodes[i].thresh);
                util::read_bin(ifs, nodes[i].lnode);
                util::read_bin(ifs, nodes[i].rnode);
                util::read_bin(ifs, nodes[i].leafid);
            }

            size_t leafsz;
            util::read_bin(ifs, leafsz);
            leafData.resize(leafsz);
            for (int i = 0; i < leafData.size(); ++i) {
                leafData[i].resize(numParts);
                for (int j = 0; j < numParts; ++j) {
                    util::read_bin(ifs, leafData[i][j]);
                }
            }

            char marker[2];
            ifs.read(marker, 2);
            if (strncmp(marker, "S\n", 2)) {
                std::cerr << "ERROR: Invalid or corrupted samples file at " << path << "\n";
                return;
            }
            size_t numSamplesTotal;
            util::read_bin(ifs, numSamplesTotal);
            samples.reserve(numSamplesTotal);
            while (ifs) {
                int imgIndex, imgSamps;
                util::read_bin(ifs, imgIndex);
                if (~max_num_images && imgIndex >= max_num_images) {
                    std::cerr << "Image index " << imgIndex << " out of bounds, invalid samples file?\n";
                    std::exit(0);
                }
                util::read_bin(ifs, imgSamps);
                if (verbose && imgIndex % 1000 == 0 && imgIndex >= 0) {
                    std::cout << "Reading samples for image #" << imgIndex << " with " << imgSamps << " sample pixels\n";
                }
                if (!ifs || imgSamps < 0) break;
                while (imgSamps--) {
                    samples.emplace_back();
                    Sample& sample = samples.back();
                    sample.index = imgIndex;
                    util::read_bin(ifs, sample.pix[0]);
                    util::read_bin(ifs, sample.pix[1]);
                }
            }
            ifs.close();
        }

    private:
        // Mutex to protect resources during training
        std::mutex trainMutex;
        // Compute information gain (expected entropy decrease) by choosing optimal threshold
        // output into optimal_thresh.col(feature_id)
        // best <= threshesPerSample thresholds are found and returned in arbitrary order
        // if place_best_thresh_first then puts the absolute best threshold first, rest still arbitrary order
        float computeOptimalThreshes2(
            const std::vector<size_t>& indices,
            const Eigen::MatrixXd& sample_feature_scores,
            const Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>& sample_parts,
            int feature_id, std::vector<float>& output_threshes,
            int threshes_per_feature) {

            // Initially everything is in left set
            RTree::Distribution distLeft(numParts), distRight(numParts);
            distLeft.setZero();
            distRight.setZero();

            // Compute scores
            std::vector<std::pair<float, int> > samplesByScore;
            samplesByScore.reserve(indices.size());
            for (size_t i = 0; i < indices.size(); ++i) {
                samplesByScore.emplace_back(
                        sample_feature_scores(i, feature_id), i);
                uint8_t samplePart = sample_parts(indices[i]);
                if (samplePart >= distLeft.size()) {
                    std::cerr << "FATAL: Invalid sample " << int(samplePart) << " detected during RTree training, "
                                 "please check the randomization code\n";
                    std::exit(0);
                }
                distLeft[samplePart] += 1.f;
            }
            static auto scoreComp = [](const std::pair<float, int> & a, const std::pair<float, int> & b) {
                return a.first < b.first;
            };
            std::sort(samplesByScore.begin(), samplesByScore.end(), scoreComp);
            // std::cerr << samplesByScore.size() << "\n";

            // Start everything in the left set ...
            float lastScore = -FLT_MAX;
            if (samplesByScore.empty()) {
                std::cerr << "FATAL: not enough samples to compute threshes indices.size()=" << indices.size() << "\n";
                std::exit(1);
            }
            std::vector<std::array<float, 2> > optimalThreshes;
            for (size_t i = 0; i < samplesByScore.size()-1; ++i) {
                // Update distributions for left, right sets
                int idx = samplesByScore[i].second;

                uint8_t samplePart = sample_parts(indices[idx]);
                distLeft[samplePart] -= 1.f;
                distRight[samplePart] += 1.f;
                if (lastScore == samplesByScore[i].first) continue;
                lastScore = samplesByScore[i].first;

                float left_entropy = entropy(distLeft / distLeft.sum());
                float right_entropy = entropy(distRight / distRight.sum());
                // Compute the information gain
                float infoGain = - ((indices.size() - i - 1) * left_entropy
                                 + (i+1)                     * right_entropy);
                if (infoGain > 0) {
                    std::cerr << "FATAL: Possibly overflow detected during training, exiting. Internal data: left entropy "
                        << left_entropy << " right entropy "
                        << right_entropy << " information gain "
                        << infoGain<< "\n";
                    std::exit(2);
                }
                // Add to candidate threshes
                optimalThreshes.push_back({infoGain, random_util::uniform(samplesByScore[i].first, samplesByScore[i+1].first)});
            }
            if (static_cast<size_t>(threshes_per_feature) < optimalThreshes.size()) {
                std::nth_element(optimalThreshes.begin(), optimalThreshes.begin() + threshes_per_feature,
                        optimalThreshes.end(), std::greater<std::array<float, 2> >());
            }
            std::nth_element(optimalThreshes.begin(), optimalThreshes.begin() + 1, optimalThreshes.end(), std::greater<std::array<float, 2> >());
            size_t numOutputThreshes = std::min(static_cast<size_t>(threshes_per_feature), optimalThreshes.size());
            output_threshes.reserve(numOutputThreshes);
            output_threshes.clear();
            for (size_t i = 0; i < numOutputThreshes; ++i) {
                output_threshes.push_back(optimalThreshes[i][1]);
            }
            if (optimalThreshes.empty()) {
                std::cerr << "FATAL: no threshes found, input indices size: " << indices.size() << "\n";
            }
            return optimalThreshes[0][0];
        }

        void initTraining(int num_images, int num_points_per_image, int max_tree_depth, int num_threads, bool verbose) {
            // 1. Choose num_images random images u.a.r. from given image list
            std::vector<int> allImages(dataLoader.dataSource.size());
            std::iota(allImages.begin(), allImages.end(), 0);
            chosenImages = allImages.size() > static_cast<size_t>(num_images) ?
                random_util::choose(allImages, num_images) : std::move(allImages);

            // 2. Choose num_points_per_image random foreground pixels from each image,
            std::atomic<size_t> imageIndex(0);
            samples.reserve(num_points_per_image * num_images);
            auto worker = [&]() {
                size_t i;
                SampleVec threadSamples;
                threadSamples.reserve(samples.size() / num_threads + 1);
                while (true) {
                    i = imageIndex++;
                    if (i >= chosenImages.size()) break;
                    if (verbose && i % 1000 == 999) {
                        std::cerr << "Preprocessing data: " << i+1 << " of " << num_images << "\n";
                    }
                    cv::Mat mask = dataLoader.get(Sample(chosenImages[i], 0, 0), DATA_PART_MASK)[DATA_PART_MASK];
                    // cv::Mat mask2 = dataLoader.get(Sample(chosenImages[i], 0, 0))[DATA_PART_MASK];
                    // cv::hconcat(mask, mask2, mask);
                    // cv::resize(mask, mask, mask.size() / 2);
                    // cv::imshow("MASKCat", mask);
                    // cv::waitKey(0);
                    std::vector<RTree::Vec2i, Eigen::aligned_allocator<RTree::Vec2i> > candidates;
                    for (int r = 0; r < mask.rows; ++r) {
                        auto* ptr = mask.ptr<uint8_t>(r);
                        for (int c = 0; c < mask.cols; ++c) {
                            if (ptr[c] != 255) {
                                candidates.emplace_back();
                                candidates.back() << c, r;
                            }
                        }
                    }
                    std::vector<RTree::Vec2i, Eigen::aligned_allocator<RTree::Vec2i> > chosenCandidates =
                        (candidates.size() > static_cast<size_t>(num_points_per_image)) ?
                        random_util::choose(candidates, num_points_per_image) : std::move(candidates);
                    for (auto& v : chosenCandidates) {
                        threadSamples.emplace_back(chosenImages[i], v);
                    }
                }
                std::lock_guard<std::mutex> lock(trainMutex);
                std::move(threadSamples.begin(), threadSamples.end(), std::back_inserter(samples));
            };

            {
                std::vector<std::thread> threads;
                for (int i = 0; i < num_threads; ++i) {
                    threads.emplace_back(worker);
                }
                for (int i = 0; i < num_threads; ++i) {
                    threads[i].join();
                }
            }

            if(verbose) {
                std::cerr << "Preprocessing done, sparsely verifying data validity before training...\n";
            }
            for (size_t i = 0; i < samples.size(); i += std::max<size_t>(samples.size() / 100, 1)) {
                auto& sample = samples[i];
                cv::Mat mask = dataLoader.get(sample, DATA_PART_MASK)[DATA_PART_MASK];
                if (mask.at<uint8_t>(sample.pix[1], sample.pix[0]) == 255) {
                    std::cerr << "FATAL: Invalid data detected during verification: background pixels were included in samples.\n";
                    std::exit(0);
                }
            }
            if(verbose) {
                std::cerr << "Result: data is valid\n";
            }

            nodes.resize(1);
            /** Start with everything as samples for root node */
            assignedNode.resize(samples.size());
            assignedNode.setZero(); // Assign all to root
            sparse.resize(1);
            sparse[0].resize(samples.size());
            std::iota(sparse[0].begin(), sparse[0].end(), size_t(0));
            currStartNode = 0;
            depth = 1;
        }

        // Reorder samples in [start, ..., end-1] by image index to improve cache performance
        void reorderByImage(SampleVec& samples, size_t start, size_t end) {
            static auto sampleComp = [](const Sample & a, const Sample & b) {
                // if (a.index == b.index) {
                //     if (a.pix[1] == b.pix[1]) return a.pix[0] < b.pix[0];
                //     return a.pix[1] < b.pix[1];
                // }
                return a.index < b.index;
            };
            sort(samples.begin() + start, samples.begin() + end, sampleComp);
        }

        std::vector<RTree::RNode, Eigen::aligned_allocator<RTree::RNode> >& nodes;
        std::vector<RTree::Distribution>& leafData;
        const int numParts;
        DataLoader<DataSource> dataLoader;

        /* Indices of sparse samples for each node (initially not sparse, is made sparse early in each loop) */
        std::vector<std::vector<size_t > > sparse;
        /* Node sample i is currently assigned to, -1 if unassigned */
        Eigen::VectorXi assignedNode;

        /** True if training has not been initialized */
        bool needInitTraining;
        /** Current start node */
        int currStartNode;
        /** Current depth */
        int depth;

        // const int SAMPLES_PER_FEATURE = 60;
        SampleVec samples;
        std::vector<int> chosenImages;
    };

    /** Fast, high memory trainer for avatar source only */
    class AvatarTrainerV3 {
    public:
        struct Sample3 {
            Sample3 () {}
            Sample3(int index, const RTree::Vec2i& pix, uint8_t label) : index(index), pix(pix), label(label) {};
            Sample3(int index, int r, int c, uint8_t label) : index(index), pix(c, r), label(label) {};

            // Image index
            int index;
            // Pixel position
            RTree::Vec2i pix;
            uint8_t label;
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        };
        using SampleVec3 = std::vector<Sample3, Eigen::aligned_allocator<Sample3> >;


        struct Feature {
            RTree::Vec2 u, v;
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        };

        AvatarTrainerV3() =delete;
        AvatarTrainerV3(const AvatarTrainerV3&) =delete;
        AvatarTrainerV3(AvatarTrainerV3&&) =delete;

        AvatarTrainerV3(std::vector<RTree::RNode, Eigen::aligned_allocator<RTree::RNode> >& nodes,
                std::vector<RTree::Distribution>& leaf_data,
                AvatarDataSource& data_source,
                int num_parts)
            : nodes(nodes), leafData(leaf_data),
              dataSource(data_source), numParts(num_parts) { }

        /** Information gain computation state
         *  (one copy per thread) to save reallocations */
        struct IGTrainState3 {
            IGTrainState3(int numParts, int numThreshes) :
                distLeft(numParts), distRight(numParts),
                buckets(numParts, numThreshes) {}
            RTree::Distribution distLeft, distRight;
            Eigen::MatrixXf buckets;
        };

        void train(int num_images, int num_points_per_image, int num_features,
                   int max_probe_offset, int min_samples, int min_samples_per_feature,
                   int max_tree_depth, int num_threads,
                   const std::string& save_path,
                   bool verbose) {
            // Initialize
            if (save_path.size()) {
                readSamples(save_path);
            }
            bool firstTime = samples.empty();
            initTraining(num_images, num_points_per_image, max_tree_depth, num_threads, verbose);

            std::cout << "\nInit RTree (v3) training with maximum depth " << max_tree_depth << "\n" << std::flush;

            // Train
            numFeatures = num_features;
            maxProbeOffset = max_probe_offset;
            minSamples = min_samples;
            minSamplesPerFeature = min_samples_per_feature;
            numThreads = num_threads;
            savePath = save_path;
            this->verbose = verbose;
            if (firstTime) {
                nodes.resize(1);
                nodes.reserve(std::max((1 << max_tree_depth), 1000));

                nodeInterval.resize(1);
                nodeInterval.reserve(nodes.capacity());
                nodeInterval[0][0] = 0;
                nodeInterval[0][1] = samples.size();
                if (save_path.size()) {
                    std::cout << "Saving to " << save_path << "\n" << std::flush;
                    writeSamples(save_path);
                }
            }

            trainFromNode(0, max_tree_depth);
            std::cout << "RTree v3 training finished (▀̿Ĺ̯▀̿ ̿)\n" << std::flush;
        }

        volatile bool panicMode = false;
    private:
        /** Initialization helper */
        void initTraining(int num_images, int num_points_per_image, int max_tree_depth, int num_threads, bool verbose) {
            // Choose num_points_per_image random foreground pixels from each image,
            std::atomic<size_t> imageIndex(0);
            data.resize(num_images);
            bool firstTime = samples.empty();
            if (firstTime) {
                samples.reserve(num_points_per_image * num_images);
                std::cout << "Initializing training: loading and preprocessing images...\n" << std::flush;
            } else {
                std::cout << "Resuming training: reloading images...\n" << std::flush;
            }
            auto worker = [&]() {
                size_t i;
                SampleVec3 threadSamples;
                threadSamples.reserve(samples.capacity() / num_threads + 1);
                while (true) {
                    i = imageIndex++;
                    if (i >= num_images) break;
                    if (verbose && i % 1000 == 999) {
                        std::cout << "Preprocessing images: " << i+1 << " of " << num_images << "\n" << std::flush;
                    }

                    cv::Mat mask, depth;
                    dataSource.loadSimple(i, depth, mask, !firstTime);
                    data[i] = depth;
                    if (!firstTime) continue;
                    // cv::Mat mask2 = dataLoader.get(Sample(chosenImages[i], 0, 0))[DATA_PART_MASK];
                    // cv::hconcat(mask, mask2, mask);
                    // cv::resize(mask, mask, mask.size() / 2);
                    // cv::imshow("MASKCat", mask);
                    // cv::waitKey(0);
                    std::vector<RTree::Vec2i, Eigen::aligned_allocator<RTree::Vec2i> > candidates;
                    for (int r = 0; r < mask.rows; ++r) {
                        auto* ptr = mask.ptr<uint8_t>(r);
                        for (int c = 0; c < mask.cols; ++c) {
                            if (ptr[c] != 255) {
                                candidates.emplace_back();
                                candidates.back() << c, r;
                            }
                        }
                    }
                    std::vector<RTree::Vec2i, Eigen::aligned_allocator<RTree::Vec2i> > chosenCandidates =
                        (candidates.size() > static_cast<size_t>(num_points_per_image)) ?
                        random_util::choose(candidates, num_points_per_image) : std::move(candidates);
                    for (auto& v : chosenCandidates) {
                        threadSamples.emplace_back(i, v, mask.at<uint8_t>(v(1), v(0)));
                    }
                }
                std::lock_guard<std::mutex> lock(trainMutex);
                std::move(threadSamples.begin(), threadSamples.end(), std::back_inserter(samples));
            };

            {
                std::vector<std::thread> threads;
                for (int i = 0; i < num_threads; ++i) {
                    threads.emplace_back(worker);
                }
                for (int i = 0; i < num_threads; ++i) {
                    threads[i].join();
                }
            }

            std::cout << "Preprocessing done, sparsely verifying data validity before training...\n" << std::flush;
            for (size_t i = 0; i < samples.size(); i += std::max<size_t>(samples.size() / 100, 1)) {
                auto& sample = samples[i];
                cv::Mat mask, depth;
                dataSource.loadSimple(sample.index, depth, mask);
                if (mask.at<uint8_t>(sample.pix[1], sample.pix[0]) == 255) {
                    std::cerr << "FATAL: Invalid data detected during verification: background pixels were included in samples.\n";
                    std::exit(0);
                }
            }
            std::cout << "Result: data is valid\n" << std::flush;
        }

        std::vector<uint8_t> samplesParts;
        std::mutex trainMutex;
        void trainFromNode(int node_id, uint32_t depth) {
            auto& node = nodes[node_id];
            size_t start = nodeInterval[node_id][0],
                   end   = nodeInterval[node_id][1];
            if (~node.leafid) return;
            if (depth <= 1 || end - start <= minSamples) {
                // Leaf
                node.leafid = static_cast<int>(leafData.size());
                if (verbose) {
                    if (node.leafid % 500 == 0) {
                        std::cout << "Added leaf node: id=" << node.leafid << "\n";
                    }
                }
                leafData.emplace_back(numParts);
                leafData.back().setZero();
                for (size_t i = start; i < end; ++i) {
                    leafData.back()(samples[i].label) += 1.f;
                }
                leafData.back() /= leafData.back().sum();
                return;
            }
            if (~node.lnode && ~node.rnode) {
                trainFromNode(node.lnode, depth - 1);
                trainFromNode(node.rnode, depth - 1);
                return;
            }
            if (depth > 4) {
                std::cout << "RTree training (v3) for internal node, remaining depth: " << depth <<
                    ". Current data interval: " << start << " to " << end << "\n";
               if (depth > 6) std::cout << std::flush;
            }
            if (savePath.size() && (depth == 15 || panicMode)) {
                std::cout << "Saving to " << savePath << "\n" << std::flush;
                writeSamples(savePath);
                std::cout << "Save complete\n" << std::flush;
            }
            if (panicMode) {
                std::cout << "PANIC: Termination procedure complete\n" << std::flush;
                std::exit(0);
            }

            int mid;
            float bestInfoGain;
            {
                Eigen::VectorXf bestInfoGains(numThreads, 1);
                Eigen::VectorXf bestThreshs(numThreads, 1);
                bestInfoGains.setConstant(-FLT_MAX);
                std::vector<Feature> bestFeatures(numThreads);

                std::atomic<int> featureCount(numFeatures);
                // Mapreduce-ish
                auto worker = [&](int thread_id) {
                    // Thread-specific training data
                    IGTrainState3 trainState(numParts, minSamplesPerFeature /*misnomer*/);
                    float& bestInfoGain = bestInfoGains(thread_id);
                    float& bestThresh = bestThreshs(thread_id);
                    float optimalThresh;
                    Feature& bestFeature = bestFeatures[thread_id];
                    Feature feature;
                    int threadFeatId;
                    while (true) {
                        threadFeatId = featureCount--;
                        if (threadFeatId <= 0) break;
                        if (end-start > 500000 && depth > 4) {
                            if (threadFeatId % 500 == 0 ||
                                (end-start > 10000000 &&
                                 (threadFeatId % 100 == 0
                                 || (end-start > 200000000
                                     && threadFeatId % 10 == 0)))) {
                                std::cout << threadFeatId << " features remain\n" << std::flush;
                            }
                        }

                        // Create random feature in-place
                        feature.u.x() = random_util::uniform(0.5, maxProbeOffset) * (random_util::randint(0, 2) * 2 - 1);
                        feature.u.y() = random_util::uniform(0.5, maxProbeOffset) * (random_util::randint(0, 2) * 2 - 1);
                        feature.v.x() = random_util::uniform(0.5, maxProbeOffset) * (random_util::randint(0, 2) * 2 - 1);
                        feature.v.y() = random_util::uniform(0.5, maxProbeOffset) * (random_util::randint(0, 2) * 2 - 1);

                        float infoGain = optimalInformationGain3(trainState,
                                start, end, feature, &optimalThresh);

                        if (infoGain >= bestInfoGain) {
                            bestInfoGain = infoGain;
                            bestThresh = optimalThresh;
                            bestFeature = feature;
                        }
                        if (panicMode) break;
                    }
                };

                std::vector<std::thread> threadMgr;
                for (int i = 0; i < numThreads; ++i) {
                    threadMgr.emplace_back(worker, i);
                }

                int bestThreadId = 0;
                for (int i = 0; i < numThreads; ++i) {
                    threadMgr[i].join();
                    if (i && bestInfoGains(i) > bestInfoGains(bestThreadId)) {
                        bestThreadId = i;
                    }
                }
                if (panicMode) {
                    trainFromNode(node_id, depth);
                    return;
                }

                mid = split(start, end, bestFeatures[bestThreadId], bestThreshs(bestThreadId));

                if (panicMode) {
                    trainFromNode(node_id, depth);
                    return;
                }

                bestInfoGain = bestInfoGains(bestThreadId);
                if (depth > 5) {
                    std::cout << "> Best info gain " << bestInfoGain << ", thresh " << bestThreshs(bestThreadId) << ", feature.u " << bestFeatures[bestThreadId].u.x() << "," << bestFeatures[bestThreadId].u.y() <<", features.v" << bestFeatures[bestThreadId].u.x() << "," << bestFeatures[bestThreadId].u.y() << "\n" << std::flush;
                }
                if (mid == end || mid == start) {
                    // force leaf
                    trainFromNode(node_id, 0);
                    return;
                }
                node.thresh = bestThreshs(bestThreadId);
                node.u = bestFeatures[bestThreadId].u;
                node.v = bestFeatures[bestThreadId].v;
            }

            // If the 'info gain' [actually is -(expected new entropy)] was zero then
            // it means all of children have same class, so we should stop
            node.lnode = static_cast<int>(nodes.size());
            nodes.emplace_back();
            nodeInterval.push_back({start, mid});

            node.rnode = static_cast<int>(nodes.size());
            nodes.emplace_back();
            nodeInterval.push_back({mid, end});

            if (bestInfoGain == 0.0) {
                trainFromNode(node.lnode, 0);
                trainFromNode(node.rnode, 0);
            } else {
                trainFromNode(node.lnode, depth - 1);
                trainFromNode(node.rnode, depth - 1);
            }
        }

        void writeSamples(const std::string & path) {
            if (nodes.size() != nodeInterval.size()) {
                std::cerr << "ERROR: node size mismatch " << nodes.size() << " != " << nodeInterval.size() << "\n";
                std::exit(1);
                return;
            }
            std::string tmpPath = path + ".partial";
            std::ofstream ofs(tmpPath, std::ios::out | std::ios::binary);
            ofs.write("RTREE_V3 ", 9);
            util::write_bin(ofs, numParts);
            // util::write_bin(ofs, curStart);
            // util::write_bin(ofs, curEnd);
            dataSource.serialize(ofs);

            ofs.write("N\n", 2);
            util::write_bin<size_t>(ofs, nodes.size());
            for (int i = 0; i < nodes.size(); ++i) {
                ofs.write(reinterpret_cast<char*>(nodes[i].u.data()),
                          2 * sizeof(float));
                ofs.write(reinterpret_cast<char*>(nodes[i].v.data()),
                          2 * sizeof(float));
                util::write_bin(ofs, nodes[i].thresh);
                util::write_bin(ofs, nodes[i].lnode);
                util::write_bin(ofs, nodes[i].rnode);
                util::write_bin(ofs, nodes[i].leafid);
            }

            for (int i = 0; i < nodeInterval.size(); ++i) {
                ofs.write(reinterpret_cast<char*>(nodeInterval[i].data()),
                          2 * sizeof(size_t));
            }

            util::write_bin<size_t>(ofs, leafData.size());
            for (int i = 0; i < leafData.size(); ++i) {
                ofs.write(reinterpret_cast<char*>(leafData[i].data()),
                          numParts * sizeof(float));
            }

            ofs.write("S\n", 2);
            util::write_bin<size_t>(ofs, samples.size());
            for (size_t i = 0; i < samples.size(); ++i) {
                util::write_bin(ofs, samples[i].index);
                util::write_bin(ofs, samples[i].label);
                ofs.write(reinterpret_cast<char*>(samples[i].pix.data()),
                          2 * sizeof(samples[i].pix[0]));
            }

            ofs.write("E\n", 2);
            ofs.close();
            if (boost::filesystem::exists(path)) {
                boost::filesystem::remove(path);
            }
            boost::filesystem::rename(tmpPath, path);
        }

        void readSamples(const std::string & path) {
            std::ifstream ifs(path, std::ios::in | std::ios::binary);
            if (!ifs) {
                std::cerr << "Note: could not open " << path << ", assuming training a new tree\n";
                return;
            }
            std::cout << "Recovering data source from samples file\n";
            char rtreev3marker[9];
            ifs.read(rtreev3marker, 9);
            if (strncmp(rtreev3marker, "RTREE_V3 ", 9)) {
                std::cerr << "ERROR: Invalid or corrupted trainer V3 samples file at " << path << "\n";
                std::exit(1);
            }

            util::read_bin(ifs, numParts);
            dataSource.deserialize(ifs);

            char marker[2];
            ifs.read(marker, 2);
            if (strncmp(marker, "N\n", 2)) {
                std::cerr << "ERROR: Invalid or corrupted samples file at " << path << " [Corrupted N section]\n";
                std::exit(1);
            }

            size_t nodesz;
            util::read_bin(ifs, nodesz);
            nodes.resize(nodesz);
            for (int i = 0; i < nodes.size(); ++i) {
                ifs.read(reinterpret_cast<char*>(nodes[i].u.data()),
                          2 * sizeof(float));
                ifs.read(reinterpret_cast<char*>(nodes[i].v.data()),
                          2 * sizeof(float));
                util::read_bin(ifs, nodes[i].thresh);
                util::read_bin(ifs, nodes[i].lnode);
                util::read_bin(ifs, nodes[i].rnode);
                util::read_bin(ifs, nodes[i].leafid);
            }

            nodeInterval.resize(nodesz);
            for (int i = 0; i < nodeInterval.size(); ++i) {
                ifs.read(reinterpret_cast<char*>(nodeInterval[i].data()),
                          2 * sizeof(size_t));
            }

            size_t leafsz;
            util::read_bin(ifs, leafsz);
            leafData.resize(leafsz);
            for (int i = 0; i < leafData.size(); ++i) {
                leafData[i].resize(numParts);
                ifs.read(reinterpret_cast<char*>(leafData[i].data()),
                            numParts * sizeof(float));
            }

            ifs.read(marker, 2);
            if (strncmp(marker, "S\n", 2)) {
                std::cerr << "ERROR: Invalid or corrupted samples file at " << path << " [Corrupted S section]\n";
                std::exit(1);
            }

            size_t samplessz;
            util::read_bin<size_t>(ifs, samplessz);
            samples.resize(samplessz);
            for (size_t i = 0; i < samplessz; ++i) {
                util::read_bin(ifs, samples[i].index);
                util::read_bin(ifs, samples[i].label);
                ifs.read(reinterpret_cast<char*>(samples[i].pix.data()),
                          2 * sizeof(samples[i].pix[0]));
            }

            ifs.read(marker, 2);
            if (strncmp(marker, "E\n", 2)) {
                std::cerr << "ERROR: Invalid or corrupted samples file at " << path << " [End marker not found]\n";
                std::exit(1);
            }
            ifs.close();
        }

        // Compute information gain (mutual information scaled and shifted) by choosing optimal threshold
        float optimalInformationGain3(IGTrainState3& state, size_t start, size_t end, const Feature& feature, float* optimal_thresh) {

            // Initially everything is in left set
            state.buckets.setZero();
            state.distLeft.setZero();
            state.distRight.setZero();

            // Compute scores
            // std::vector<std::pair<float, int> > samplesByScore;
            float minScore = std::numeric_limits<float>::max();
            float maxScore = std::numeric_limits<float>::lowest();
            for (size_t i = start; i < end; ++i) {
                const Sample3& sample = samples[i];
                float score = scoreByFeature(data[sample.index],
                            sample.pix, feature.u, feature.v);
                minScore = std::min(score, minScore);
                maxScore = std::max(score, maxScore);
                // samplesByScore.emplace_back(
                        // , i);
                state.distLeft[sample.label] += 1.f;
            }

            if (panicMode || minScore > maxScore) return std::numeric_limits<float>::lowest();

            float scoreStep = (maxScore - minScore + std::numeric_limits<float>::epsilon()) / (minSamplesPerFeature + 1.f);

            // Counts per part for each threshold bucket
            for (size_t i = start; i < end; ++i) {
                const Sample3& sample = samples[i];
                float score = scoreByFeature(data[sample.index],
                            sample.pix, feature.u, feature.v);
                size_t buckId = static_cast<size_t>((score - minScore) / scoreStep);
                if (buckId < minSamplesPerFeature) {
                    state.buckets(sample.label, buckId) += 1.f;
                }
            }

            if (panicMode) return std::numeric_limits<float>::lowest();

            // Start everything in the left set ...
            float bestInfoGain = std::numeric_limits<float>::lowest();
            *optimal_thresh = minScore;
            for (size_t i = 0; i < minSamplesPerFeature; ++i) {
                // Update distributions for left, right sets
                state.distLeft.noalias() -= state.buckets.col(i);
                state.distRight.noalias() += state.buckets.col(i);

                float leftSum = state.distLeft.sum(),
                      rightSum = state.distRight.sum();
                float left_entropy = entropy(state.distLeft / leftSum);
                float right_entropy = entropy(state.distRight / rightSum);
                // Compute the information gain
                float infoGain = - (  leftSum * left_entropy
                                    + rightSum * right_entropy);
                if (infoGain > 0) {
                    std::cerr << "FATAL ERROR: Possibly overflow detected during training, exiting. Internal data: left entropy "
                        << left_entropy << " right entropy "
                        << right_entropy << " information gain "
                        << infoGain<< "\n";
                    std::exit(2);
                }
                if (infoGain > bestInfoGain) {
                    // If better then update optimal thresh to between samples
                    *optimal_thresh = minScore + (i + 1) * scoreStep;
                    bestInfoGain = infoGain;
                }
            }
            return bestInfoGain;
        }

        // Split samples {start ... end-1} by feature+thresh in-place and return the dividing index
        // left (less) set willInit  be {start ... idx-1}, right (greater) set is {idx ... end-1}
        size_t split(size_t start, size_t end, const Feature& feature, float thresh) {
            // size_t nextIndex = start;
            // for (size_t i = start; i < end; ++i) {
            //     const Sample3& sample = samples[i];
            //     if (scoreByFeature(data[sample.index],
            //                 sample.pix, feature.u, feature.v) < thresh) {
            //         if (nextIndex != i) {
            //             std::swap(samples[nextIndex], samples[i]);
            //         }
            //         ++nextIndex;
            //     }
            // }
            // return nextIndex;
            //
            size_t nextIndex = start;
            // SampleVec temp;
            // temp.reserve(end-start / 2);
            // More concurrency (LOL)
            std::vector<SampleVec3> workerLefts(numThreads),
                                   workerRights(numThreads);
            auto worker = [&](int tid, size_t left, size_t right) {
                auto& workerLeft = workerLefts[tid];
                auto& workerRight = workerRights[tid];
                workerLeft.reserve((right - left) / 2);
                workerRight.reserve((right - left) / 2);
                for (size_t i = left; i < right; ++i) {
                    const Sample3& sample = samples[i];
                    if (scoreByFeature(data[sample.index],
                                sample.pix, feature.u, feature.v) < thresh) {
                        workerLeft.push_back(samples[i]);
                    } else {
                        workerRight.push_back(samples[i]);
                    }
                }
            };
            size_t step = (end-start) / numThreads;
            std::vector<std::thread> threadMgr;
            for (int i = 0; i < numThreads - 1; ++i) {
                threadMgr.emplace_back(worker, i,
                        start + step * i, start + step * (i + 1));
            }
            threadMgr.emplace_back(worker, numThreads - 1, start + step * (numThreads-1), end);
            for (int i = 0; i < numThreads; ++i) {
                threadMgr[i].join();
                std::copy(workerLefts[i].begin(), workerLefts[i].end(), samples.begin() + nextIndex);
                nextIndex += workerLefts[i].size();
            }
            size_t splitIndex = nextIndex;
            for (int i = 0; i < numThreads; ++i) {
                std::copy(workerRights[i].begin(), workerRights[i].end(), samples.begin() + nextIndex);
                nextIndex += workerRights[i].size();
            }
            if (nextIndex != end) {
                std::cerr << "FATAL: Tree internal node splitting failed, "
                    "next index mismatch " << nextIndex << " != " << end << ", something is fishy\n";
                std::exit(0);
            }
            return splitIndex;
            /*
            size_t nextIndex = start;
            for (size_t i = start; i < end; ++i) {
                const Sample& sample = samples[i];
                if (scoreByFeature(dataLoader.get(sample)[DATA_DEPTH],
                            sample.pix, feature.u, feature.v) < thresh) {
                    if (nextIndex != i) {
                        std::swap(samples[nextIndex], samples[i]);
                    }
                    ++nextIndex;
                }
            }
            reorderByImage(samples, start, nextIndex);
            reorderByImage(samples, nextIndex, end);
            return nextIndex;
            */
        }

        enum {
            DATA_DEPTH,
            DATA_PART_MASK,
            _DATA_TYPE_COUNT
        };

        std::vector<RTree::RNode, Eigen::aligned_allocator<RTree::RNode> >& nodes;
        std::vector<RTree::Distribution>& leafData;
        std::vector<Eigen::Matrix<size_t, 2, 1>, Eigen::aligned_allocator<Eigen::Matrix<size_t, 2, 1>> > nodeInterval;
        std::string savePath;
        SampleVec3 samples;
        std::vector<SparseImage> data;
        AvatarDataSource dataSource;
        bool verbose;
        int numFeatures, maxProbeOffset, minSamples, numThreads, numParts, minSamplesPerFeature;
        size_t curStart, curEnd;

        const int IMREAD_FLAGS[2] = { cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH, cv::IMREAD_GRAYSCALE };
    };

    // UGLY SIGINT handling
    AvatarTrainerV3* currentTrainer = nullptr;
    void sigHandler(int signal){
        if (currentTrainer != nullptr) {
            std::cout << "PANIC: RTree: received SIGINT, entering panic mode (tries to halt and save)\n" << std::flush;
            currentTrainer->panicMode = true;
        }
    }

    // RTree implementation
    RTree::RTree(int num_parts) : numParts(num_parts) {}
    RTree::RTree(const std::string & path) {
        if (!loadFile(path)) {
            fprintf(stderr, "ERROR: RTree failed to initialize from %s\n", path.c_str());
        }
    }

    bool RTree::loadFile(const std::string & path) {
        std::ifstream bifs(path, std::ios::in | std::ios::binary);
        char marker;
        bifs.get(marker);
        if (marker == 'R') {
            // New binary format
            uint32_t nNodes, nLeafs;
            util::read_bin<uint32_t>(bifs, nNodes);
            util::read_bin<uint32_t>(bifs, nLeafs);
            util::read_bin<int32_t>(bifs, numParts);
            nodes.resize(nNodes);
            leafData.resize(nLeafs);
            uint32_t lastLeafId = 0;
            for (size_t i = 0; i < nodes.size(); ++i) {
                uint8_t isLeaf;
                util::read_bin<uint8_t>(bifs, isLeaf);
                if (isLeaf) {
                    leafData[lastLeafId].resize(numParts);
                    leafData[lastLeafId].setZero();
                    uint8_t cnt = 0;
                    util::read_bin<uint8_t>(bifs, cnt);
                    if (cnt > numParts) {
                        std::cerr << "FATAL: leaf has " << int(cnt) << " parts, expected " << numParts << " at most\n";
                        std::exit(1);
                    }
                    for (uint8_t j = 0; j < cnt; ++j) {
                        uint8_t k;
                        util::read_bin<uint8_t>(bifs, k);
                        if (k > numParts) {
                            std::cerr << "FATAL: leaf index " << int(k) << " is out of bounds, at most " << numParts << "\n";
                            std::exit(1);
                        }
                        util::read_bin<float>(bifs, leafData[lastLeafId](k));
                    }
                    nodes[i].leafid = lastLeafId++;
                } else {
                    util::read_bin<int32_t>(bifs, nodes[i].lnode);
                    util::read_bin<int32_t>(bifs, nodes[i].rnode);
                    util::read_bin<float>(bifs, nodes[i].thresh);
                    bifs.read(reinterpret_cast<char*>(nodes[i].u.data()), sizeof(float) * 2);
                    bifs.read(reinterpret_cast<char*>(nodes[i].v.data()), sizeof(float) * 2);
                }
            }
            bifs.get(marker);
            if (marker != 'T') {
                std::cerr << "Error: incorrect RTree format, T end marker missing\n";
                std::exit(1);
            }
            bifs.close();
        } else {
            std::cout << "Note: loading rtree stored in legacy text format\n" << std::flush;
            // Legacy format
            bifs.close();
            size_t nNodes, nLeafs;
            std::ifstream ifs(path);
            if (!ifs) return false;

            ifs >> nNodes >> nLeafs >> numParts;
            nodes.resize(nNodes);
            leafData.resize(nLeafs);

            for (size_t i = 0; i < nNodes; ++i) {
                ifs >> nodes[i].leafid;
                if (nodes[i].leafid < 0) {
                    ifs >> nodes[i].lnode >>
                        nodes[i].rnode >>
                        nodes[i].thresh >>
                        nodes[i].u[0] >>
                        nodes[i].u[1] >>
                        nodes[i].v[0] >>
                        nodes[i].v[1];
                }
            }

            for (size_t i = 0; i < nLeafs; ++i) {
                leafData[i].resize(numParts);
                for (int j = 0 ; j < numParts; ++j){
                    ifs >> leafData[i](j);
                }
            }
        }

        updateBestMatchTable();

        std::ifstream partmap_ifs(path + ".partmap");
        if (!partmap_ifs) {
            std::cout << "Warning: partmap not found, please ensure you downloaded the .partmap file and placed it beside the .srtr; using default map\n";
        } else {
            int numNewParts = 0;
            if (!readPartMap(partmap_ifs, partMap, numNewParts, partMapType)) {
                std::cout << "Warning: partmap at '" << path << ".partmap' has invalid format or is corrupted; using default map\n";
            }
        }
        return true;
    }

    bool RTree::exportFile(const std::string & path) {
        std::ofstream ofs(path, std::ios::out | std::ios::binary);
        ofs.put('R');
        util::write_bin<uint32_t>(ofs, nodes.size());
        util::write_bin<uint32_t>(ofs, leafData.size());
        util::write_bin<int32_t>(ofs, numParts);
        for (size_t i = 0; i < nodes.size(); ++i) {
            util::write_bin<uint8_t>(ofs, (nodes[i].leafid < 0) ? uint8_t(0) : uint8_t(255));
            if (nodes[i].leafid < 0) {
                util::write_bin<int32_t>(ofs, nodes[i].lnode);
                util::write_bin<int32_t>(ofs, nodes[i].rnode);
                util::write_bin<float>(ofs, nodes[i].thresh);
                ofs.write(reinterpret_cast<char*>(nodes[i].u.data()), sizeof(float) * 2);
                ofs.write(reinterpret_cast<char*>(nodes[i].v.data()), sizeof(float) * 2);
            } else {
                uint8_t cnt = 0;
                for (int j = 0; j < numParts; ++j) {
                    if (leafData[nodes[i].leafid](j) != 0.0) {
                        ++cnt;
                    }
                }
                util::write_bin<uint8_t>(ofs, cnt);
                for (int j = 0; j < numParts; ++j) {
                    if (leafData[nodes[i].leafid](j) != 0.0) {
                        util::write_bin<uint8_t>(ofs, j);
                        util::write_bin<float>(ofs, leafData[nodes[i].leafid](j));
                    }
                }
            }
        }
        ofs.put('T');
        ofs.close();
        // Note: below code is for writing legacy format, disabled
        // ofs << std::fixed << std::setprecision(8);
        // ofs << nodes.size() << " " << leafData.size() << " " << numParts << "\n";
        // for (size_t i = 0; i < nodes.size(); ++i) {
        //     ofs << " " << nodes[i].leafid;
        //     if (nodes[i].leafid < 0) {
        //         ofs << "  " << nodes[i].lnode <<
        //             " " << nodes[i].rnode <<
        //             " " << nodes[i].thresh <<
        //             " " << nodes[i].u[0] <<
        //             " " << nodes[i].u[1] <<
        //             " " << nodes[i].v[0] <<
        //             " " << nodes[i].v[1];
        //     }
        //     ofs << "\n";
        // }
        // for (size_t i = 0; i < leafData.size(); ++i) {
        //     ofs << " ";
        //     for (int j = 0 ; j < numParts; ++j){
        //         ofs << leafData[i](j) << " ";
        //     }
        //     ofs << "\n";
        // }
        // ofs.close();
        return true;
    }

     RTree::Distribution RTree::predictRecursive(int nodeid, const cv::Mat& depth, const Vec2i& pix) {
         auto& node = nodes[nodeid];
         if (node.leafid == -1) {
             if (scoreByFeature(depth, pix, node.u, node.v) < node.thresh) {
                 return predictRecursive(node.lnode, depth, pix);
             } else {
                 return predictRecursive(node.rnode, depth, pix);
             }
         } else {
             return leafData[node.leafid];
         }
     }

     uint8_t RTree::predictRecursiveBest(int nodeid, const cv::Mat& depth, const Vec2i& pix) {
         auto& node = nodes[nodeid];
         if (node.leafid == -1) {
             if (scoreByFeature(depth, pix, node.u, node.v) < node.thresh) {
                 return predictRecursiveBest(node.lnode, depth, pix);
             } else {
                 return predictRecursiveBest(node.rnode, depth, pix);
             }
         } else {
             return leafBestMatch[node.leafid];
         }
     }

    RTree::Distribution RTree::predict(const cv::Mat& depth, const Vec2i& pix) {
        return predictRecursive(0, depth, pix);
    }

    uint8_t RTree::predictBest(const cv::Mat& depth, const Vec2i& pix) {
        return predictRecursiveBest(0, depth, pix);
    }

    std::vector<cv::Mat> RTree::predict(const cv::Mat& depth) {
        std::vector<cv::Mat> result;
        result.reserve(numParts);
        for (int i = 0; i < numParts; ++i) {
            result.emplace_back(depth.size(), CV_32F);
            result[i].setTo(0.f);
        }
        Vec2i pix;
        Distribution distr;
        std::vector<float*> ptr(numParts);
        for (int r = 0; r < depth.rows; ++r) {
            pix(1) = r;
            for (int i = 0; i < numParts; ++i) {
                ptr[i] = result[i].ptr<float>(r);
            }
            const auto* inPtr = depth.ptr<float>(r);
            for (int c = 0; c < depth.cols; ++c) {
                if (inPtr[c] <= 0.f) continue;
                pix(0) = c;
                distr = predictRecursive(0, depth, pix);
                for (int i = 0; i < numParts; ++i) {
                    ptr[i][c] = distr(i);
                }
            }
        }
        return result;
    }

    cv::Mat RTree::predictBest(const cv::Mat& depth, int num_threads, int interval,
            cv::Point top_left,
            cv::Point bot_right,
            bool fill_in_gaps) {
        cv::Mat result(depth.size(), CV_8U);
        result.setTo(255);
        if (bot_right.x == -1) {
            bot_right.x = depth.cols - 1;
            bot_right.y = depth.rows - 1;
        }
        std::atomic<int> row(top_left.y);
        auto worker = [&]() {
            Vec2i pix;
            uint8_t* ptr;
            int r;
            while(true) {
                r = (row += interval);
                if (r > bot_right.y) break;
                pix(1) = r;
                ptr = result.ptr<uint8_t>(r);
                const auto* inPtr = depth.ptr<float>(r);
                for (int c = top_left.x; c <= bot_right.x; c += interval) {
                    if (inPtr[c] == 0.f) continue;
                    pix(0) = c;
                    int nodeid = 0;
                    float sampleDepth = inPtr[c];
                    while (nodes[nodeid].leafid == -1) {
                        auto& node = nodes[nodeid];

                        // Add feature u,v and round
                        Eigen::Vector2f ut = node.u / sampleDepth,
                            vt = node.v / sampleDepth;
                        Eigen::Vector2i uti, vti;
                        uti[0] = static_cast<int32_t>(std::round(ut.x()));
                        uti[1] = static_cast<int32_t>(std::round(ut.y()));
                        vti[0] = static_cast<int32_t>(std::round(vt.x()));
                        vti[1] = static_cast<int32_t>(std::round(vt.y()));
                        uti += pix.cast<int32_t>(); vti += pix.cast<int32_t>();

                        float zu, zv;
                        if (uti.x() < top_left.x || uti.y() < top_left.y ||
                            uti.x() > bot_right.x || uti.y() > bot_right.y) {
                            zu = ark::RTree::BACKGROUND_DEPTH;
                        } else {
                            zu = depth.at<float>(uti.y(), uti.x());
                            if (zu == 0.0) zu = ark::RTree::BACKGROUND_DEPTH;
                        }
                        if (vti.x() < top_left.x || vti.y() < top_left.y ||
                            vti.x() > bot_right.x || vti.y() > bot_right.y) {
                            zv = ark::RTree::BACKGROUND_DEPTH;
                        } else {
                            zv = depth.at<float>(vti.y(), vti.x());
                            if (zv == 0.0) zv = ark::RTree::BACKGROUND_DEPTH;
                        }

                        if (zu - zv < node.thresh) {
                            nodeid = node.lnode;
                        } else {
                            nodeid = node.rnode;
                        }
                    }
                    ptr[c] = leafBestMatch[nodes[nodeid].leafid];
                }
            }
        };
        // int step = depth.rows / num_threads;
        std::vector<std::thread> threadMgr;
        for (int i = 0; i < num_threads; ++i) {
            threadMgr.emplace_back(worker);
        }
        for (int i = 0; i < num_threads; ++i) {
            threadMgr[i].join();
        }

        if (fill_in_gaps && interval > 1) {
            upscaleGrid(result, interval, num_threads, top_left, bot_right);
        }
        return result;
    }

    void RTree::train(const std::string& depth_dir,
                   const std::string& part_mask_dir,
                   int num_threads,
                   bool verbose,
                   int num_images,
                   int num_points_per_image,
                   int num_features,
                   int num_features_filtered,
                   int max_probe_offset,
                   int min_samples,
                   int max_tree_depth,
                   int min_samples_per_feature,
                   float frac_samples_per_feature,
                   int threshes_per_feature,
                   int max_images_loaded,
                   int mem_limit_mb,
                   const std::string& train_partial_save_path
               ) {
        nodes.reserve(1 << std::min(max_tree_depth, 22));
        FileDataSource dataSource(depth_dir, part_mask_dir);
        TrainerV2<FileDataSource> trainer(nodes, leafData, dataSource, numParts, static_cast<size_t>(max_images_loaded));
        trainer.train(num_images, num_points_per_image, num_features,
                num_features_filtered,
                max_probe_offset, min_samples, max_tree_depth, min_samples_per_feature, frac_samples_per_feature, threshes_per_feature,
                num_threads, train_partial_save_path, mem_limit_mb, verbose);
        updateBestMatchTable();
    }

    void RTree::trainFromAvatar(AvatarModel& avatar_model,
                   AvatarPoseSequence& pose_seq,
                   CameraIntrin& intrin,
                   cv::Size& image_size,
                   int num_threads,
                   bool verbose,
                   int num_images,
                   int num_points_per_image,
                   int num_features,
                   int num_features_filtered,
                   int max_probe_offset,
                   int min_samples,
                   int max_tree_depth,
                   int min_samples_per_feature,
                   float frac_samples_per_feature,
                   int threshes_per_feature,
                   const std::vector<int>& part_map,
                   int max_images_loaded,
                   int mem_limit_mb,
                   const std::string& train_partial_save_path
               ) {
        nodes.reserve(1 << std::min(max_tree_depth, 22));
        AvatarDataSource dataSource(avatar_model, pose_seq, intrin, image_size, num_images, part_map);
        // TrainerV2<AvatarDataSource> trainer(nodes, leafData, dataSource, numParts, static_cast<size_t>(max_images_loaded));
        // trainer.train(num_images, num_points_per_image, num_features, num_features_filtered,
                // max_probe_offset, min_samples, max_tree_depth, min_samples_per_feature, frac_samples_per_feature,
                // threshes_per_feature, num_threads, train_partial_save_path, mem_limit_mb, verbose);
        AvatarTrainerV3 trainer(nodes, leafData, dataSource, numParts);
        // Ugly way to save when we get SIGINT
        currentTrainer = &trainer;
        signal(SIGINT, sigHandler);
        trainer.train(num_images, num_points_per_image, num_features,
                   max_probe_offset, min_samples, min_samples_per_feature,
                   max_tree_depth, num_threads,
                   train_partial_save_path, verbose);
        currentTrainer = nullptr;
        partMap = part_map;
        updateBestMatchTable();
    }

    void RTree::trainTransfer(AvatarModel& avatar_model,
            AvatarPoseSequence& pose_seq,
            CameraIntrin& intrin,
            cv::Size& image_size,
            int num_threads,
            bool verbose,
            int num_images
            ) {

        Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> newLeafData(numParts, leafData.size());
        newLeafData.setZero();

        {
            AvatarDataSource dataSource(avatar_model, pose_seq, intrin, image_size, num_images, partMap);
            std::atomic<size_t> atomicImageCnt(0);
            std::mutex transMutex;
            auto worker = [&]() {
                Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> threadLeafData(numParts, leafData.size());
                threadLeafData.setZero();
                size_t i;
                while (true) {
                    i = atomicImageCnt++;
                    if (i >= num_images) break;
                    cv::Mat depth, mask;
                    if (i % 1000 == 999) {
                        std::cout << "Training on images: " << i+1 << " of " << num_images << "\n" << std::flush;
                    }
                    dataSource.loadSimple(i, depth, mask);

                    // std::vector<RTree::Vec2i, Eigen::aligned_allocator<RTree::Vec2i> > candidates;
                    // candidates.reserve(mask.rows * mask.cols / 15);
                    RTree::Vec2i v;
                    for (int r = 0; r < mask.rows; ++r) {
                        v[1] = r;
                        auto* ptr = mask.ptr<uint8_t>(r);
                        for (int c = 0; c < mask.cols; ++c) {
                            if (ptr[c] != 255) {
                                v[0] = c;
                                // candidates.emplace_back();
                                int nodeid = 0;
                                while (nodes[nodeid].leafid == -1) {
                                    auto& node = nodes[nodeid];
                                    if (scoreByFeature(depth, v, node.u, node.v) < node.thresh) {
                                        nodeid = nodes[nodeid].lnode;
                                    } else {
                                        nodeid = nodes[nodeid].rnode;
                                    }
                                }

                                int partId = mask.at<uint8_t>(v(1), v(0));
                                int leafId = nodes[nodeid].leafid;
                                threadLeafData(partId, leafId) += 1;
                            }
                        }
                    }
                    // std::vector<RTree::Vec2i, Eigen::aligned_allocator<RTree::Vec2i> > chosenCandidates =
                    //     (candidates.size() > static_cast<size_t>(num_points_per_image)) ?
                    //     random_util::choose(candidates, num_points_per_image) : std::move(candidates);
                    // for (auto& v : chosenCandidates) {
                    // }
                }
                {
                    std::lock_guard<std::mutex> lock(transMutex);
                    newLeafData += threadLeafData;
                }
            };

            std::vector<std::thread> threads;
            for (int i = 0; i < num_threads; ++i) {
                threads.emplace_back(worker);
            }
            for (int i = 0; i < num_threads; ++i) {
                threads[i].join();
            }
        }

        size_t zeroCnt = 0;
        for (size_t i = 0; i < leafData.size(); ++i) {
            uint64_t sum = newLeafData.col(i).sum();
            if (sum > 0.0) {
                leafData[i] = newLeafData.col(i).cast<float>() / static_cast<float>(sum);
            } else {
                ++zeroCnt;
            }
        }
        if (zeroCnt) {
            std::cout << "WARNING: " << zeroCnt << " leaves were unvisited, keeping old weights.\n";
        }
    }

    void RTree::postProcess(cv::Mat& image,
            Eigen::Matrix<double, 2, Eigen::Dynamic>& com_pre,
            int interval,
            int num_threads,
            cv::Point top_left, cv::Point bot_right,
            double dist_to_pre_weight) const {
        if (bot_right.x == -1) {
            bot_right.x = image.cols - 1;
            bot_right.y = image.rows - 1;
        }
        if (com_pre.cols() != numParts) {
            com_pre.resize(2, numParts);
            com_pre.topRows<1>().setConstant(-1.);
            com_pre.bottomRows<1>().setZero();
        }
        // if (interval > 1) majorityGrid(image, interval, numParts);
        if (partMapType == 0) {
            // 'Contiguous' part map: take contiguous blob with best score
            // (size, minus small cost to encourage staying close to previous edstimate)
            suppressPartNonMax(image, interval, numParts, num_threads,
                    top_left, bot_right, com_pre, dist_to_pre_weight);
        } else {
            // 'Disjoint' part map: same body part may not be in a contiguous blob, can't take max
            // instead, just remove small pieces of image to reduce noise
            removeSmallPieces(image, interval, numParts, num_threads, top_left, bot_right);
        }
        if (interval > 1) upscaleGrid(image, interval, num_threads,
                top_left, bot_right);
    }

    void RTree::updateBestMatchTable() {
        leafBestMatch.resize(leafData.size());
        for (size_t i = 0; i < leafData.size(); ++i) {
            float best = std::numeric_limits<float>::lowest();
            for (int j = 0; j < numParts; ++j) {
                if (leafData[i](j) > best) {
                    best = leafData[i](j);
                    leafBestMatch[i] = j;
                }
            }
        }
    }

    bool RTree::readPartMap(std::istream& is, std::vector<int>& result, int& num_new_parts, int& partmap_type) {
        std::string marker;
        is >> marker;
        if (marker != "partmap") {
            return false;
        }
        is >> marker;
        if (marker == "disjoint") {
            partmap_type = 1;
        } else if (marker == "contiguous") {
            partmap_type = 0;
        } else {
            return false;
        }

        int nOldParts, nNewParts;
        is >> marker;
        if (marker != "src") {
            return false;
        }
        is >> nOldParts;
        std::map<std::string, int> oldEnum, newEnum; 
        for (int i = 0; i < nOldParts; ++i) {
            std::string name; is >> name;
            oldEnum[name] = i;
        }
        is >> marker;
        if (marker != "dest") {
            return false;
        }
        is >> nNewParts;
        for (int i = 0; i < nNewParts; ++i) {
            std::string name; is >> name;
            newEnum[name] = i;
        }

        std::string oldName, newName;
        result.resize(nOldParts);
        for (int i = 0; i < nOldParts; ++i) {
            if (!is) break;
            is >> oldName >> newName;
            result[oldEnum[oldName]] = newEnum[newName];
        }
        return true;
    }
}
