// Measures how well RHPD ranks the revisits that are actually there.
//
// Loop closure on 0723 misses the worst disagreement in the map, between two
// keyframes a metre apart that GICP registers at fitness 0.024 with an inlier
// ratio of 0.999. The candidate never reaches verification through the
// descriptor, and the question is whether that is one unlucky pair or the
// general case.
//
// It has to be answered with the descriptor's own distance function. That is
// not an L2 norm: the three blocks are weighted separately, the auxiliary block
// is halved, the first block is scaled by the lower of the two PCA confidences,
// and the whole thing is minimised against a 180-degree flip of the query.
// Reimplementing that in a script is how a measurement quietly becomes wrong,
// so this links the real one.
//
// Ground truth is geometric and needs no labels: two keyframes within a metre
// of each other horizontally, separated by enough travelled path that the
// odometry no longer pins them together, are the same place.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "n3mapping/config.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/map_serializer.h"
#include "n3mapping/RHPDescriptor.h"

namespace {

double percentile(std::vector<double> v, double q)
{
    if (v.empty()) return std::nan("");
    std::sort(v.begin(), v.end());
    const double i = q * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(i));
    const auto hi = static_cast<std::size_t>(std::ceil(i));
    return v[lo] + (v[hi] - v[lo]) * (i - static_cast<double>(lo));
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::printf("usage: %s <map.pbstream> [xy_tol_m] [min_path_m] [top_k] [part_a_scale] [aux_scale]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const double xy_tol = argc > 2 ? std::atof(argv[2]) : 1.0;
    const double min_path = argc > 3 ? std::atof(argv[3]) : 5.0;
    const int top_k = argc > 4 ? std::atoi(argv[4]) : 30;

    n3mapping::Config config;
    n3mapping::KeyframeManager keyframes(config);
    n3mapping::LoopDetector detector(config);
    n3mapping::GraphOptimizer optimizer(config);
    n3mapping::MapSerializer serializer(config);
    if (!serializer.loadMap(path, keyframes, detector, optimizer)) {
        std::printf("failed to load %s\n", path.c_str());
        return 1;
    }

    const auto all = keyframes.getAllKeyframes();
    std::vector<n3mapping::Keyframe::Ptr> kfs;
    for (const auto& kf : all) {
        if (kf && kf->rhpd_descriptor.size() > 0) kfs.push_back(kf);
    }
    std::sort(kfs.begin(), kfs.end(), [](const auto& a, const auto& b) {
        return a->timestamp < b->timestamp;
    });
    if (kfs.size() < 10) {
        std::printf("only %zu keyframes carry a descriptor\n", kfs.size());
        return 1;
    }

    // Travelled path from the odometry, which is what "far enough apart to be
    // informative" is measured in.
    std::vector<double> travelled(kfs.size(), 0.0);
    for (std::size_t i = 1; i < kfs.size(); ++i) {
        travelled[i] = travelled[i - 1] +
            (kfs[i]->pose_odom.translation() - kfs[i - 1]->pose_odom.translation()).norm();
    }

    const double part_a_scale = argc > 5 ? std::atof(argv[5]) : 1.0;
    const double aux_scale = argc > 6 ? std::atof(argv[6]) : 1.0;

    n3mapping::RHPDescriptor::Params params;
    params.max_range = config.rhpd_max_range;
    params.z_min = config.rhpd_z_min;
    params.z_max = config.rhpd_z_max;
    params.enable_negative_space = config.rhpd_enable_negative_space;
    params.enable_vertical_tokens = config.rhpd_enable_vertical_tokens;
    params.enable_pca_confidence = config.rhpd_enable_pca_confidence;
    params.part_a_scale = part_a_scale;
    params.aux_scale = aux_scale;
    n3mapping::RHPDescriptor descriptor(params);

    std::vector<double> fractional_rank;
    std::vector<double> truth_distance;
    std::vector<double> all_distance;
    int within_top_k = 0;
    int pairs = 0;
    int queries_with_truth = 0;

    for (std::size_t qi = 0; qi < kfs.size(); ++qi) {
        std::vector<std::pair<double, std::size_t>> ranked;
        std::vector<std::size_t> truth;
        for (std::size_t mi = 0; mi < qi; ++mi) {
            if (travelled[qi] - travelled[mi] < min_path) continue;
            const double d = descriptor.distance(kfs[qi]->rhpd_descriptor,
                                                 kfs[mi]->rhpd_descriptor);
            if (!std::isfinite(d)) continue;
            ranked.emplace_back(d, mi);
            if (qi % 5 == 0 && mi % 5 == 0) all_distance.push_back(d);
            const auto& a = kfs[qi]->pose_optimized.translation();
            const auto& b = kfs[mi]->pose_optimized.translation();
            if (std::hypot(a.x() - b.x(), a.y() - b.y()) <= xy_tol) truth.push_back(mi);
        }
        if (truth.empty() || ranked.size() < 2) continue;
        ++queries_with_truth;
        std::sort(ranked.begin(), ranked.end());
        std::map<std::size_t, int> rank_of;
        for (int r = 0; r < static_cast<int>(ranked.size()); ++r) rank_of[ranked[r].second] = r;
        for (const auto mi : truth) {
            const int r = rank_of[mi];
            fractional_rank.push_back(static_cast<double>(r) /
                                      static_cast<double>(ranked.size() - 1));
            truth_distance.push_back(ranked[r].first);
            if (r < top_k) ++within_top_k;
            ++pairs;
        }
    }

    if (pairs == 0) {
        std::printf("no revisit pairs found at xy<=%.2f m, path>=%.2f m\n", xy_tol, min_path);
        return 0;
    }

    std::printf("map              %s\n", path.c_str());
    std::printf("keyframes        %zu with descriptors\n", kfs.size());
    std::printf("block scales     part_a %.2f   aux %.2f\n", part_a_scale, aux_scale);
    std::printf("revisit ground truth: xy<=%.2f m and path>=%.2f m apart\n", xy_tol, min_path);
    std::printf("                 %d pairs across %d queries\n\n", pairs, queries_with_truth);

    int better_than_tenth = 0, better_than_half = 0;
    for (const double f : fractional_rank) {
        if (f < 0.10) ++better_than_tenth;
        if (f < 0.50) ++better_than_half;
    }
    std::printf("where RHPD puts a true revisit (0 = first, 1 = last):\n");
    std::printf("  median %.3f   quartiles %.3f / %.3f      random would be 0.500\n",
                percentile(fractional_rank, 0.50), percentile(fractional_rank, 0.25),
                percentile(fractional_rank, 0.75));
    std::printf("  in the best 10%%: %d/%d (%.0f%%)          random would be 10%%\n",
                better_than_tenth, pairs, 100.0 * better_than_tenth / pairs);
    std::printf("  in the better half: %d/%d (%.0f%%)        random would be 50%%\n",
                better_than_half, pairs, 100.0 * better_than_half / pairs);
    std::printf("  inside the top %d actually taken: %d/%d (%.0f%%)   <- recall\n\n",
                top_k, within_top_k, pairs, 100.0 * within_top_k / pairs);

    // Which block carries the discrimination, if any. Separation is the gap
    // between the two medians as a fraction of the elsewhere median: a block
    // that tells the same place from a different one gives a large positive
    // number, one that does not gives roughly zero.
    {
        const int a_dim = n3mapping::RHPD_PART_A_DIM;
        const int b_dim = n3mapping::RHPD_PART_B_DIM;
        const int aux_dim = n3mapping::RHPD_AUX_DIM;
        const int offsets[3] = {0, a_dim, a_dim + b_dim};
        const int widths[3] = {a_dim, b_dim, aux_dim};
        const char* names[3] = {"Part A (planes)", "Part B (ring-height)", "Aux (neg+tokens)"};
        std::printf("which block separates same-place from elsewhere:\n");
        for (int blk = 0; blk < 3; ++blk) {
            std::vector<double> same, other;
            for (std::size_t qi = 0; qi < kfs.size(); ++qi) {
                for (std::size_t mi = 0; mi < qi; ++mi) {
                    if (travelled[qi] - travelled[mi] < min_path) continue;
                    const double d =
                        (kfs[qi]->rhpd_descriptor.segment(offsets[blk], widths[blk]) -
                         kfs[mi]->rhpd_descriptor.segment(offsets[blk], widths[blk])).norm();
                    if (!std::isfinite(d)) continue;
                    const auto& pa = kfs[qi]->pose_optimized.translation();
                    const auto& pb = kfs[mi]->pose_optimized.translation();
                    if (std::hypot(pa.x() - pb.x(), pa.y() - pb.y()) <= xy_tol) {
                        same.push_back(d);
                    } else if (qi % 5 == 0 && mi % 5 == 0) {
                        other.push_back(d);
                    }
                }
            }
            if (same.empty() || other.empty()) continue;
            const double ms = percentile(same, 0.50);
            const double mo = percentile(other, 0.50);
            std::printf("  %-22s same place %7.3f   elsewhere %7.3f   separation %+5.1f%%\n",
                        names[blk], ms, mo, 100.0 * (mo - ms) / std::max(1e-9, mo));
        }
        std::printf("\n");
    }

    // The shipped distance mixes all three blocks. If two of them separate
    // nothing, the mixture is diluting the one that does, and ranking on that
    // block alone should recall more. Measured rather than argued.
    {
        const int a_dim = n3mapping::RHPD_PART_A_DIM;
        const int b_dim = n3mapping::RHPD_PART_B_DIM;
        const int aux_dim = n3mapping::RHPD_AUX_DIM;
        const int offsets[3] = {0, a_dim, a_dim + b_dim};
        const int widths[3] = {a_dim, b_dim, aux_dim};
        const char* names[3] = {"Part A alone", "Part B alone", "Aux alone"};
        std::printf("recall if ranking used one block only:\n");
        for (int blk = 0; blk < 3; ++blk) {
            int hit = 0, seen = 0;
            std::vector<double> frac;
            for (std::size_t qi = 0; qi < kfs.size(); ++qi) {
                std::vector<std::pair<double, std::size_t>> ranked;
                std::vector<std::size_t> truth;
                for (std::size_t mi = 0; mi < qi; ++mi) {
                    if (travelled[qi] - travelled[mi] < min_path) continue;
                    const double d =
                        (kfs[qi]->rhpd_descriptor.segment(offsets[blk], widths[blk]) -
                         kfs[mi]->rhpd_descriptor.segment(offsets[blk], widths[blk])).norm();
                    if (!std::isfinite(d)) continue;
                    ranked.emplace_back(d, mi);
                    const auto& pa = kfs[qi]->pose_optimized.translation();
                    const auto& pb = kfs[mi]->pose_optimized.translation();
                    if (std::hypot(pa.x() - pb.x(), pa.y() - pb.y()) <= xy_tol) truth.push_back(mi);
                }
                if (truth.empty() || ranked.size() < 2) continue;
                std::sort(ranked.begin(), ranked.end());
                std::map<std::size_t, int> rank_of;
                for (int r = 0; r < static_cast<int>(ranked.size()); ++r) {
                    rank_of[ranked[r].second] = r;
                }
                for (const auto mi : truth) {
                    const int r = rank_of[mi];
                    frac.push_back(static_cast<double>(r) /
                                   static_cast<double>(ranked.size() - 1));
                    if (r < top_k) ++hit;
                    ++seen;
                }
            }
            if (seen == 0) continue;
            std::printf("  %-14s  median position %.3f   top-%d recall %d/%d (%.0f%%)\n",
                        names[blk], percentile(frac, 0.50), top_k, hit, seen,
                        100.0 * hit / seen);
        }
        std::printf("\n");
    }

    std::printf("descriptor distance:\n");
    std::printf("  true revisits  median %.3f   5%%..95%% %.3f .. %.3f\n",
                percentile(truth_distance, 0.50), percentile(truth_distance, 0.05),
                percentile(truth_distance, 0.95));
    std::printf("  all pairs      median %.3f   5%%..95%% %.3f .. %.3f\n",
                percentile(all_distance, 0.50), percentile(all_distance, 0.05),
                percentile(all_distance, 0.95));
    std::printf("  (overlapping ranges mean the descriptor is not separating them)\n");
    return 0;
}
