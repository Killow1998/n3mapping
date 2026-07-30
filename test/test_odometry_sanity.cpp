#include <gtest/gtest.h>

#include "n3mapping/odometry_sanity.h"

namespace {

using n3mapping::OdometrySanity;
using n3mapping::OdometrySanityLimits;

Eigen::Isometry3d at(double x, double y = 0.0, double z = 0.0) {
    Eigen::Isometry3d p = Eigen::Isometry3d::Identity();
    p.translation() << x, y, z;
    return p;
}

Eigen::Isometry3d yawed(double degrees) {
    Eigen::Isometry3d p = Eigen::Isometry3d::Identity();
    p.rotate(Eigen::AngleAxisd(degrees * M_PI / 180.0, Eigen::Vector3d::UnitZ()));
    return p;
}

// Walking pace on the b22 recording peaks at 2.66 m/s, well inside the limit.
TEST(OdometrySanity, LeavesOrdinaryMotionAlone) {
    OdometrySanity sanity;
    for (int i = 0; i < 500; ++i) {
        sanity.check(i * 0.1, at(i * 0.25));  // 2.5 m/s
    }
    EXPECT_FALSE(sanity.diverged());
}

// One bad interval is a glitch, not a runaway, and must not stop a mapping run.
TEST(OdometrySanity, ToleratesAnIsolatedJump) {
    OdometrySanity sanity;
    sanity.check(0.0, at(0.0));
    sanity.check(0.1, at(0.1));
    sanity.check(0.2, at(50.0));  // 498 m/s for one interval
    sanity.check(0.3, at(50.1));
    sanity.check(0.4, at(50.2));
    EXPECT_FALSE(sanity.diverged());
}

// The f7tof9 runaway is monotonic and never returns; it trips after the
// configured run of violations and not before.
TEST(OdometrySanity, StopsOnASustainedRunaway) {
    OdometrySanityLimits limits;
    limits.max_consecutive_violations = 5;
    OdometrySanity sanity(limits);
    double x = 0.0;
    for (int i = 0; i < 4; ++i) {
        sanity.check(i * 0.1, at(x));
        x += 100.0;  // 1000 m/s
    }
    EXPECT_FALSE(sanity.diverged()) << "tripped before the run was long enough";
    for (int i = 4; i < 10; ++i) {
        sanity.check(i * 0.1, at(x));
        x += 100.0;
    }
    EXPECT_TRUE(sanity.diverged());
    EXPECT_GT(sanity.verdict().speed_mps, 900.0);
}

// A front end that has run away cannot supply the evidence that it recovered,
// so the verdict is not withdrawn when the numbers happen to look sane again.
TEST(OdometrySanity, DoesNotUnstickItself) {
    OdometrySanityLimits limits;
    limits.max_consecutive_violations = 2;
    OdometrySanity sanity(limits);
    sanity.check(0.0, at(0.0));
    sanity.check(0.1, at(100.0));
    sanity.check(0.2, at(200.0));
    ASSERT_TRUE(sanity.diverged());
    const double tripped_at = sanity.verdict().timestamp;
    for (int i = 3; i < 20; ++i) {
        sanity.check(i * 0.1, at(200.0 + (i - 2) * 0.05));
    }
    EXPECT_TRUE(sanity.diverged());
    EXPECT_DOUBLE_EQ(sanity.verdict().timestamp, tripped_at);
}

// Rotation is judged on the shortest arc, so a fast but physical turn passes
// while a tumbling estimate does not.
TEST(OdometrySanity, JudgesRotationOnTheShortestArc) {
    OdometrySanityLimits limits;
    limits.max_angular_rate_dps = 720.0;
    limits.max_consecutive_violations = 3;
    OdometrySanity turning(limits);
    for (int i = 0; i < 40; ++i) {
        turning.check(i * 0.1, yawed(i * 30.0));  // 300 deg/s
    }
    EXPECT_FALSE(turning.diverged());

    OdometrySanity tumbling(limits);
    for (int i = 0; i < 40; ++i) {
        tumbling.check(i * 0.1, yawed(i * 100.0));  // 1000 deg/s
    }
    EXPECT_TRUE(tumbling.diverged());
}

// Replayed bags repeat and reorder stamps; an interval of zero carries no
// information about the front end and must not be divided by.
TEST(OdometrySanity, IgnoresNonAdvancingStamps) {
    OdometrySanity sanity;
    sanity.check(1.0, at(0.0));
    for (int i = 0; i < 20; ++i) {
        sanity.check(1.0, at(1000.0 * i));
    }
    EXPECT_FALSE(sanity.diverged());
}

TEST(OdometrySanity, TreatsANonFinitePoseAsAViolation) {
    OdometrySanityLimits limits;
    limits.max_consecutive_violations = 3;
    OdometrySanity sanity(limits);
    sanity.check(0.0, at(0.0));
    Eigen::Isometry3d nan_pose = Eigen::Isometry3d::Identity();
    nan_pose.translation().x() = std::numeric_limits<double>::quiet_NaN();
    for (int i = 1; i < 6; ++i) {
        sanity.check(i * 0.1, nan_pose);
    }
    EXPECT_TRUE(sanity.diverged());
}

}  // namespace
