#include "n3mapping_node.h"
#include "n3mapping/product_build_identity.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <glog/logging.h>

int main(int argc, char** argv)
{
    if (n3mapping::productBuildIdentityRequested(argc, argv)) {
        std::cout << n3mapping::productBuildIdentityJson() << '\n';
        return 0;
    }
    ros::init(argc, argv, "n3mapping_node");
    const char* configured_log_dir = std::getenv("GLOG_log_dir");
    FLAGS_log_dir = configured_log_dir && *configured_log_dir
        ? configured_log_dir : std::string(N3MAPPING_SOURCE_DIR) + "/logs";
    std::filesystem::create_directories(FLAGS_log_dir);
    FLAGS_logtostderr = false;
    FLAGS_alsologtostderr = false;
    FLAGS_stderrthreshold = google::GLOG_FATAL;
    FLAGS_logbufsecs = 0;
    google::InitGoogleLogging(argv[0]);
    {
        n3mapping::N3MappingNoeticNode node;
        node.spin();
    }
    google::ShutdownGoogleLogging();
    return 0;
}
