#include "n3mapping_node.h"
#include "n3mapping/product_build_identity.h"
#include <iostream>
#include "n3mapping/async_log.h"

int main(int argc, char** argv)
{
    if (n3mapping::productBuildIdentityRequested(argc, argv)) {
        std::cout << n3mapping::productBuildIdentityJson() << '\n';
        return 0;
    }
    n3mapping::ProcessLogging logging(argv[0]);
    n3mapping::RosLogging ros_logging;
    ros::init(argc, argv, "n3mapping_node", ros::init_options::NoRosout);
    {
        n3mapping::N3MappingNoeticNode node;
        node.spin();
    }
    ros::shutdown();
    return 0;
}
