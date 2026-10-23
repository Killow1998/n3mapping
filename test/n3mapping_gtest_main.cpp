#include <gtest/gtest.h>
#ifdef N3MAPPING_TEST_GMOCK
#include <gmock/gmock.h>
#endif

int main(int argc, char** argv)
{
#ifdef N3MAPPING_TEST_GMOCK
    testing::InitGoogleMock(&argc, argv);
#else
    testing::InitGoogleTest(&argc, argv);
#endif
    return RUN_ALL_TESTS();
}
