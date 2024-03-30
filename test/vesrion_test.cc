#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

TEST(VersionTest, CheckGrpcVersion) {
  const auto& version = grpc::Version();
  EXPECT_EQ(version, "1.62.0");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}