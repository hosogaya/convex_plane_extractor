#pragma once
#include <rclcpp/rclcpp.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <grid_map_cv/grid_map_cv.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <convex_plane_converter/convex_plane_converter.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <iris.h>

namespace convex_plane
{
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
class ConvexPlaneExtractor: public rclcpp::Node
{
public:
    ConvexPlaneExtractor(const rclcpp::NodeOptions options=rclcpp::NodeOptions().use_intra_process_comms(true));
    ~ConvexPlaneExtractor();

private:
    rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr sub_map_;
    rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr pub_map_;
    rclcpp::Publisher<convex_plane_msgs::msg::ConvexPlanesWithGridMap>::SharedPtr pub_plane_with_map_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_;

    void callbackGridMap(const grid_map_msgs::msg::GridMap::UniquePtr msg);
    
    void getContours(grid_map::GridMap& map, const int label, std::vector<Eigen::MatrixXd>& contours_matrix, Eigen::Vector3d& normal, Eigen::Vector2d& seed_pos);
    std::vector<int> findLabels(const grid_map::Matrix& labels);

    void setMarkerArray(const std::vector<Eigen::MatrixXd>& contours, const grid_map::GridMap& map, const int label);
    void setMarkerArray(const Eigen::VectorXd& seed, const grid_map::GridMap& map, const int label);
    void setMarkerArray(const Eigen::MatrixXd& C, const Eigen::VectorXd& d, const grid_map::GridMap& map, const int label);
    iris::IRISOptions options_;
    iris::Solver solver_;
    visualization_msgs::msg::MarkerArray marker_array_;
};
}