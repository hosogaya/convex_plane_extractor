#include <convex_plane_extractor/convex_plane_extractor.h>
#include <omp.h>

namespace convex_plane
{

ConvexPlaneExtractor::ConvexPlaneExtractor(const rclcpp::NodeOptions options)
: rclcpp::Node("convex_plane_extractor", options)
{
    sub_map_ = create_subscription<grid_map_msgs::msg::GridMap>(
        "convex_plane_extractor/input/grid_map", 1, std::bind(&ConvexPlaneExtractor::callbackGridMap, this, std::placeholders::_1)
    );

    pub_map_ = create_publisher<grid_map_msgs::msg::GridMap>(
        "convex_plane_extractor/output/grid_map", 1
    );
    pub_plane_with_map_ = create_publisher<convex_plane_msgs::msg::ConvexPlanesWithGridMap>
    (
        "convex_plane_extractor/output/planes_with_map", 1
    );
    pub_marker_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "convex_plane_extractor/output/contours", 1
    );

    omp_set_num_threads(5);
    problem_.resize(omp_get_max_threads());
    for (size_t i=0; i<problem_.size(); ++i)
        problem_[i].setMaxIteration(5);

}

ConvexPlaneExtractor::~ConvexPlaneExtractor(){}

void ConvexPlaneExtractor::callbackGridMap(const grid_map_msgs::msg::GridMap::UniquePtr msg)
{
    std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();   
    // RCLCPP_INFO(get_logger(), "subscribe map address: 0x%x", &(msg->data));
    grid_map::GridMap map;
    grid_map::GridMapRosConverter::fromMessage(*msg, map);

    const grid_map::Matrix labels = map.get("valid_labels");
    std::vector<int> label_list = findLabels(labels);
    // RCLCPP_INFO(get_logger(), "label list size: %ld", label_list.size());

    convex_plane_msgs::msg::ConvexPlanesWithGridMap::UniquePtr message = std::make_unique<convex_plane_msgs::msg::ConvexPlanesWithGridMap>();
    marker_array_.markers.clear();
    marker_array_.markers.shrink_to_fit();
    grid_map::Index d_index;
    auto total_s = std::chrono::system_clock::now();

    #pragma omp parallel for
    for (size_t i=0; i<label_list.size(); ++i)
    {
        int id = omp_get_thread_num();
        std::vector<iris_2d::Obstacle> contours;
        Eigen::Vector3d normal;
        Eigen::Vector2d seed_pos;
        {
            #pragma omp critical
            getContours(map, label_list[i], contours, normal);
        }
        // setMarkerArray(contours, map, label_list[i]); // show contour
        int j=0;
        problem_[id].setObstacle(contours);
        RCLCPP_INFO(get_logger(), "The number of Obstacles: %ld", contours.size());
        auto iter_s = std::chrono::system_clock::now();
        for (; j<max_iteration_; ++j)
        {
            problem_[id].reset();
            seed_pos = getSeedPos(map, label_list[i]);
            problem_[id].setSeedPos(seed_pos);
            auto solve_s = std::chrono::system_clock::now();
            bool result = problem_[id].solve();
            auto solve_e = std::chrono::system_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(solve_e - solve_s).count();
            // RCLCPP_INFO(get_logger(), "Iris processing time: %lf", elapsed);
            // RCLCPP_INFO(get_logger(), "Ccp time: %lf, IeTime: %lf, iter: %d", problem_[id].getCcpTime(), problem_[id].getIeTime(), problem_[id].getIteration());
            if (result) break;
        }
        auto iter_e = std::chrono::system_clock::now();
        double elapsed  = std::chrono::duration_cast<std::chrono::milliseconds>(iter_e - iter_s).count();
        {
            #pragma omp ciritical
            RCLCPP_INFO(get_logger(), "Total time for a label: %lf", elapsed);
        }
        if (j == max_iteration_) {
            RCLCPP_ERROR(get_logger(), "Failed to get convex area");
            continue;
        }
        RCLCPP_INFO(get_logger(), "The number of iteration: %d", j+1);
        {
            #pragma omp critical
            setMarkerArray(seed_pos, map, label_list[i]); // show seed 
        }
        // RCLCPP_INFO(get_logger(), "Convex region labeled %d is found", label_list[i]);
        // RCLCPP_INFO_STREAM(get_logger(), "C: " << problem_[id].getC());
        // RCLCPP_INFO_STREAM(get_logger(), "d: " << problem_[id].getD().transpose());
        // RCLCPP_INFO_STREAM(get_logger(), "A: " << problem_[id].getA());
        // RCLCPP_INFO_STREAM(get_logger(), "b: " << problem_[id].getB().transpose());
        // RCLCPP_INFO_STREAM(get_logger(), "normal: " << normal.transpose());

        if (!convex_plane::ConvexPlaneConverter::addPlaneToMessage(problem_[id].getRegion(), normal, label_list[i], message->plane))
        {
            RCLCPP_ERROR(get_logger(), "ConvexPlanes message is invalid because the sizes of components are different");
        }
        {
            #pragma omp critical
            setMarkerArray(problem_[id].getC(), problem_[id].getD(), map, label_list[i]); // show ellipsoid
        }
    }   
    auto total_e = std::chrono::system_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_e - total_s).count();
    RCLCPP_INFO(get_logger(), "Total time for computing all convex area: %lf", total_duration);
    pub_marker_->publish(marker_array_);
    // RCLCPP_INFO(get_logger(), "convert to message");
    grid_map::GridMapRosConverter::toMessage(map, message->map);
    // grid_map_msgs::msg::GridMap::UniquePtr out_msg = grid_map::GridMapRosConverter::toMessage(map);

    // RCLCPP_INFO(get_logger(), "publish map address: 0x%x", &(out_msg->data));
    // pub_map_->publish(std::move(out_msg));
    pub_plane_with_map_->publish(std::move(message));
    
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();

    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    RCLCPP_INFO(get_logger(), "processing time: %lf", elapsed);

}

std::vector<int> ConvexPlaneExtractor::findLabels(const grid_map::Matrix& labels)
{
    std::vector<int> label_list(0);
    for (int i=0; i<labels.rows(); ++i)
    {
        for (int j=0; j<labels.cols(); ++j)
        {
            if (std::isnan(labels(i,j))) continue;
            if (labels(i,j) < 1) continue;

            const float value = labels(i,j);
            auto iter = std::find(label_list.begin(), label_list.end(), value);
            if (iter == label_list.end())
            {
                // RCLCPP_INFO(get_logger(), "found label %f", value);
                label_list.push_back(value);
            }
        }
    }

    return label_list;
}



void ConvexPlaneExtractor::getContours(grid_map::GridMap& map, const int label, std::vector<iris_2d::Obstacle>& contours_matrix, Eigen::Vector3d& normal)
{
    // RCLCPP_INFO(get_logger(), "create binary matrix");
    auto start = std::chrono::system_clock::now();
    if (!map.exists("binary_matrix"))
    {
        map.add("binary_matrix", map.get("valid_labels"));
    }
    else 
    {
        map.get("binary_matrix") = map.get("valid_labels");
    }
    grid_map::Matrix& binary = map.get("binary_matrix");
    const grid_map::Matrix& normal_x = map.get("normal_vectors_x");
    const grid_map::Matrix& normal_y = map.get("normal_vectors_y");
    const grid_map::Matrix& normal_z = map.get("normal_vectors_z");
    
    int labeled_cell_num = 0;
    normal.fill(0.0);
    for (grid_map::GridMapIterator iter(map); !iter.isPastEnd(); ++iter)
    {
        grid_map::Index index = *iter;
        if (std::isnan(map.at("valid_labels", *iter))) 
            binary(index(0), index(1)) = 0;

        if (map.at("valid_labels", *iter) != label)
        {
            binary(index(0), index(1)) = 0;
        }
        else
        {
            binary(index(0), index(1)) = 255;
            normal(0) += normal_x(index(0), index(1));
            normal(1) += normal_y(index(0), index(1));
            normal(2) += normal_z(index(0), index(1));
            ++labeled_cell_num;
        }
    }
    if (labeled_cell_num == 0)
    {
        RCLCPP_ERROR(get_logger(), "number of cells labeled %f is equal to 0", label);
        rclcpp::shutdown();
    }
    normal /= labeled_cell_num;

    // RCLCPP_INFO(get_logger(), "conver to image");
    cv::Mat label_image;
    grid_map::GridMapCvConverter::toImage<unsigned char, 1>(map, "binary_matrix", CV_8UC1, label_image);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat label_image_copy = label_image;
    cv::findContours(label_image_copy, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

    int point_num = 0;
    for (size_t i=0; i<contours.size(); ++i)
    {
        point_num += contours[i].size();
    }

    // RCLCPP_INFO(get_logger(), "add layer");
    std::string output_layer = "plane_"+std::to_string(label);
    grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 1>(label_image, output_layer, map, 0.0, label);    

    int contours_index = 0;
    double median_value = (map.get(output_layer).maxCoeff() - map.get(output_layer).minCoeff())/2;
    contours_matrix.resize(point_num);
    grid_map::Position pos_xy;
    grid_map::Index index;
    for (int i=0; i<point_num; ++i) contours_matrix[i].resize(2, 2);
    for (const std::vector<cv::Point>& contour: contours)
    {
        for (size_t i=0; i<contour.size(); ++i)
        {
            index(0) = contour[i].y;
            index(1) = contour[i].x;
            if (!map.getPosition(index, pos_xy))
            {
                RCLCPP_ERROR(get_logger(), "The index (%d, %d) is out of range", index.x(), index.y());
                continue;
            }
            contours_matrix[contours_index+i].col(0) = pos_xy;
            if (i==0) contours_matrix[contours_index+contour.size()-1].col(1) = pos_xy;
            else contours_matrix[contours_index + i-1].col(1) = pos_xy;

            map.at(output_layer, index) = median_value;
        }
        contours_index+=contour.size();
    }

    auto end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    RCLCPP_INFO(get_logger(), "getContours time: %lf", elapsed);
}

iris_2d::Vector ConvexPlaneExtractor::getSeedPos(const grid_map::GridMap& map, const int label)
{
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    iris_2d::Vector seed_pos;
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_int_distribution<> dist_x(0, map.getSize()[0]);
    std::uniform_int_distribution<> dist_y(0, map.getSize()[1]);
    grid_map::Index index;
    std::string layer = "plane_"+std::to_string(label);
    while (true)
    {
        index.x() = dist_x(engine);
        index.y() = dist_y(engine);
        if (map.at(layer, index) == label)
        {
            map.getPosition(index, seed_pos);
            break;
        }
    }
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    RCLCPP_INFO(get_logger(), "get seed time: %lf", elapsed);
    return seed_pos;
}

void ConvexPlaneExtractor::setMarkerArray(const std::vector<iris_2d::Obstacle>& contours, const grid_map::GridMap& map, const int label)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = map.getFrameId();
    marker.header.stamp = rclcpp::Time(map.getTimestamp());
    marker.id = label;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.05;
    marker.scale.y = 0.01;
    marker.scale.z = 0.01;
    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
    marker.color.a = 1.0;
    marker.lifetime = rclcpp::Duration(0.0);

    for (size_t i=0; i<contours.size(); ++i)
    {
        geometry_msgs::msg::Point point;
        point.x = contours[i].col(0)[0];
        point.y = contours[i].col(0)[1];
        point.z = 1.0;

        marker.points.push_back(point);
    }

    marker_array_.markers.push_back(marker);
}


void ConvexPlaneExtractor::setMarkerArray(const Eigen::VectorXd& seed, const grid_map::GridMap& map, const int label)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = map.getFrameId();
    marker.header.stamp = rclcpp::Time(map.getTimestamp());
    marker.id = label+1;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.2;
    marker.scale.y = 0.2;
    marker.scale.z = 0.2;
    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
    marker.color.a = 1.0;
    marker.lifetime = rclcpp::Duration(0.0);

    marker.pose.position.x = seed[0];
    marker.pose.position.y = seed[1];
    marker.pose.position.z = 1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker_array_.markers.push_back(marker);
}

void ConvexPlaneExtractor::setMarkerArray(const Eigen::MatrixXd& C, const Eigen::VectorXd& d, const grid_map::GridMap& map, const int label)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = map.getFrameId();
    marker.header.stamp = rclcpp::Time(map.getTimestamp());
    marker.id = label+2;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;
    marker.lifetime = rclcpp::Duration(0.0);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(C);
    if (eigensolver.info() != Eigen::Success) {
        RCLCPP_ERROR(get_logger(), "failed to get eigen value of C");
        return ;
    }

    Eigen::Vector2d eigen = eigensolver.eigenvalues();
    Eigen::Matrix2d eigen_vec = eigensolver.eigenvectors();

    // RCLCPP_INFO_STREAM(get_logger(), "eigen value: " << eigen);
    // RCLCPP_INFO_STREAM(get_logger(), "eigen vector: " << eigen_vec);

    double rad = std::acos(eigen_vec.col(1).dot(Eigen::Vector2d::UnitX()));
    int first, second;
    if (rad > M_PI_2)
    {
        first = 0;
        second = 1;
        rad = 3*M_PI_2 - rad;
    }
    else 
    {
        first = 1;
        second = 0;
    }

    marker.scale.x = eigen(first);
    marker.scale.y = eigen(second);
    marker.scale.z = 1.0;

    marker.pose.position.x = d[0];
    marker.pose.position.y = d[1];
    marker.pose.position.z = 1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = std::sin(rad/2);
    marker.pose.orientation.w = std::cos(rad/2);

    marker_array_.markers.push_back(marker);
}


}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(convex_plane::ConvexPlaneExtractor)