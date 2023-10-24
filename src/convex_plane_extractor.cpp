#include <convex_plane_extractor/convex_plane_extractor.h>

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
}

ConvexPlaneExtractor::~ConvexPlaneExtractor(){}

void ConvexPlaneExtractor::callbackGridMap(const grid_map_msgs::msg::GridMap::UniquePtr msg)
{
    auto begin = std::chrono::high_resolution_clock::now();   
    RCLCPP_INFO(get_logger(), "subscribe map address: 0x%x", &(msg->data));
    // RCLCPP_INFO(get_logger(), "convert form message");
    grid_map::GridMap map;
    grid_map::GridMapRosConverter::fromMessage(*msg, map);

    const grid_map::Matrix labels = map.get("valid_labels");
    std::vector<int> label_list = findLabels(labels);
    RCLCPP_INFO(get_logger(), "label list size: %ld", label_list.size());
    // RCLCPP_INFO(get_logger(), "min label: %d", *std::min_element(label_list.begin(), label_list.end()));
    // RCLCPP_INFO(get_logger(), "max label: %d", *std::max_element(label_list.begin(), label_list.end()));
    std::vector<Eigen::MatrixXd> contours;
    Eigen::Vector3d normal;
    Eigen::Vector2d seed_pos;

    convex_plane_msgs::msg::ConvexPlanesWithGridMap::UniquePtr message;
    for (size_t i=0; i<label_list.size(); ++i)
    {
        iris::IRISProblem problem(2);
        getContours(map, label_list[i], contours, normal, seed_pos);
        problem.setSeedPoint(seed_pos);
        for (Eigen::MatrixXd& obs : contours) problem.addObstacle(obs);
        iris::IRISRegion region(2);
        bool success = iris::inflate_region(problem, options_, region, solver_);

        if (success)
        {
            RCLCPP_INFO(get_logger(), "Convex region is found");
            RCLCPP_DEBUG_STREAM(get_logger(), "C: " << region.ellipsoid.getC());
            RCLCPP_DEBUG_STREAM(get_logger(), "d: " << region.ellipsoid.getD());
            RCLCPP_DEBUG_STREAM(get_logger(), "A: " << region.polyhedron.getA());
            RCLCPP_DEBUG_STREAM(get_logger(), "b: " << region.polyhedron.getB());

            if (!convex_plane::ConvexPlaneConverter::addPlaneToMessage(region, normal, label_list[i], message->plane))
            {
                RCLCPP_ERROR(get_logger(), "ConvexPlanes message is invalid because the sizes of components are different");
            }
            RCLCPP_INFO(get_logger(), "Add Plane to the message");
        }
    }

    // RCLCPP_INFO(get_logger(), "convert to message");
    // grid_map::GridMapRosConverter::toMessage(map, message->map);
    grid_map_msgs::msg::GridMap::UniquePtr out_msg = grid_map::GridMapRosConverter::toMessage(map);

    RCLCPP_INFO(get_logger(), "publish map address: 0x%x", &(out_msg->data));
    pub_map_->publish(std::move(out_msg));
    // pub_plane_with_map_->publish(std::move(message));
    
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(end - begin);
    RCLCPP_INFO(get_logger(), "processing time: %f", elapsed);

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



void ConvexPlaneExtractor::getContours(grid_map::GridMap& map, const int label, std::vector<Eigen::MatrixXd>& contours_matrix, Eigen::Vector3d& normal, Eigen::Vector2d& seed_pos)
{
    // RCLCPP_INFO(get_logger(), "create binary matrix");
    if (!map.exists("binary_matrix"))
    {
        map.add("binary_matrix", map.get("valid_labels"));
    }
    grid_map::Matrix& binary = map.get("binary_matrix");
    const grid_map::Matrix& normal_x = map.get("normal_vectors_x");
    const grid_map::Matrix& normal_y = map.get("normal_vectors_y");
    const grid_map::Matrix& normal_z = map.get("normal_vectors_z");
    
    int labeled_cell_num = 0;
    bool seed_flag = false;
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
            if (!seed_flag)
                if (map.getPosition(index, seed_pos)) seed_flag = true;
            ++labeled_cell_num;
        }
    }
    if (labeled_cell_num == 0)
    {
        RCLCPP_ERROR(get_logger(), "number of cells labeled %f is equal to 0", label);
        rclcpp::shutdown();
    }
    normal /= labeled_cell_num;
    RCLCPP_DEBUG(get_logger(), "normal for label %f: (%lf, %lf, %lf)", label, normal(0), normal(1), normal(2));

    // RCLCPP_INFO(get_logger(), "conver to image");
    cv::Mat label_image;
    grid_map::GridMapCvConverter::toImage<unsigned char, 1>(map, "binary_matrix", CV_8UC1, label_image);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat label_image_copy = label_image;
    cv::findContours(label_image_copy, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
    RCLCPP_INFO(get_logger(), "contour size: %ld", contours.size());

    int point_num = 0;
    for (size_t i=0; i<contours.size(); ++i)
    {
        point_num += contours[i].size();
    }

    // RCLCPP_INFO(get_logger(), "add layer");
    std::string output_layer = "plane_"+std::to_string(static_cast<int>(label));
    grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 1>(label_image, output_layer, map);    

    int contours_index = 0;
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
        }
        contours_index+=contour.size();
    }

    // for (size_t i=0; i<contours_matrix.size(); ++i)
    //     RCLCPP_INFO_STREAM(get_logger(), "obs " << i << ": " << std::endl << contours_matrix[i] << std::endl);

    // RCLCPP_INFO(get_logger(), "col_ind: %d, point num: %d", col_ind, point_num);
    const float min_value = map.get(output_layer).minCoeff();
    const float max_value =  map.get(output_layer).maxCoeff();
    const float median_value = (min_value + max_value)*0.5;

    for (int i=0; i<point_num; ++i)
    {
        if (!map.getIndex(contours_matrix[i].col(0), index))
        {
            RCLCPP_ERROR(get_logger(), "The position (%f, %f) is out of range", contours_matrix[i].col(0)(0), contours_matrix[i].col(0)(1));
            continue;
        }
        map.at(output_layer, index) = median_value;
    }
}

}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(convex_plane::ConvexPlaneExtractor)