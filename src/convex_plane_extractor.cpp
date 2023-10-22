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
    std::vector<float> label_list = findLabels(labels);
    RCLCPP_INFO(get_logger(), "label list size: %ld", label_list.size());
    // RCLCPP_INFO(get_logger(), "min label: %f", *std::min_element(label_list.begin(), label_list.end()));
    // RCLCPP_INFO(get_logger(), "max label: %f", *std::max_element(label_list.begin(), label_list.end()));
    for (size_t i=0; i<label_list.size(); ++i)
        getContours(map, label_list[i]);

    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(end - begin);
    RCLCPP_INFO(get_logger(), "processing time: %f", elapsed);
}

std::vector<float> ConvexPlaneExtractor::findLabels(const grid_map::Matrix& labels)
{
    std::vector<float> label_list(0);
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



bool ConvexPlaneExtractor::getContours(grid_map::GridMap& map, const float label)
{
    // RCLCPP_INFO(get_logger(), "create binary matrix");
    if (!map.exists("binary_matrix"))
    {
        map.add("binary_matrix", map.get("valid_labels"));
    }
    grid_map::Matrix& binary = map.get("binary_matrix");
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
        }
    }

    // RCLCPP_INFO(get_logger(), "conver to image");
    cv::Mat label_image;
    grid_map::GridMapCvConverter::toImage<unsigned char, 1>(map, "binary_matrix", CV_8UC1, label_image);

    // // const grid_map::Matrix& label_mat = map.get("labels");
    // RCLCPP_INFO(get_logger(), "get iterator");
    // grid_map::Index base_ind;
    // for (grid_map::GridMapIterator iter(map); !iter.isPastEnd(); ++iter)
    // {
    //     if (map.at("labels", *iter) == NAN) continue;

    //     if (map.at("labels", *iter) == label)
    //     {
    //         base_ind = iter.getUnwrappedIndex();
    //         break;
    //     }
    // }

    // RCLCPP_INFO(get_logger(), "conver to image");
    // cv::Mat image;
    // std::string input_layer = "labels";
    // grid_map::GridMapCvConverter::toImage<unsigned char, 1>(map, input_layer, CV_8UC1, image);
    // RCLCPP_INFO(get_logger(), "get base value");
    // const int base_value = image.at<cv::Vec<unsigned char, 1>>(base_ind(0), base_ind(1))[0];

    // RCLCPP_INFO(get_logger(), "get binary image");
    // cv::Mat label_image;
    // // set values of cells smaller than label to 0
    // cv::threshold(image, label_image, base_value-1, 255, CV_THRESH_TOZERO | CV_THRESH_OTSU);
    // // set values of cell larger than label to 0
    // cv::threshold(label_image, label_image, base_value+1, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat label_image_copy = label_image;
    cv::findContours(label_image_copy, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    RCLCPP_INFO(get_logger(), "contour size: %ld", contours.size());
    // if (contours.size() != 1)
    // {
    //     RCLCPP_ERROR(get_logger(), "Size of the contour of region labeled %f ", label);
    // }
    for (size_t i=0; i<contours.size(); ++i)
    {
        // RCLCPP_INFO(get_logger(), "Size of contour %d: %ld", i, contours[i].size());
        for (const cv::Point& ponit : contours[i])
        {
            label_image.at<cv::Vec<unsigned char, 1>>(ponit)[0] = 128;
        }
    }


    // RCLCPP_INFO(get_logger(), "add layer");
    std::string output_layer = "plane_"+std::to_string(static_cast<int>(label));
    grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 1>(label_image, output_layer, map);    

    // RCLCPP_INFO(get_logger(), "convert to message");
    grid_map_msgs::msg::GridMap::UniquePtr out_msg = grid_map::GridMapRosConverter::toMessage(map);

    RCLCPP_INFO(get_logger(), "publish map address: 0x%x", &(out_msg->data));
    pub_map_->publish(std::move(out_msg));

    return true;
}

}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(convex_plane::ConvexPlaneExtractor)