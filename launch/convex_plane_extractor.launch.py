from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
        params = data['/**']['ros__parameters']
    return params

def generate_launch_description():
    # param_file = os.path.join(
    #     get_package_share_directory("convex_plane_extractor"), 
    #     'config',
    #     'param.yaml')
    # params = read_yaml(param_file)

    container = ComposableNodeContainer(
        name='cpe_container',
        namespace='',
        package='rclcpp_components', 
        executable='component_container', 
        composable_node_descriptions=[
            ComposableNode(
                package='convex_plane_extractor', 
                plugin='convex_plane::ConvexPlaneExtractor', 
                name='convex_plane_extractor_node',
                remappings=[("convex_plane_extractor/input/grid_map", "/labeled_map"), ("convex_plane_extractor/output/grid_map", "/extract_map")],
                # parameters=[params],
            )
        ],
        output='screen',
    )

    return LaunchDescription([
        container,
    ])