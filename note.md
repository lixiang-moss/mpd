cd /workspaces/MPDLX-B-new/mpd-splines-public/scripts/inference
source ../../set_env_variables.sh
python inference.py


cd /workspaces/MPDLX-B-new/mpd-splines-public
source set_env_variables.sh
cd scripts/inference
python inference.py   # 保持默认 planner_alg，不含 rrtconnect


cd /workspaces/MPDLX-B-new/mpd-splines-public/scripts/inference
python inference.py run_evaluation_issac_gym=True render_isaacgym_movie=True render_joint_space_time_iters=False


DDIM
----------------METRICS----------------
t_inference_total: 2.077 sec
t_generator: 0.073 sec
t_guide: 1.998 sec
isaacgym_statistics:
DotMap()
metrics:
{'trajs_all': {'collision_intensity': array(0.005, dtype=float32),
               'ee_pose_goal_error_orientation_norm_mean': array(1.364, dtype=float32),
               'ee_pose_goal_error_orientation_norm_std': array(0.719, dtype=float32),
               'ee_pose_goal_error_position_norm_mean': array(0.02, dtype=float32),
               'ee_pose_goal_error_position_norm_std': array(0.01, dtype=float32),
               'fraction_valid': 0.85,
               'fraction_valid_no_joint_limits_vel_acc': 0.85,
               'success': 1,
               'success_no_joint_limits_vel_acc': 1},
 'trajs_best': {'ee_pose_goal_error_orientation_norm': array(0.688, dtype=float32),
                'ee_pose_goal_error_position_norm': array(0.002, dtype=float32),
                'path_length': array(6.992, dtype=float32),
                'smoothness': array(57.6, dtype=float32)},
 'trajs_valid': {'diversity': array(85., dtype=float32),
                 'ee_pose_goal_error_orientation_norm_mean': array(1.319, dtype=float32),
                 'ee_pose_goal_error_orientation_norm_std': array(0.669, dtype=float32),
                 'ee_pose_goal_error_position_norm_mean': array(0.019, dtype=float32),
                 'ee_pose_goal_error_position_norm_std': array(0.01, dtype=float32),
                 'path_length_mean': array(8.245, dtype=float32),
                 'path_length_std': array(1.092, dtype=float32),
                 'smoothness_mean': array(78.746, dtype=float32),
                 'smoothness_std': array(26.033, dtype=float32)}}






(mpd-splines-public) vscode@443db72e513d:/workspaces/MPDLX-B-new$ cd /workspaces/MPDLX-B-new/mpd-splines-public/scripts/inference
(mpd-splines-public) vscode@443db72e513d:/workspaces/MPDLX-B-new/mpd-splines-public/scripts/inference$ source ../../set_env_variables.sh
(mpd-splines-public) vscode@443db72e513d:/workspaces/MPDLX-B-new/mpd-splines-public/scripts/inference$ python inference.py
Importing module 'gym_38' (/workspaces/MPDLX-B-new/mpd-splines-public/deps/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so)
Setting GYM_USD_PLUG_INFO_PATH to /workspaces/MPDLX-B-new/mpd-splines-public/deps/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugInfo.json
pybullet build time: Nov 28 2023 23:51:11
<frozen importlib._bootstrap>:219: RuntimeWarning: to-Python converter for std::vector<unsigned long, std::allocator<unsigned long> > already registered; second conversion method ignored.
<frozen importlib._bootstrap>:219: RuntimeWarning: to-Python converter for std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > already registered; second conversion method ignored.
<frozen importlib._bootstrap>:219: RuntimeWarning: to-Python converter for std::vector<int, std::allocator<int> > already registered; second conversion method ignored.
<frozen importlib._bootstrap>:219: RuntimeWarning: to-Python converter for std::vector<double, std::allocator<double> > already registered; second conversion method ignored.
<frozen importlib._bootstrap>:219: RuntimeWarning: to-Python converter for Eigen::Ref<Eigen::Matrix<double, -1, -1, 0, -1, -1>, 0, Eigen::OuterStride<-1> > already registered; second conversion method ignored.
<frozen importlib._bootstrap>:219: RuntimeWarning: to-Python converter for Eigen::Ref<Eigen::Matrix<double, -1, -1, 0, -1, -1> const, 0, Eigen::OuterStride<-1> > already registered; second conversion method ignored.
<frozen importlib._bootstrap>:219: RuntimeWarning: to-Python converter for Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1>, 0, Eigen::InnerStride<1> > already registered; second conversion method ignored.
<frozen importlib._bootstrap>:219: RuntimeWarning: to-Python converter for Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > already registered; second conversion method ignored.
PyTorch version 2.0.0+cu118
Device count 1
/workspaces/MPDLX-B-new/mpd-splines-public/deps/isaacgym/python/isaacgym/_bindings/src/gymtorch
Using /home/vscode/.cache/torch_extensions/py38_cu118 as PyTorch extensions root...
Emitting ninja build file /home/vscode/.cache/torch_extensions/py38_cu118/gymtorch/build.ninja...
Building extension module gymtorch...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module gymtorch...

-------------------------------------------------------------------------------------------------
cfg_inference_path:
./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml
Model:
/workspaces/MPDLX-B-new/data_public/data_trained_models/launch_train_diffusion_models-v04_2024-09-18_08-12-54/generative_model_class___GaussianDiffusionModel/dataset_subdir___EnvSpheres3D-RobotPanda-joint_joint-one-RRTConnect/dataset_file_merged___dataset_merged_doubled.hdf5/context_ee_goal_pose___True/unet_input_dim___32/unet_dim_mults_option___1/context_q_out_dim___128/context_ee_goal_pose_out_dim___128/context_combined_out_dim___128/bspline_num_control_points_desired___24/parametric_trajectory_class___ParametricTrajectoryBspline/0
--------------------------------------------------------------------------------------------------
Computing the SDF grid and gradients of FIXED objects took: 0.169 sec
-----------------------------------
Torchkin robot: panda
Num links: 69
DOF: 7

-----------------------------------
Warning: No dynamics information for link: panda_link0, setting all inertial properties to 1.
Warning: No dynamics information for link: panda_link1, setting all inertial properties to 1.
Warning: No dynamics information for link: panda_link2, setting all inertial properties to 1.
Warning: No dynamics information for link: panda_link3, setting all inertial properties to 1.
Warning: No dynamics information for link: panda_link4, setting all inertial properties to 1.
Warning: No dynamics information for link: panda_link5, setting all inertial properties to 1.
Warning: No dynamics information for link: panda_link6, setting all inertial properties to 1.
Warning: No dynamics information for link: panda_link7, setting all inertial properties to 1.
Warning: No dynamics information for link: panda_link8, setting all inertial properties to 1.
Warning: No dynamics information for link: panda_hand, setting all inertial properties to 1.
Warning: No dynamics information for link: panda_leftfinger, setting all inertial properties to 1.
Warning: No dynamics information for link: panda_rightfinger, setting all inertial properties to 1.
Warning: No dynamics information for link: tool_link, setting all inertial properties to 1.
--------------- Parametric trajectory -- ParametricTrajectoryBspline
Number of B-spline control points.
        desired          : 24
        adjusted         : 29
        learnable + fixed: 24 + 5


--------------- Loading data
Loading data ...
... done loading data.
Loading data took 3.04 seconds.
TrajectoryDatasetBspline
n_tasks: 1974584
n_trajs: 1974584
control_points_dim: (24, 7)

train_subset size: 1925219
val_subset size  : 49365
argv[0]=
b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
panda_link0b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
panda_link1b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
panda_link2b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
panda_link3b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
panda_link4b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
panda_link5b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
panda_link6b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
panda_link7b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
panda_link8b3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
panda_handb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
panda_leftfingerb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
panda_rightfingerb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1, identity local inertial frameb3Warning[examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp,126]:
tool_link[0, 1, 2, 3, 4, 5, 6]

-------------------------------------------------------------------------------------------------
----------------PLANNING 1/1------------------
--------------------------------------------------------------------------------------------------
Start and goal states are too close. Getting new sample...

----------------START AND GOAL states----------------
q_pos_start: tensor([ 1.2302,  0.2920,  2.6436, -0.6515,  0.1719,  2.6971, -1.8612],
       device='cuda:0')
q_pos_goal: tensor([-2.3215,  0.1834,  2.0889, -1.8095, -0.3883,  2.2543,  2.3621],
       device='cuda:0')
ee_pose_goal: tensor([[-0.1685, -0.8664,  0.4701,  0.5483],
        [-0.9608,  0.0378, -0.2747, -0.1854],
        [ 0.2202, -0.4979, -0.8388,  0.6301]], device='cuda:0')

----------------PLAN TRAJECTORIES----------------
Starting inference...
...inference finished.

----------------METRICS----------------
t_inference_total: 1.037 sec
t_generator: 1.024 sec
t_guide: 0.831 sec
isaacgym_statistics:
DotMap()
metrics:
{'trajs_all': {'collision_intensity': array(0.008, dtype=float32),
               'ee_pose_goal_error_orientation_norm_mean': array(2.394, dtype=float32),
               'ee_pose_goal_error_orientation_norm_std': array(1.177, dtype=float32),
               'ee_pose_goal_error_position_norm_mean': array(0.048, dtype=float32),
               'ee_pose_goal_error_position_norm_std': array(0.036, dtype=float32),
               'fraction_valid': 0.69,
               'fraction_valid_no_joint_limits_vel_acc': 0.69,
               'success': 1,
               'success_no_joint_limits_vel_acc': 1},
 'trajs_best': {'ee_pose_goal_error_orientation_norm': array(0.769, dtype=float32),
                'ee_pose_goal_error_position_norm': array(0.007, dtype=float32),
                'path_length': array(6.618, dtype=float32),
                'smoothness': array(90.22, dtype=float32)},
 'trajs_valid': {'diversity': array(69., dtype=float32),
                 'ee_pose_goal_error_orientation_norm_mean': array(2.249, dtype=float32),
                 'ee_pose_goal_error_orientation_norm_std': array(1.101, dtype=float32),
                 'ee_pose_goal_error_position_norm_mean': array(0.043, dtype=float32),
                 'ee_pose_goal_error_position_norm_std': array(0.021, dtype=float32),
                 'path_length_mean': array(9.326, dtype=float32),
                 'path_length_std': array(1.407, dtype=float32),
                 'smoothness_mean': array(162.368, dtype=float32),
                 'smoothness_std': array(40.407, dtype=float32)}}




---------------METRICS----------------
t_inference_total: 2.963 sec
t_generator: 2.960 sec
t_guide: 2.586 sec
isaacgym_statistics:
DotMap()
metrics:
{'trajs_all': {'collision_intensity': array(0.007, dtype=float32),
               'ee_pose_goal_error_orientation_norm_mean': array(1.48, dtype=float32),
               'ee_pose_goal_error_orientation_norm_std': array(0.788, dtype=float32),
               'ee_pose_goal_error_position_norm_mean': array(0.035, dtype=float32),
               'ee_pose_goal_error_position_norm_std': array(0.025, dtype=float32),
               'fraction_valid': 0.68,
               'fraction_valid_no_joint_limits_vel_acc': 0.68,
               'success': 1,
               'success_no_joint_limits_vel_acc': 1},
 'trajs_best': {'ee_pose_goal_error_orientation_norm': array(0.211, dtype=float32),
                'ee_pose_goal_error_position_norm': array(0.007, dtype=float32),
                'path_length': array(6.332, dtype=float32),
                'smoothness': array(44.411, dtype=float32)},
 'trajs_valid': {'diversity': array(68., dtype=float32),
                 'ee_pose_goal_error_orientation_norm_mean': array(1.479, dtype=float32),
                 'ee_pose_goal_error_orientation_norm_std': array(0.702, dtype=float32),
                 'ee_pose_goal_error_position_norm_mean': array(0.035, dtype=float32),
                 'ee_pose_goal_error_position_norm_std': array(0.017, dtype=float32),
                 'path_length_mean': array(8.945, dtype=float32),
                 'path_length_std': array(1.252, dtype=float32),
                 'smoothness_mean': array(110.909, dtype=float32),
                 'smoothness_std': array(32.088, dtype=float32)}}
5656565656















xhost +local:root



cd /workspaces/MPDLX-B-new/mpd-splines-public/deps/isaacgym/python/examples


1.解决了conda转换问题 2.增加了一些依赖（其实是修补conda） 3.因为docker的文件结构设置错误导致很长一段时间在解决文件路径无法找到的问题 4.存在PB_OMPL库依赖的问题，正在删减对应的代码。 5.正在寻找ROS2/MPD联动的方式 6.PPT正在制作中，有没有什么模板或者要求？ 7待议

1.不同的难度梯度设置的初衷-模仿真实环境，轨迹解从多模态到收敛于唯一解（即环境更严苛） 2探究问题：训练后的泛化性怎么样

！！！！可能要向老师提的问题：泛化性如果不好，我不就要自己训练模型了吗？？

提一下DDPM可能无法参与NFE的比较，因为需要的步数太多

询问ppt用xxx模板是否可以




1探究保存的动画是什么？2为什么没有可视化的轨迹（与pybullt有关？重读MPD）  3.确定MPD接受的信息是什么








给老师展示实验快照的相关参数以说明3060的推理速度完全可以


如果用gazebo大概要：
写一个 ROS2 节点，把 /joint_states + 目标位姿/关节，通过 service / action 转成 MPD 需要的 q_start、goal；

把 Gazebo 里的障碍（model_states）转成 MPD 里用的“球/几何体参数”（EnvSpheres3D 那一套）；

MPD 推理完，把 B 样条轨迹离散成 trajectory_msgs/JointTrajectory，再丢给 ros2_control 或 MoveIt 执行。


工程量可能太大，不稳定：
1.不再用ros2 改用isaac+GYM
2.以有的训练数据是否适用于新的实验场景
3.解决isaac的问题后，可能工程量减小很多，但是对于学习isaac的成本未知
项目中isaac各种指标已经写好了，如果是这样，so101是否还有意义？还是用panda

isaac的不确定性：
如果不用gazebo而是isaac要做的事情：解决目前的一些问题，进行相应的issac学习，通过相应的标的来创建一些新的合适的实验场景

询问ai，新场景是否适用：ai回答场景只要没有新物品问题不大
可以在isaac中增加一些新的衡量指标（具体已有哪些指标需要看代码问ai）

项目观察结论:cost不是isaac给MPD的而是自己先写进去的
找一下还有哪些采样器可以进行替换以弥补工作量。搞清MPD和采样器接口的具体形式，参数
确实MPD用的是否是我以为的isaac

同一个3D球障碍场景只针对规划难度进行位置变化（如球之前从宽阔到狭窄，到唯一解 ：还是要问ai和论文）

这是我在从复现MPD+isaac的论文项目在docker中到转换成在MPD+gazebo然后前期调研发现不行转而尝试回到mpd+isaac的过程中的一系列琐碎的备忘录。请你帮我整理这些文字




除了项目，还有isaac的docker
排查出了isaac的问题（有gui界面但是黑屏）
    1.有可能是起GPU Pepline在docker中无法正常运行，换成在宿主机中运行 
    2.显卡版本太高换成535（此问题可能性较小

排除了pybullt的相关问题：

‘’‘
planning_task.parametric_trajectory 没有 dt，在播放 PyBullet 轨迹时取这个属性就报错了。已在基础类里补上 dt（从时间网格前两点相减得到步长），这样推理时的 sleep_time 不再抛异常。

修改文件：
mpd/parametric_trajectory/trajectory_base.py：
在line 34 初始化时新增 self.dt = float((self.phase_time.t[1] - self.phase_time.t[0]).item())
现在在同样命令运行推理，不会再出现 ParametricTrajectoryBspline 没有 dt 的错误

’‘’

关于这一项render_joint_space_env_iters: bool = True
由于显存要求过大所以无法开启
