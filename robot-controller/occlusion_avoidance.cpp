/**************************************************************************

Copyright:BUAA Biologically Inspired Mobile Robot Labratory

Author: Zhuo Yijiang

Date:2022-05-29

Description:Provide  functions  of occlusion avoidance

**************************************************************************/

#include "occlusion_avoidance.h"
#include <filesystem>
#include "pcl/io/pcd_io.h"

#include <chrono>
#include <cstdlib>
#include <direct.h>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <windows.h>

namespace
{
    const int kJointCount = 6;
    const char* kJointNames[kJointCount] = {
        "arm_joint_1", "arm_joint_2", "arm_joint_3", "tilt", "pan", "roll"
    };

    std::mutex g_log_mutex;
    std::ofstream g_event_log;
    std::ofstream g_joint_log;

    std::chrono::steady_clock::time_point g_program_start;
    std::chrono::steady_clock::time_point g_avoidance_start;
    std::chrono::steady_clock::time_point g_last_motion_sample;
    bool g_initialized = false;
    bool g_avoidance_active = false;
    int g_avoidance_id = 0;

    bool g_has_prev_joint = false;
    bool g_has_prev_velocity = false;
    double g_prev_command[kJointCount] = { 0 };
    double g_prev_actual[kJointCount] = { 0 };
    double g_prev_actual_velocity[kJointCount] = { 0 };
    std::chrono::steady_clock::time_point g_prev_joint_time;

    bool DirectoryExists(const std::string& path)
    {
        struct _stat info;
        return _stat(path.c_str(), &info) == 0 && (info.st_mode & _S_IFDIR);
    }

    void MakeDirectories(const std::string& path)
    {
        if (path.empty() || DirectoryExists(path))
            return;

        std::string current;
        for (size_t i = 0; i < path.size(); ++i)
        {
            current.push_back(path[i]);
            if (path[i] == '\\' || path[i] == '/')
            {
                if (current.size() > 1 && current[current.size() - 2] != ':')
                    _mkdir(current.c_str());
            }
        }
        _mkdir(path.c_str());
    }

    std::string JoinPath(const std::string& left, const std::string& right)
    {
        if (left.empty())
            return right;
        const char last = left[left.size() - 1];
        if (last == '\\' || last == '/')
            return left + right;
        return left + "\\" + right;
    }

    std::string ExeDirectory()
    {
        char path[MAX_PATH] = { 0 };
        GetModuleFileNameA(NULL, path, MAX_PATH);
        std::string full_path(path);
        const size_t pos = full_path.find_last_of("\\/");
        if (pos == std::string::npos)
            return ".";
        return full_path.substr(0, pos);
    }

    std::string NowStampForFile()
    {
        time_t raw_time = time(NULL);
        tm local_time;
        localtime_s(&local_time, &raw_time);
        char buffer[32] = { 0 };
        strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &local_time);
        return buffer;
    }

    std::string NowText()
    {
        time_t raw_time = time(NULL);
        tm local_time;
        localtime_s(&local_time, &raw_time);
        char buffer[32] = { 0 };
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &local_time);
        return buffer;
    }

    std::string RelativeTimeText()
    {
        const auto now = std::chrono::steady_clock::now();
        const long long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_program_start).count();
        const long long hours = total_ms / 3600000;
        const long long minutes = (total_ms % 3600000) / 60000;
        const long long seconds = (total_ms % 60000) / 1000;
        const long long ms = total_ms % 1000;

        std::ostringstream oss;
        oss << std::setfill('0') << std::setw(2) << hours << ":"
            << std::setw(2) << minutes << ":"
            << std::setw(2) << seconds << "."
            << std::setw(3) << ms;
        return oss.str();
    }

    std::string NumberText(double value)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << value;
        return oss.str();
    }

    std::string MaybeNumberText(bool valid, double value)
    {
        return valid ? NumberText(value) : "NA";
    }

    void WriteAbsoluteStart(std::ofstream& log, const std::string& name)
    {
        log << "log_name=" << name << "\n";
        log << "Beijing absolute start time = " << NowText() << "\n";
    }
}

namespace RunLogger
{
    void Initialize(const std::string& root_dir)
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (g_initialized)
            return;

        g_program_start = std::chrono::steady_clock::now();

        std::string root = root_dir;
        if (root.empty())
        {
            const char* env_root = std::getenv("SSBOT_LOG_ROOT");
            if (env_root != NULL && env_root[0] != '\0')
                root = env_root;
            else
                root = JoinPath(ExeDirectory(), "RunLogs");
        }

        const std::string stamp = NowStampForFile();
        const std::string experiment_dir = JoinPath(root, "experiment_" + stamp);
        MakeDirectories(experiment_dir);

        g_event_log.open(JoinPath(experiment_dir, "RL_experiment.txt").c_str(), std::ios::out);
        g_joint_log.open(JoinPath(experiment_dir, "joint_log.txt").c_str(), std::ios::out);

        WriteAbsoluteStart(g_event_log, "occlusion_avoidance_event");
        g_event_log << "relative_time avoidance_id event occlusion_value camera_x camera_y camera_z control_cycle_ms failure_reason\n";

        WriteAbsoluteStart(g_joint_log, "joint_motion");
        g_joint_log << "relative_time avoidance_id joint_id command_position actual_position tracking_error command_velocity actual_velocity velocity_error actual_acceleration\n";


        g_initialized = true;
    }

    void Shutdown()
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (!g_initialized)
            return;

        if (g_avoidance_active && g_event_log.is_open())
            g_event_log << RelativeTimeText() << " " << g_avoidance_id << " END NA NA NA NA NA program_shutdown\n";

        if (g_event_log.is_open())
            g_event_log << "Beijing absolute end time = " << NowText() << "\n";
        if (g_joint_log.is_open())
            g_joint_log << "Beijing absolute end time = " << NowText() << "\n";

        g_event_log.close();
        g_joint_log.close();
        g_initialized = false;
        g_avoidance_active = false;
    }

    int BeginAvoidance()
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (!g_initialized)
            return 0;

        if (!g_avoidance_active)
        {
            ++g_avoidance_id;
            g_avoidance_active = true;
            g_avoidance_start = std::chrono::steady_clock::now();
            g_last_motion_sample = g_avoidance_start - std::chrono::milliseconds(1000);
            g_has_prev_joint = false;
            g_has_prev_velocity = false;
            if (g_event_log.is_open())
                g_event_log << RelativeTimeText() << " " << g_avoidance_id << " START NA NA NA NA NA NA\n";
        }
        return g_avoidance_id;
    }

    void EndAvoidance()
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (!g_initialized || !g_avoidance_active)
            return;

        if (g_event_log.is_open())
            g_event_log << RelativeTimeText() << " " << g_avoidance_id << " END NA NA NA NA NA NA\n";
        g_avoidance_active = false;
    }

    void LogOcclusion(double occlusion_value, double camera_x, double camera_y,
        double camera_z, double control_cycle_ms)
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (!g_initialized || !g_avoidance_active || !g_event_log.is_open())
            return;

        g_event_log << RelativeTimeText() << " " << g_avoidance_id
            << " SAMPLE " << NumberText(occlusion_value)
            << " " << NumberText(camera_x)
            << " " << NumberText(camera_y)
            << " " << NumberText(camera_z)
            << " " << MaybeNumberText(control_cycle_ms >= 0.0, control_cycle_ms)
            << " NA\n";
    }

    void LogFailure(const std::string& failure_reason)
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (!g_initialized || !g_avoidance_active || !g_event_log.is_open())
            return;

        g_event_log << RelativeTimeText() << " " << g_avoidance_id
            << " FAILURE NA NA NA NA NA " << failure_reason << "\n";
    }

    bool IsAvoidanceActive()
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        return g_initialized && g_avoidance_active;
    }

    bool ShouldSampleMotion()
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (!g_initialized || !g_avoidance_active)
            return false;

        const auto now = std::chrono::steady_clock::now();
        const long long avoidance_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_avoidance_start).count();
        const long long sample_interval_ms = avoidance_ms < 10000 ? 100 : 200;
        const long long since_last_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_last_motion_sample).count();
        return since_last_ms >= sample_interval_ms;
    }

    void LogMotionSample(const std::vector<double>& command_position,
        const std::vector<double>& actual_position)
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (!g_initialized || !g_avoidance_active)
            return;

        const auto now = std::chrono::steady_clock::now();
        const long long avoidance_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_avoidance_start).count();
        const long long sample_interval_ms = avoidance_ms < 10000 ? 100 : 200;
        const long long since_last_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_last_motion_sample).count();
        if (since_last_ms < sample_interval_ms)
            return;

        double dt = 0.0;
        if (g_has_prev_joint)
            dt = std::chrono::duration_cast<std::chrono::duration<double> >(now - g_prev_joint_time).count();

        for (int i = 0; i < kJointCount; ++i)
        {
            const bool has_command = i < static_cast<int>(command_position.size());
            const bool has_actual = i < static_cast<int>(actual_position.size());
            const double command = has_command ? command_position[i] : 0.0;
            const double actual = has_actual ? actual_position[i] : 0.0;
            const double tracking_error = (has_command && has_actual) ? command - actual : 0.0;

            const bool has_velocity = g_has_prev_joint && dt > 0.0 && has_command && has_actual;
            const double command_velocity = has_velocity ? (command - g_prev_command[i]) / dt : 0.0;
            const double actual_velocity = has_velocity ? (actual - g_prev_actual[i]) / dt : 0.0;
            const double velocity_error = has_velocity ? command_velocity - actual_velocity : 0.0;

            const bool has_acceleration = has_velocity && g_has_prev_velocity;
            const double actual_acceleration = has_acceleration ? (actual_velocity - g_prev_actual_velocity[i]) / dt : 0.0;

            if (g_joint_log.is_open())
            {
                g_joint_log << RelativeTimeText() << " " << g_avoidance_id
                    << " " << kJointNames[i]
                    << " " << MaybeNumberText(has_command, command)
                    << " " << MaybeNumberText(has_actual, actual)
                    << " " << MaybeNumberText(has_command && has_actual, tracking_error)
                    << " " << MaybeNumberText(has_velocity, command_velocity)
                    << " " << MaybeNumberText(has_velocity, actual_velocity)
                    << " " << MaybeNumberText(has_velocity, velocity_error)
                    << " " << MaybeNumberText(has_acceleration, actual_acceleration)
                    << "\n";
            }

            g_prev_command[i] = command;
            g_prev_actual[i] = actual;
            g_prev_actual_velocity[i] = actual_velocity;
        }


        g_prev_joint_time = now;
        g_last_motion_sample = now;
        if (g_has_prev_joint)
            g_has_prev_velocity = true;
        g_has_prev_joint = true;
    }
}

using std::string;

extern std::map<std::string, std::string> parameters;

#pragma region 外部引用变量
//机器人坐标变换矩阵
extern Eigen::Isometry3d T10;
extern Eigen::Isometry3d T21;
extern Eigen::Isometry3d T32;
extern Eigen::Isometry3d T43;
extern Eigen::Isometry3d T54;
extern Eigen::Isometry3d T65;
extern Eigen::Isometry3d T73;
extern Eigen::Isometry3d T83;
extern Eigen::Isometry3d T90;
extern Eigen::Isometry3d TA0;

double zoom_ratio = 4;	//相机视野缩放比例,因2.5代机器人相机不支持查询缩放倍数，故将其固定为 2（暂定）
extern std::vector<double> joint_now;
extern std::vector<traj_point>traj;

extern int lc_now;
extern int lc_zero;

extern int ARM_STATE;
extern bool FOCUS_STATE;
extern int MOVE_TYPE;

extern cv::Mat oa_bg;

extern arm_collision_model ACM;
extern std::vector<cv::RotatedRect> obstacles;

extern int rs_end_left, rs_end_right, rs_top_left, rs_top_right;

extern vector<gloRS*> RSs;
#pragma endregion

#pragma region 全局变量
double oa_height = 1.25;    //目标拍摄高度
double d_c2t = -(lc_now + lc_zero) / 1000.0 + oa_height;    //拍摄平面z坐标
double oa_val = 0;  //遮挡值
double oa_threshold = 20;    //遮挡阈值

Eigen::Isometry3d T_3d_cam_left = Eigen::Isometry3d::Identity();   //3D相机0（末端左侧）变换矩阵
Eigen::Isometry3d T_3d_cam_right = Eigen::Isometry3d::Identity();   //3D相机2（末端右侧）变换矩阵
Eigen::Isometry3d T_cam_main = Eigen::Isometry3d::Identity();    //主相机变换矩阵
Eigen::Isometry3d T_cam_inv = Eigen::Isometry3d::Identity();   //主相机逆变换矩阵

Eigen::Vector3d focus_point;    //目标锁定点

int data_num = 0;
int pcd_num = 1;

//强化学习相关
int RL_point_num = 1;  //点云保存
int RL_occlusion_avoid_flag = 1;
static std::wstring policy_model_path()
{
    if (const char* path = std::getenv("AUTOSURGVIEW_MODEL"))
        return std::filesystem::path(path).wstring();
    return std::filesystem::path("robot-controller/models/ppo_AnzhenRobot_20.onnx").wstring();
}
static const std::wstring g_policy_model_path = policy_model_path();
const ORTCHAR_T* modelPath = g_policy_model_path.c_str();
//const ORTCHAR_T* modelPath = L"C:/software/SSBOT2-数据测试/myonnx/RL/ppo_AnzhenRobot_pointcloud.onnx";
// 初始化 ONNX Runtime
// 初始化ONNX Runtime环境
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
// 创建会话配置选项对象
// 用于配置会话的各种参数：
Ort::SessionOptions session_options;
Ort::Session* global_session = nullptr; // 全局指针
std::vector<std::string> input_node_names_alloc; // 用于存储名称字符串
std::vector<const char*> input_node_names;       // 用于传入 Run
std::vector<std::string> output_node_names_alloc;
std::vector<const char*> output_node_names;
#pragma endregion

#pragma region Pyramid类函数
//Pyramid类默认构造函数
Pyramid::Pyramid()
{
    color = Eigen::Vector3d(255, 255, 255);

    center = Eigen::Vector4d(0, 0, 0, 1);
    center_img = Eigen::Vector4d(0, 0, 1, 1);
    
    points.resize(4);

    points[0] = Eigen::Vector4d(0.621, 0.349, 1, 1);
    points[1] = Eigen::Vector4d(-0.621, 0.349, 1, 1);
    points[2] = Eigen::Vector4d(-0.621, -0.349, 1, 1);
    points[3] = Eigen::Vector4d(0.621, -0.349, 1, 1);
}

//Pyramid类构造函数
//@param Ratio 相机缩放比例
//@param Height 目标拍摄高度（机器人坐标系下）
//@param Name 锥体名称
//@param Color 锥体绘制颜色
//@param T 锥体至机器人坐标系变换矩阵
Pyramid::Pyramid(double Ratio, double Height, std::string Name, Eigen::Vector3d Color, Eigen::Isometry3d T)
{
    color = Color;

    center = Eigen::Vector4d(0, 0, 0, 1);
    center_img = Eigen::Vector4d(0, 0, 1, 1);
    
    //根据变换矩阵，将锥体坐标转换至机器人坐标系
    center = T * center;
    center_img = T * center_img;

    points.resize(4);

    T_cam = T;
    ratio = Ratio;
    height = Height;

    name = Name;

    //根据缩放比例，计算角点在锥体坐标系内坐标
    points[0] = Eigen::Vector4d(0.621 / ratio, 0.349 / ratio, 1, 1);
    points[1] = Eigen::Vector4d(-0.621 / ratio, 0.349 / ratio, 1, 1);
    points[2] = Eigen::Vector4d(-0.621 / ratio, -0.349 / ratio, 1, 1);
    points[3] = Eigen::Vector4d(0.621 / ratio, -0.349 / ratio, 1, 1);

    //坐标变换，并将锥体四棱边延长至目标拍摄高度平面
    Eigen::Vector4d point_temp;
    for (auto& point : points)
    {
        point_temp = T * point;
        point = center + (point_temp - center) / (point_temp[2] - center[2]) * (height - center[2]);
        point[3] = 1;
    }

    //计算锥体坐标轴矢量在机器人坐标系下数据
    cal_vec(T);
    //计算用于绘制锥体坐标轴所需点
    axis_points.resize(4);
    axis_points[0] = point_o;
    axis_points[1] = point_o + vec_x;
    axis_points[2] = point_o + vec_y;
    axis_points[3] = point_o + vec_n;
}

Pyramid::~Pyramid()
{

}

//Pyramid类成员函数：设置锥体缩放比例、目标拍摄高度、变换矩阵
//@param Ratio 相机缩放比例
//@param Height 目标拍摄高度（机器人坐标系下）
//@param T 锥体至机器人坐标系变换矩阵
void Pyramid::set_ratio_height(double Ratio, double Height, Eigen::Isometry3d T)
{
    //判断上述参数与锥体现有参数是否相同（避免无用计算）
    if (ratio != Ratio || height != Height || T_cam.data() != T.data())
    {
        //更新参数
        T_cam = T;
        ratio = Ratio;
        height = Height;

        //根据变换矩阵，将锥体坐标转换至机器人坐标系
        center = Eigen::Vector4d(0, 0, 0, 1);
        center_img = Eigen::Vector4d(0, 0, 1, 1);

        center = T * center;
        center_img = T * center_img;

        //根据缩放比例，计算角点在锥体坐标系内坐标
        points[0] = Eigen::Vector4d(0.621 / ratio, 0.349 / ratio, 1, 1);
        points[1] = Eigen::Vector4d(-0.621 / ratio, 0.349 / ratio, 1, 1);
        points[2] = Eigen::Vector4d(-0.621 / ratio, -0.349 / ratio, 1, 1);
        points[3] = Eigen::Vector4d(0.621 / ratio, -0.349 / ratio, 1, 1);

        //坐标变换，并将锥体四棱边延长至目标拍摄高度平面
        Eigen::Vector4d point_temp;
        for (auto& point : points)
        {
            point_temp = T * point;
            point = center + (point_temp - center) / (point_temp[2] - center[2]) * (height - center[2]);
            point[3] = 1;
        }

        //计算锥体坐标轴矢量在机器人坐标系下数据
        cal_vec(T);
        //计算用于绘制锥体坐标轴所需点
        axis_points.resize(4);
        axis_points[0] = point_o;
        axis_points[1] = point_o + vec_x;
        axis_points[2] = point_o + vec_y;
        axis_points[3] = point_o + vec_n;
    }
}

//Pyramid类成员函数：PCL窗口中绘制锥体
//@param viewer 绘制窗口
void Pyramid::draw_pyramid(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
    //绘制四条棱边
    for (int i = 0; i < 4; i++)
    {
        viewer->addLine(pcl::PointXYZ(center[0], center[1], center[2]), pcl::PointXYZ(points[i][0], points[i][1], points[i][2]), 
            color[0], color[1], color[2], name + "_cp" + std::to_string(i));
    }

    //绘制四条底边
    for (int i = 0; i < 4; i++)
    {
        int j = (i + 1) % 4;
        viewer->addLine(pcl::PointXYZ(points[i][0], points[i][1], points[i][2]), pcl::PointXYZ(points[j][0], points[j][1], points[j][2]),
            color[0], color[1], color[2], name + "_p"+ std::to_string(i) +"p" + std::to_string(j));
    }

    //绘制三条坐标轴
    for (int i = 1; i < 4; i++)
    {
        viewer->addLine(pcl::PointXYZ(axis_points[0][0], axis_points[0][1], axis_points[0][2]), pcl::PointXYZ(axis_points[i][0], axis_points[i][1], axis_points[i][2]),
            color[0], color[1], color[2], name + "_axis" + std::to_string(i));
    }
}

//Pyramid类成员函数：PCL窗口中删除锥体
//@param viewer 绘制窗口
void Pyramid::remove_pyramid(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
    //删除四条棱边
    for (int i = 0; i < 4; i++)
    {
        viewer->removeShape(name + "_cp" + std::to_string(i));
    }

    //删除四条底边
    for (int i = 0; i < 4; i++)
    {
        int j = (i + 1) % 4;
        viewer->removeShape(name + "_p" + std::to_string(i) + "p" + std::to_string(j));
    }

    //删除三条坐标轴
    for (int i = 1; i < 4; i++)
    {
        viewer->removeShape(name + "_axis" + std::to_string(i));
    }
}

//Pyramid类成员函数：判断一点是否在锥体内部
//@param v 坐标点
//@return true if v is in the pyramid
bool Pyramid::is_point_in(Eigen::Vector4d v)
{
    //将坐标点v与锥体中心连线，延伸射线至目标拍摄平面（即投影至目标拍摄平面）
    v = center + (points[0][2] - center[2]) / (v[2] - center[2]) * (v - center);

    //判断投影点是否在锥体底面四边形内部
    //采用射线延伸法判断
    int num = 0;
    for (int i = 0; i < 4; i++)
    {
        int ind1 = i;
        int ind2 = (i + 1) % 4;

        if ((v[1] - points[ind2][1]) * (v[1] - points[ind1][1]) <= 0 && v[1] != points[ind2][1])
        {
            double x_hat = points[ind1][0] + (points[ind2][0] - points[ind1][0]) * (v[1] - points[ind1][1]) / (points[ind2][1] - points[ind1][1]);
            if (x_hat >= v[0])
            {
                num++;
            }
        }
    }

    if (num % 2 != 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

//Pyramid类成员函数：计算锥体坐标轴矢量
//@param T 变换矩阵
void Pyramid::cal_vec(Eigen::Isometry3d T)
{
    point_o = T * Eigen::Vector4d(0, 0, 0, 1);
    vec_n = T * Eigen::Vector4d(0, 0, 1, 0);
    vec_x = T * Eigen::Vector4d(1, 0, 0, 0);
    vec_y = T * Eigen::Vector4d(0, 1, 0, 0);
}
#pragma endregion


#pragma region 辅助函数
//设置目标拍摄点
void set_focus_point()
{
    get_T(T_3d_cam_left, T_3d_cam_right, T_cam_main);  //更新矩阵
    focus_point = Eigen::Vector3d(T_cam_main.data()[12], T_cam_main.data()[13], T_cam_main.data()[14]); //相机原点
    
    Eigen::Vector3d z_axis; //相机Z轴矢量
    z_axis[0] = T_cam_main.data()[8];
    z_axis[1] = T_cam_main.data()[9];
    z_axis[2] = T_cam_main.data()[10];

    d_c2t = -(lc_now + lc_zero) * 0.001 + oa_height;

    focus_point += (d_c2t - focus_point[2]) / (z_axis[2]) * z_axis; //相机原点延Z轴投影至目标拍摄平面
    
    cout << "set focus_point: [" << focus_point[0] << ", " << focus_point[1] << ", " << focus_point[2] << "]" << endl;

    //data_num++;
    //std::string filePath = "C:/Users/Butel/Desktop/SSBOT2/pointcloud/" + std::to_string(data_num) + ".txt";
    //// 打开文件
    //std::ofstream outFile(filePath);
    //if (outFile.is_open()) {
    //    // 将向量的数据写入文件
    //    outFile << focus_point[0] << " " << focus_point[1] << " " << focus_point[2];
    //    // 关闭文件
    //    outFile.close();
    //    std::cout << "数据已成功保存到 " << filePath << std::endl;
    //}
    //else {
    //    std::cerr << "无法打开文件 " << filePath << std::endl;
    //}
    //pcd_num = 1;
}


//将3D camera点云转化为pcl点云
//@param cloud_in   //锥体内部点云
//@param cloud_out  //锥体外部点云
//@param points     //realsense点云数据
//@param T_3d_cam   //3D camera变换矩阵
//@param pyramid    //视场锥体
void PCL_Conversion(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out, const rs2::points& points, Eigen::Isometry3d& T_3d_cam, Pyramid& pyramid)  //点云分割为锥形内外
{
    double z_min = T_cam_main.data()[14] + d_c2t + 0.2; //最低高度，高于目标拍摄平面0.2m

    double z_max = T_cam_main.data()[14] - 0.05;    //最高高度，低于相机0.05m
   
    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    auto Vertex = points.get_vertices();    //获取realsense点云

    //realsense点云转换至PCL
    Eigen::Vector4d temp;
    Eigen::Vector4d temp_after;
    for (int i = 0; i < points.size(); i += 50)    //每100个点采1个
    {
        //realsense坐标系下
        temp[0] = Vertex[i].x;
        temp[1] = Vertex[i].y;
        temp[2] = Vertex[i].z;
        temp[3] = 1;

        //转换至机器人坐标系下
        temp_after = T_3d_cam * temp;

        if (temp_after[2] >= z_min && temp_after[2] <= z_max
            && abs(temp_after[1]) <= 2.5 && abs(temp_after[0]) <= 2.5)  //判断点云是否在可能发生遮挡的区间内
        {
            if (pyramid.is_point_in(temp_after))    //若点在锥体内
            {
                cloud_in->points.push_back(pcl::PointXYZ(temp_after[0], temp_after[1], temp_after[2])); //加入cloud_in
            }
            else
            {
                cloud_out->points.push_back(pcl::PointXYZ(temp_after[0], temp_after[1], temp_after[2])); //加入cloud_out
            }
        }
    }
}

//将pcl点云保存用于强化学习使用
//@param cloud_RL    //强化学习保存点云
//@param points     //realsense点云数据
//@param T_3d_cam   //3D camera变换矩阵
void PCL_RLSave(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_RL, const rs2::points& points, Eigen::Isometry3d& T_3d_cam)
{
    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    auto Vertex = points.get_vertices();    //获取realsense点云

    //realsense点云转换至PCL
    Eigen::Vector4d temp;
    Eigen::Vector4d temp_after;
    for (int i = 0; i < points.size(); i += 50)    //每80个点采1个
    {
        //realsense坐标系下
        temp[0] = Vertex[i].x;
        temp[1] = Vertex[i].y;
        temp[2] = Vertex[i].z;
        temp[3] = 1;

        //转换至机器人坐标系下
        temp_after = T_3d_cam * temp;

        cloud_RL->points.push_back(pcl::PointXYZ(temp_after[0], temp_after[1], temp_after[2]));
    }
}

//根据实际关节角度更新矩阵
//@param T_3d_cam 3D camera变换矩阵
//@param T_cam_main 相机矩阵
void get_T(Eigen::Isometry3d& T_3d_cam_left, Eigen::Isometry3d& T_3d_cam_right, Eigen::Isometry3d& T_cam_main)
{
    //变换矩阵初始化
    {
        T10 = Eigen::Isometry3d::Identity();
        T21 = Eigen::Isometry3d::Identity();
        T32 = Eigen::Isometry3d::Identity();
        T43 = Eigen::Isometry3d::Identity();
        T54 = Eigen::Isometry3d::Identity();
        T65 = Eigen::Isometry3d::Identity();
        T73 = Eigen::Isometry3d::Identity();
        T83 = Eigen::Isometry3d::Identity();
    }

    //以关节1圆心为原点（高度方向与关节4轴心平齐） T10
    {
        double theta1 = joint_now[0] / 180 * PI;

        T10.rotate(Eigen::AngleAxisd(theta1, Eigen::Vector3d(0, 0, 1)));
    }

    //关节2->关节1 T21
    {
        double theta2 = joint_now[1] / 180 * PI;

        T21.translate(Eigen::Vector3d(0, 0.7, 0));
        T21.rotate(Eigen::AngleAxisd(theta2, Eigen::Vector3d(0, 0, 1)));
    }

    //关节3->关节2 T32
    {
        double theta3 = joint_now[2] / 180 * PI;

        T32.translate(Eigen::Vector3d(0, 0.5, 0));
        T32.rotate(Eigen::AngleAxisd(theta3, Eigen::Vector3d(0, 0, 1)));
    }

    //关节4->关节3 T43
    {
        double theta4 = joint_now[3] / 180 * PI;

        T43.translate(Eigen::Vector3d(0, 0.4, 0));
        T43.rotate(Eigen::AngleAxisd(theta4, Eigen::Vector3d(0, 1, 0)));
    }

    //关节5->关节4 T54
    {
        double theta5 = joint_now[4] / 180 * PI;

        T54.translate(Eigen::Vector3d(0, 0.1118, 0));
        T54.rotate(Eigen::AngleAxisd(theta5, Eigen::Vector3d(1, 0, 0)));
    }

    //相机->关节5 T65
    {
        double theta6 = joint_now[5] / 180 * PI;
        T65.rotate(Eigen::AngleAxisd(PI, Eigen::Vector3d(1, 0, 0)));
        T65.rotate(Eigen::AngleAxisd(theta6, Eigen::Vector3d(0, 0, 1)));
    }

    if (atoi(parameters["rs_tilt"].c_str()))//相机是否倾斜安装
    {
        //3D相机0（末端左侧）->相机 T73
        {
            T73.translate(Eigen::Vector3d(-0.01023, 0.35197, -0.05226));
            T73.rotate(Eigen::AngleAxisd(-PI / 2, Eigen::Vector3d(0, 0, 1)));
            T73.rotate(Eigen::AngleAxisd(PI / 180 * 160.32, Eigen::Vector3d(1, 0, 0)));
            T73.rotate(Eigen::AngleAxisd(-PI / 180 * 9.39, Eigen::Vector3d(0, 1, 0)));
        }

        //3D相机1（末端右侧）->3D相机0（末端左侧） T83
        {
            T83.translate(Eigen::Vector3d(0.01023, 0.35197, -0.05226));
            T83.rotate(Eigen::AngleAxisd(-PI / 2, Eigen::Vector3d(0, 0, 1)));
            T83.rotate(Eigen::AngleAxisd(-PI / 180 * 160.32, Eigen::Vector3d(1, 0, 0)));
            T83.rotate(Eigen::AngleAxisd(-PI / 180 * 9.39, Eigen::Vector3d(0, 1, 0)));
        }
    }
    else
    {
        //3D相机0（末端左侧）->相机 T73
        {
            T73.translate(Eigen::Vector3d(-0.01023, 0.35316, -0.06088));
            T73.rotate(Eigen::AngleAxisd(-PI / 2, Eigen::Vector3d(0, 0, 1)));
            T73.rotate(Eigen::AngleAxisd(PI / 180 * 160, Eigen::Vector3d(1, 0, 0)));
        }

        //3D相机2（末端右侧）->3D相机0（末端左侧） T83
        {
            T83.translate(Eigen::Vector3d(0.01023, 0.35316, -0.06088));
            T83.rotate(Eigen::AngleAxisd(-PI / 2, Eigen::Vector3d(0, 0, 1)));
            T83.rotate(Eigen::AngleAxisd(-PI / 180 * 160, Eigen::Vector3d(1, 0, 0)));
        }
    }

    T_3d_cam_left = T10 * T21 * T32 * T73;
    T_3d_cam_right = T10 * T21 * T32 * T83;

    T_cam_main = T10 * T21 * T32 * T43 * T54 * T65;
}

//根据给定关节角度更新矩阵
//@param T_cam_main 相机矩阵
//@param theta 给定关节角度数组
void get_T(Eigen::Isometry3d& T_cam_main, std::vector<double> theta)
{
    //变换矩阵初始化
    {
        T10 = Eigen::Isometry3d::Identity();
        T21 = Eigen::Isometry3d::Identity();
        T32 = Eigen::Isometry3d::Identity();
        T43 = Eigen::Isometry3d::Identity();
        T54 = Eigen::Isometry3d::Identity();
        T65 = Eigen::Isometry3d::Identity();
    }

    //以关节1圆心为原点（高度方向与关节4轴心平齐） T10
    {
        T10.rotate(Eigen::AngleAxisd(theta[0], Eigen::Vector3d(0, 0, 1)));
    }

    //关节2->关节1 T21
    {
        T21.translate(Eigen::Vector3d(0, 0.7, 0));
        T21.rotate(Eigen::AngleAxisd(theta[1], Eigen::Vector3d(0, 0, 1)));
    }

    //关节3->关节2 T32
    {
        T32.translate(Eigen::Vector3d(0, 0.5, 0));
        T32.rotate(Eigen::AngleAxisd(theta[2], Eigen::Vector3d(0, 0, 1)));
    }

    //关节4->关节3 T43
    {
        T43.translate(Eigen::Vector3d(0, 0.4, 0));
        T43.rotate(Eigen::AngleAxisd(theta[3], Eigen::Vector3d(0, 1, 0)));
    }

    //关节5->关节4 T54
    {
        T54.translate(Eigen::Vector3d(0, 0.1118, 0));
        T54.rotate(Eigen::AngleAxisd(theta[4], Eigen::Vector3d(1, 0, 0)));
    }

    //相机->关节5 T65
    {
        T65.rotate(Eigen::AngleAxisd(PI + theta[5], Eigen::Vector3d(1, 0, 0)));
    }


    T_cam_main = T10 * T21 * T32 * T43 * T54 * T65;
}

//获得点云遮挡值,应用于实时遮挡检测
//@param pyramid 视场锥体
//@param cloud_in 锥体内部点云
//@param src 背景图
double get_occlusion_val(Pyramid& pyramid, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
    cv::Mat& src)
{
    if (cloud_in->empty())  //若锥体内无点云，直接返回0
    {
        return 0;
    }

    double res = 0;

    Eigen::Vector2d vec_temp;
    double v_val = 0;
    

    src *= 0;

    
    for (auto point : cloud_in->points)
    {
        Eigen::Vector4d vec_point;  //由锥体中心指向点point
        vec_point[0] = point.x - pyramid.center[0];
        vec_point[1] = point.y - pyramid.center[1];
        vec_point[2] = point.z - pyramid.center[2];
        vec_point[3] = 0;

        double k = 1 / (vec_point.dot(pyramid.vec_n));
        Eigen::Vector4d vec_project = k * vec_point - pyramid.vec_n;    //将点投影至图像平面

        double x = vec_project.dot(pyramid.vec_x);  //投影点在图像平面的X方向分量
        double y = vec_project.dot(pyramid.vec_y);  //投影点在图像平面的Y方向分量

        //模拟图像平面遮挡情况
        cv::circle(src, cv::Point2d(160 + x * pyramid.ratio * 386.47, 90 + y * pyramid.ratio * 257.88), 1, cv::Scalar(255, 255, 255));

        res += 1 / vec_project.norm();  //遮挡值累加
    }

    return res;
}

//获得点云遮挡值（无绘制功能），应用于遮挡规避搜索
//@param pyramid 视场锥体
//@param cloud_in 锥体内部点云
void get_occlusion_val(Pyramid& pyramid, pcl::PointCloud<pcl::PointXYZ>& cloud_in, pos_with_oa_info& pwoi)
{
    if (cloud_in.empty())  //若锥体内无点云，直接返回0
    {
        pwoi.oa_val = 0;
        return;
    }

    double res = 0;
    Eigen::Vector4d vec = Eigen::Vector4d(0, 0, 0, 0);

    Eigen::Vector2d vec_temp;
    double v_val = 0;


    for (auto point : cloud_in.points)
    {
        Eigen::Vector4d vec_point;  //由锥体中心指向点point
        vec_point[0] = point.x - pyramid.center[0];
        vec_point[1] = point.y - pyramid.center[1];
        vec_point[2] = point.z - pyramid.center[2];
        vec_point[3] = 0;

        double k = 1 / (vec_point.dot(pyramid.vec_n));

        Eigen::Vector4d vec_project = k * vec_point - pyramid.vec_n;    //由图像中心 指向 点point投影至图片点

        //double x = vec_project.dot(pyramid.vec_x);  //投影点在图像平面的X方向分量
        //double y = vec_project.dot(pyramid.vec_y);  //投影点在图像平面的Y方向分量

        double vec_project_norm = vec_project.norm();
        vec_project /= vec_project_norm;
        vec.x() -= vec_project.x() / vec_project_norm;
        vec.y() -= vec_project.y() / vec_project_norm;

        res += 1 / vec_project_norm;  //遮挡值累加
    }

    pwoi.oa_val = res;

    vec /= vec.norm();
    pwoi.oa_vec.x() = vec.x();
    pwoi.oa_vec.y() = vec.y();
}

//获得强化信息点云遮挡值，应用于遮挡规避搜索
//@param pyramid 视场锥体
//@param cloud_in 锥体内部点云
//@param cloud_out 锥体外部点云
//@param pwoi 术野相机位置
void get_occlusion_info_RL(Pyramid& pyramid, pcl::PointCloud<pcl::PointXYZ>& cloud_in,
    pcl::PointCloud<pcl::PointXYZ>& cloud_out, pos_with_oa_info& pwoi)
{
    if (cloud_in.empty())  //若锥体内无点云，直接返回0
    {
        pwoi.oa_val = 0;
        return;
    }

    double res = 0;
    Eigen::Vector2d vec = Eigen::Vector2d(0.0, 0.0);
    Eigen::Vector2d global_vec = Eigen::Vector2d(0.0, 0.0);

    Eigen::Vector2d vec_temp(0.0, 0.0);
    double vec_sum = 0;

    double centroid_x = 0.0;
    double centroid_y = 0.0;
    double centroid_z = 0.0;
    int cloud_in_points_num = 0;

    for (auto point : cloud_in.points)
    {
        Eigen::Vector4d vec_point;  //由锥体中心指向点point
        vec_point[0] = point.x - pyramid.center[0];
        vec_point[1] = point.y - pyramid.center[1];
        vec_point[2] = point.z - pyramid.center[2];
        vec_point[3] = 0;

        centroid_x += point.x;
        centroid_y += point.y;
        centroid_z += point.z;
        cloud_in_points_num++;

        double k = 1 / (vec_point.dot(pyramid.vec_n));

        Eigen::Vector4d vec_project = k * vec_point - pyramid.vec_n;    //由图像中心 指向 点point投影至图片点

        vec_temp.x() = vec_project.x();
        vec_temp.y() = vec_project.y();

        double vec_project_norm = vec_project.norm();
        double vec_temp_norm = vec_temp.norm();

        vec.x() -= vec_temp.x() / vec_temp_norm * vec_project_norm;
        vec.y() -= vec_temp.y() / vec_temp_norm * vec_project_norm;

        res += 1 / vec_project_norm;  //遮挡值累加
        vec_sum += vec_project_norm;
    }

    pwoi.oa_val = res;
    global_vec = vec;
    vec /= vec_sum;
    pwoi.oa_RL_vec = vec / vec.norm();

    centroid_x /= cloud_in_points_num;
    centroid_y /= cloud_in_points_num;
    centroid_z /= cloud_in_points_num;
    vec_temp.x() = centroid_x - focus_point[0];
    vec_temp.y() = centroid_y - focus_point[1];
    pwoi.global_oa_vec = vec_temp;

    for (auto point : cloud_out.points)
    {
        Eigen::Vector4d vec_point;  //由锥体中心指向点point
        vec_point[0] = point.x - pyramid.center[0];
        vec_point[1] = point.y - pyramid.center[1];
        vec_point[2] = point.z - pyramid.center[2];
        vec_point[3] = 0;

        double k = 1 / (vec_point.dot(pyramid.vec_n));

        Eigen::Vector4d global_vec_project = k * vec_point - pyramid.vec_n;    //由图像中心 指向 点point投影至图片点

        vec_temp.x() = global_vec_project.x();
        vec_temp.y() = global_vec_project.y();

        double  global_vec_project_norm = global_vec_project.norm();
        double vec_temp_norm = vec_temp.norm();

        global_vec.x() -= vec_temp.x() / vec_temp_norm * global_vec_project_norm;
        global_vec.y() -= vec_temp.y() / vec_temp_norm * global_vec_project_norm;

        vec_sum += global_vec_project_norm;
    }

    global_vec /= vec_sum;
    pwoi.global_oa_vec = global_vec / global_vec.norm();
}

//调用强化学习模型计算新的术野相机位置，应用于遮挡规避搜索
//@param pwoi 术野相机位置
Eigen::Vector2d get_delta_pos_RL(pos_with_oa_info& pwoi, double unreach_num)
{
    Eigen::Vector2d delta_pos(0.0, 0.0);
    double max_position_x = 1.4, min_position_x = -1.4, max_position_y = 1.75, min_position_y = 0.6;
    double max_occlusion_val = 500.0, max_unreach_num = 10.0;

    // 创建ONNX推理会话
    // 参数1: 之前创建的环境对象(env)
    // 参数2: ONNX模型文件路径
    // 参数3: 会话配置选项
    // 说明: 此操作会加载模型到内存并准备执行环境
    Ort::Session session(env, modelPath, session_options);

    // 使用 Allocator 获取输入/输出名称（1.14.0 新API）
    Ort::AllocatorWithDefaultOptions allocator;

    // 获取输入名称
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();

    // 获取输出名称
    Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* output_name = output_name_ptr.get();

    // 定义输入形状
    std::vector<int64_t> input_shape = { 1, 8 };

    // 创建输入张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtDeviceAllocator,
        OrtMemType::OrtMemTypeCPU
    );

    // 准备输入数据
    float x_positoin_range = max_position_x - min_position_x;
    float y_positoin_range = max_position_y - min_position_y;
    float RL_input[9];
    RL_input[0] = (pwoi.position.first - min_position_x) / x_positoin_range;
    RL_input[1] = (pwoi.position.second - min_position_y) / y_positoin_range;
    RL_input[2] = std::min(pwoi.oa_val / max_occlusion_val, 1.0);
    RL_input[3] = pwoi.oa_RL_vec[0];
    RL_input[4] = pwoi.oa_RL_vec[1];
    RL_input[5] = pwoi.global_oa_vec[0];
    RL_input[6] = pwoi.global_oa_vec[1];
    RL_input[7] = pwoi.oa_centroid_vec[0];
    RL_input[8] = pwoi.oa_centroid_vec[1];
    std::vector<float> input_data = { RL_input[0], RL_input[1],RL_input[2], RL_input[3],RL_input[4], RL_input[5],
        RL_input[6], (float)std::min(unreach_num / max_unreach_num, 1.0) };
    std::vector<const char*> input_names = { input_name };
    std::vector<const char*> output_names = { output_name };

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
        );

    // 运行推理（使用新API）
    auto output_tensors = session.Run(
        Ort::RunOptions{ nullptr },
        input_names.data(),      // 输入名称数组
        &input_tensor,           // 输入张量数组
        1,                       // 输入数量
        output_names.data(),    // 输出名称数组
        1                        // 输出数量
    );

    // 获取输出
    float* output_data = output_tensors.front().GetTensorMutableData<float>(); //front获取张量中的第一个元素

    delta_pos[0] = std::max(std::min(output_data[0],1.0f), -1.0f);
    delta_pos[1] = std::max(std::min(output_data[1], 1.0f), -1.0f);

    return delta_pos;
}

//强化学习点云处理函数，将PCL点云处理为强化学习模型的输入格式
// 辅助函数：处理单个 PCL 点云
std::vector<float> ProcessSingleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Eigen::Isometry3d& T_cam_inv,
    int target_point_num) {

    std::vector<float> processed_points(target_point_num * 3);

    // PCL点云判空与获取大小
    if (!cloud || cloud->empty()) {
        std::fill(processed_points.begin(), processed_points.end(), 0.0f);
        return processed_points;
    }

    int num_raw_points = cloud->points.size();

    // 创建索引列表 [0, 1, 2, ..., N-1]
    std::vector<int> indices(num_raw_points);
    std::iota(indices.begin(), indices.end(), 0);

    // 随机数生成器
    std::random_device rd;
    std::mt19937 g(rd());

    // === 关键修改 1: 预先计算目标点在相机坐标系下的位置 ===
    // focus_point 是全局变量，代表世界系下的目标
    Eigen::Vector4d target_world(focus_point[0], focus_point[1], focus_point[2], 1.0);
    Eigen::Vector4d target_cam_4d = T_cam_inv * target_world;
    // 取出 x, y, z (相机系下的目标点)
    float target_cam_x = (float)target_cam_4d[0];
    float target_cam_y = (float)target_cam_4d[1];
    float target_cam_z = (float)target_cam_4d[2];

    Eigen::Vector4d temp;       // 原始点(世界系)
    Eigen::Vector4d temp_cam;   // 变换后的点(相机系)

    if (num_raw_points >= target_point_num) {
        // === 情况A：点数过多，随机采样 ===
        std::shuffle(indices.begin(), indices.end(), g);
        for (int i = 0; i < target_point_num; ++i) {
            int idx = indices[i]; // 获取随机打乱后的索引

            // 1. 取出原始点 (世界坐标系)
            temp[0] = cloud->points[idx].x;
            temp[1] = cloud->points[idx].y;
            temp[2] = cloud->points[idx].z;
            temp[3] = 1.0;

            // 2. 变换到相机坐标系 (T_cam_inv * P_world)
            temp_cam = T_cam_inv * temp;

            // 3. 减去相机系下的目标点 (计算相对位置)
            processed_points[i * 3 + 0] = (float)temp_cam[0] - target_cam_x;
            processed_points[i * 3 + 1] = (float)temp_cam[1] - target_cam_y;
            processed_points[i * 3 + 2] = (float)temp_cam[2] - target_cam_z;
        }
    }
    else {
        // === 情况B：点数不足，重复填充 ===
        // 第一步：先拷贝所有原始点
        for (int i = 0; i < num_raw_points; ++i) {
            // 1. 取出原始点
            temp[0] = cloud->points[i].x;
            temp[1] = cloud->points[i].y;
            temp[2] = cloud->points[i].z;
            temp[3] = 1.0;

            // 2. 变换到相机坐标系
            temp_cam = T_cam_inv * temp;

            // 3. 减去相机系下的目标点
            processed_points[i * 3 + 0] = (float)temp_cam[0] - target_cam_x;
            processed_points[i * 3 + 1] = (float)temp_cam[1] - target_cam_y;
            processed_points[i * 3 + 2] = (float)temp_cam[2] - target_cam_z;
        }

        // 第二步：剩余位置随机采样填充
        std::uniform_int_distribution<> dis(0, num_raw_points - 1);
        for (int i = num_raw_points; i < target_point_num; ++i) {
            // 随机选一个已有的点的索引
            int rand_idx = dis(g);

            // 1. 取出原始点
            temp[0] = cloud->points[rand_idx].x;
            temp[1] = cloud->points[rand_idx].y;
            temp[2] = cloud->points[rand_idx].z;
            temp[3] = 1.0;

            // 2. 变换到相机坐标系
            temp_cam = T_cam_inv * temp;

            // 3. 减去相机系下的目标点
            processed_points[i * 3 + 0] = (float)temp_cam[0] - target_cam_x;
            processed_points[i * 3 + 1] = (float)temp_cam[1] - target_cam_y;
            processed_points[i * 3 + 2] = (float)temp_cam[2] - target_cam_z;
        }
    }

    return processed_points;
}

//调用强化学习模型计算新的术野相机位置，应用于遮挡规避搜索
//@param pwoi 术野相机位置
Eigen::Vector2d get_delta_pos_RL_POINT(pos_with_oa_info& pwoi, double unreach_num, 
    std::vector<float>& point_cloud_in_data, std::vector<float>& point_cloud_out_data)
{
    // 确保模型已初始化
    if (global_session == nullptr) 
        InitRLModel();

    Eigen::Vector2d delta_pos(0.0, 0.0);
    double max_position_x = 1.4, min_position_x = -1.4, max_position_y = 1.75, min_position_y = 0.6;
    double max_occlusion_val = 500.0, max_unreach_num = 5.0;
    int target_point_in_num = 256, target_point_out_num = 512;

    // 定义输入形状
    std::vector<int64_t> point_cloud_in_shape = { 1, target_point_in_num, 3 };
    std::vector<int64_t> point_cloud_out_shape = { 1, target_point_out_num, 3 };
    std::vector<int64_t> robot_state_shape = { 1, 6 }; 

    // 准备机器人状态输入数据
    float x_positoin_range = max_position_x - min_position_x;
    float y_positoin_range = max_position_y - min_position_y;
    float RL_input[6];
    RL_input[0] = (pwoi.position.first - min_position_x) / x_positoin_range;
    RL_input[1] = (pwoi.position.second - min_position_y) / y_positoin_range;
    RL_input[2] = std::min(pwoi.oa_val / max_occlusion_val, 1.0);
    RL_input[3] = pwoi.oa_RL_vec[0];
    RL_input[4] = pwoi.oa_RL_vec[1];
    std::vector<float> robot_state_data = { RL_input[0], RL_input[1],RL_input[2], RL_input[3],RL_input[4],
        (float)std::min(unreach_num / max_unreach_num, 1.0) };

    //创建输入张量
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        point_cloud_in_data.data(), point_cloud_in_data.size(),
        point_cloud_in_shape.data(), point_cloud_in_shape.size()
        ));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        point_cloud_out_data.data(), point_cloud_out_data.size(),
        point_cloud_out_shape.data(), point_cloud_out_shape.size()
        ));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        robot_state_data.data(), robot_state_data.size(),
        robot_state_shape.data(), robot_state_shape.size()
        ));

    Ort::RunOptions run_options;
    float* output_data;
    try {
        // 注意：session.Run 的输入顺序必须与 input_node_names 里的顺序一致
        auto output_tensors = global_session->Run(
            run_options,
            reinterpret_cast<const char* const*>(input_node_names.data()), // 输入名称数组
            input_tensors.data(),    // 输入张量数组
            input_tensors.size(),                       // 输入数量 (现在是 2 个)
            reinterpret_cast<const char* const*>(output_node_names.data()),       // 输出名称数组
            1                        // 输出数量
        );
        //获取输出结果
        output_data = output_tensors[0].GetTensorMutableData<float>();
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Inference Error: " << e.what() << std::endl;
    }

    delta_pos[0] = std::max(std::min(output_data[0], 1.0f), -1.0f);
    delta_pos[1] = std::max(std::min(output_data[1], 1.0f), -1.0f);

    return delta_pos;
}

// 初始化函数（在 main 函数或者程序启动时调用一次）
void InitRLModel() {
    if (global_session != nullptr) return;

    // 1. 创建 Session
    global_session = new Ort::Session(env, modelPath, session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // 2. 处理输入名称 (关键修改：分两步走)
    size_t num_input_nodes = global_session->GetInputCount();

    // 第一步：先把所有名称字符串拷贝下来，存入 input_node_names_alloc
    // 这一步可能会触发 vector 扩容，导致内存地址变动
    input_node_names_alloc.clear();
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = global_session->GetInputNameAllocated(i, allocator);
        input_node_names_alloc.push_back(input_name.get());
    }

    // 第二步：等 vector 稳定了，再统一获取指针
    input_node_names.clear();
    for (size_t i = 0; i < num_input_nodes; i++) {
        input_node_names.push_back(input_node_names_alloc[i].c_str());
    }

    // 3. 处理输出名称 (同样的逻辑)
    size_t num_output_nodes = global_session->GetOutputCount();

    output_node_names_alloc.clear();
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = global_session->GetOutputNameAllocated(i, allocator);
        output_node_names_alloc.push_back(output_name.get());
    }

    output_node_names.clear();
    for (size_t i = 0; i < num_output_nodes; i++) {
        output_node_names.push_back(output_node_names_alloc[i].c_str());
    }
}

//判断函数，对正视目标拍摄点的位置进行遮挡检测
bool judge_init_pos(Pyramid& pyramid, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out,
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in)
{
    pos_with_oa_info pwoi_find;
    Eigen::Vector3d color(0, 255, 0);   //设置探索视场锥体颜色为绿色
    std::pair<double, double> pi = { focus_point(0), focus_point(1) };
    pwoi_find.position = pi;
    pwoi_find.oa_val = 1.0;
    pwoi_find.pos_val = 1.0;
    Eigen::Isometry3d T_cam_main;
    std::vector<double> theta_solv(6);
    pcl::PointCloud<pcl::PointXYZ> cloud_in_temp;

    if (!(solve_focus_kdl(joint_now, pwoi_find.position, focus_point, theta_solv) && !ACM.collision(theta_solv[0] / PI * 180, theta_solv[1] / PI * 180, theta_solv[2] / PI * 180, obstacles)))
        return false;
    get_T(T_cam_main, theta_solv);  //获得theta_solv情况下的相机变换矩阵
    Pyramid pyramid_temp(pyramid.ratio, pyramid.height, "pyramid_temp", color, T_cam_main); //创建临时视场锥体

    for (auto& point : cloud_in->points)
    {
        Eigen::Vector4d vec_point;
        vec_point[0] = point.x;
        vec_point[1] = point.y;
        vec_point[2] = point.z;
        vec_point[3] = 1;

        if (pyramid_temp.is_point_in(vec_point))
        {
            cloud_in_temp.push_back(point);
        }
    }

    for (auto& point : cloud_out->points)
    {
        Eigen::Vector4d vec_point;
        vec_point[0] = point.x;
        vec_point[1] = point.y;
        vec_point[2] = point.z;
        vec_point[3] = 1;

        if (pyramid_temp.is_point_in(vec_point))
        {
            cloud_in_temp.push_back(point);
        }
    }

    get_occlusion_val(pyramid_temp, cloud_in_temp, pwoi_find);
    if (pwoi_find.oa_val < oa_threshold)
        return true;
    else
        return false;
}

#pragma endregion


#pragma region 遮挡规避功能函数
//遮挡规避功能函数 
void occlusion_avoidance()
{
    //点云可视化窗口
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2(new pcl::visualization::PCLVisualizer("pt Viewer"));
   
    rs2::colorizer color_map;   //realsense深度图渲染色彩

    rs2::pointcloud pc; //realsense点云

    //设置pcl窗口参数
    viewer2->addCoordinateSystem(0.1);  //绘制坐标系
    viewer2->setCameraPosition(0, -1, 1, 0, 1, -1, 0, 0, 1);    //设置观测点，观测方向
    viewer2->setBackgroundColor(0, 0, 0);   //设置背景色

    std::string cloud_RL_filepath;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_RL(new pcl::PointCloud<pcl::PointXYZ>);     //点云文件保存，强化学习使用
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);                   //在相机视场内点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_in(new pcl::PointCloud<pcl::PointXYZ>);          //在相机视场内点云——滤波后
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);                  //在相机视场外点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_out(new pcl::PointCloud<pcl::PointXYZ>);         //在相机视场外点云——滤波后

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> yellow(cloud_filtered_in, 0, 255, 0); // cloud_in配套颜色，黄色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> white(cloud_filtered_out, 255, 255, 255); //  cloud_out配套颜色，白色

    get_T(T_3d_cam_left, T_3d_cam_right, T_cam_main);    //获取3D camera变换矩阵，相机变换矩阵
    Pyramid rt_pyramid(zoom_ratio, -(lc_now + lc_zero) * 0.001 + oa_height, "rt_pyramid", Eigen::Vector3d(255, 0, 0), T_cam_main); //创建实时相机视场锥体

    string RL_experiment_path = "logs/RL_experiment_legacy.txt";
    ofstream fout(RL_experiment_path, ios::out | ios::app);
    int RL_experiment_save = 0;

    Eigen::Isometry3d T_cam_main_confront = Eigen::Isometry3d::Identity();    //主相机变换矩阵
    int init_pos_count = 0;
    double unreach_num = 0.0;
    InitRLModel();
    clock_t init_pos_time = clock();
    clock_t RL_start_ov = clock();

    while (true)
    {
        if (FOCUS_STATE)    //若处于目标锁定状态
        {
            d_c2t = -(lc_now + lc_zero) * 0.001 + oa_height;   //实时更新目标拍摄平面高度

            get_T(T_3d_cam_left, T_3d_cam_right, T_cam_main);    //获取实时3D camera变换矩阵，相机变换矩阵

            //清空点云数据
            cloud_in->clear();
            cloud_out->clear();
            cloud_RL->clear();

            rt_pyramid.set_ratio_height(zoom_ratio, d_c2t, T_cam_main); //更新实时视场锥体参数

            // 0号相机，末端左侧
            RSs[rs_end_left]->waitForData();
            auto depth_left = RSs[rs_end_left]->getDepthImage();
            auto depth_left_color = color_map.colorize(depth_left);
            //cv::imshow("cam_0_末端左侧", cv::Mat(cv::Size(depth_left_color.get_width(), depth_left_color.get_height()), CV_8UC3, (void*)depth_left_color.get_data()));

            // 2号相机，末端右侧
            RSs[rs_end_right]->waitForData();
            auto depth_right = RSs[rs_end_right]->getDepthImage();
            auto depth_right_color = color_map.colorize(depth_right);
            //cv::imshow("cam_2_末端右侧", cv::Mat(cv::Size(depth_right_color.get_width(), depth_right_color.get_height()), CV_8UC3, (void*)depth_right_color.get_data()));

            //将realsense深度图转化为realsense点云
            auto points_left = pc.calculate(depth_left);
            auto points_right = pc.calculate(depth_right);

            //将realsense点云转化为pcl点云
            PCL_Conversion(cloud_in, cloud_out, points_right, T_3d_cam_right, rt_pyramid);
            PCL_Conversion(cloud_in, cloud_out, points_left, T_3d_cam_left, rt_pyramid);

            //强化学习点云录入
            /*PCL_RLSave(cloud_RL, points_right, T_3d_cam_right);
            PCL_RLSave(cloud_RL, points_left, T_3d_cam_left);
            cloud_RL->height = 1;
            cloud_RL->width = cloud_RL->size();
            cloud_RL_filepath = "logs/pointcloud_" + std::to_string(data_num) + "_" + std::to_string(pcd_num) + ".pcd";
            pcl::io::savePCDFileASCII(cloud_RL_filepath, *cloud_RL);
            pcd_num++;*/

            //std::cout << "滤波前点云数量: " << cloud_in->size()+ cloud_out->size() << std::endl;

            //体素滤波
            pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
            approximate_voxel_filter.setLeafSize(0.03, 0.03, 0.03);
            approximate_voxel_filter.setInputCloud(cloud_in);
            approximate_voxel_filter.filter(*cloud_filtered_in);

            approximate_voxel_filter.setInputCloud(cloud_out);
            approximate_voxel_filter.filter(*cloud_filtered_out);

            //直通滤波
            pcl::PassThrough<pcl::PointXYZ> Cloud_Filter;
            //x方向
            Cloud_Filter.setInputCloud(cloud_filtered_in);
            Cloud_Filter.setFilterFieldName("x");
            Cloud_Filter.setFilterLimits(-2.0, 2.0);
            Cloud_Filter.filter(*cloud_filtered_in);
            Cloud_Filter.setInputCloud(cloud_filtered_out);
            Cloud_Filter.setFilterFieldName("x");
            Cloud_Filter.setFilterLimits(-2.0, 2.0);
            Cloud_Filter.filter(*cloud_filtered_out);


            //y方向
            Cloud_Filter.setInputCloud(cloud_filtered_in);
            Cloud_Filter.setFilterFieldName("y");
            Cloud_Filter.setFilterLimits(-2.0, 2.0);
            Cloud_Filter.filter(*cloud_filtered_in);
            Cloud_Filter.setInputCloud(cloud_filtered_out);
            Cloud_Filter.setFilterFieldName("y");
            Cloud_Filter.setFilterLimits(-2.0, 2.0);
            Cloud_Filter.filter(*cloud_filtered_out);

            /*std::cout << "滤波后点云数量: " << cloud_filtered_in->size() + cloud_filtered_out->size() << std::endl;
            if (!cloud_filtered_in->empty())
            {
                std::cout << "第一个点的 X 坐标: " << cloud_filtered_in->points[0].x << std::endl;
                std::cout << "第一个点的 Y 坐标: " << cloud_filtered_in->points[0].y << std::endl;
                std::cout << "第一个点的 Z 坐标: " << cloud_filtered_in->points[0].z << std::endl;
            }*/

            ////可视化
            viewer2->removeAllPointClouds();

            viewer2->addPointCloud<pcl::PointXYZ>(cloud_filtered_in, yellow, "cloud_filtered_in");
            viewer2->addPointCloud<pcl::PointXYZ>(cloud_filtered_out, white, "cloud_filtered_out");

            viewer2->removeAllShapes();

            rt_pyramid.draw_pyramid(viewer2);

            viewer2->spinOnce(1);

            //无遮挡回原位置
            //if ((clock() - init_pos_time) / CLOCKS_PER_SEC  > 8 && ARM_STATE == ARM_FREE)
            //{
            //    if (judge_init_pos(rt_pyramid, cloud_filtered_in, cloud_filtered_out) && abs(focus_point(0) - rt_pyramid.center[0])>1e-2 && abs(focus_point(1) - rt_pyramid.center[1]) > 1e-2)
            //    {
            //        Eigen::Vector2d oa_direction;   //相机运动方向
            //        oa_direction[0] = focus_point(0) - rt_pyramid.center[0];
            //        oa_direction[1] = focus_point(1) - rt_pyramid.center[1];
            //        oa_direction /= oa_direction.norm();    //归一化

            //        Eigen::Vector2d occlusion_target;   //相机运动目标位置
            //        occlusion_target[0] = focus_point(0);
            //        occlusion_target[1] = focus_point(1);

            //        if (get_traj_kdl(joint_now, traj, oa_direction, true, focus_point, occlusion_target))
            //        {
            //            ARM_STATE = ARM_GET_TRAJ_SUCCESS;   //若求解笛卡尔直线路径成功，将机械臂状态置为ARM_GET_TRAJ_SUCCESS
            //        }

            //        init_pos_count = 0;
            //        continue;
            //    }
            //    init_pos_time = clock();
            //}

            //获得遮挡值
            oa_val = get_occlusion_val(rt_pyramid, cloud_filtered_in, oa_bg);
            unreach_num = 0.0;
            if (RunLogger::IsAvoidanceActive())
                RunLogger::LogOcclusion(oa_val, rt_pyramid.center[0], rt_pyramid.center[1], rt_pyramid.center[2]);

            if (oa_val < oa_threshold && ARM_STATE == ARM_FOLLOW_TRAJ && MOVE_TYPE == MOVE_TYPE_FOCUS)
            /*if (oa_val < oa_threshold && RL_experiment_save == 1)*/
            {
                //若遮挡值小于阈值，且机械臂在遮挡规避运动过程中，机械臂运动停止，针对遮挡消失情况
                arm_stop();
                if (RL_experiment_save == 1)
                {
                    fout << "遮挡值:" << oa_val << endl;
                    fout << "相机位置:" << rt_pyramid.center[0] << rt_pyramid.center[1] << endl;
                    fout << "遮挡规避时间" << (clock() - RL_start_ov) / (double)CLOCKS_PER_SEC * 1000 << endl;
                    fout << "遮挡规避结束\n" << endl;
                    RL_experiment_save = 0;
                    RunLogger::EndAvoidance();
                }
                //udp_send_debug_info(DEBUG_RL_OCCLUSION_AVOIDANCE_END);
            }
            else if (oa_val >= oa_threshold && RL_occlusion_avoid_flag == 1 && ARM_STATE == ARM_FREE)
            {
                /*udp_send_debug_info(DEBUG_RL_OCCLUSION_AVOIDANCE_START);
                std::this_thread::sleep_for(std::chrono::milliseconds(600)); */

                cout << "oa_val = " << oa_val << endl;
                if (RL_experiment_save == 0)
                {
                    fout << "遮挡规避开始" << endl;
                    RL_start_ov = clock();
                    RL_experiment_save = 1;
                    RunLogger::BeginAvoidance();
                }
                fout << "遮挡值:" << oa_val << endl;

                pos_with_oa_info pwoi_ori;  //创建带有遮挡值的相机位置
                pwoi_ori.position = { rt_pyramid.center[0],rt_pyramid.center[1] };  //确定相机初始位置
                get_occlusion_info_RL(rt_pyramid, *cloud_filtered_in, *cloud_filtered_out, pwoi_ori);

                fout << "相机位置:" << pwoi_ori.position << endl;
                RunLogger::LogOcclusion(oa_val, pwoi_ori.position.first, pwoi_ori.position.second, rt_pyramid.center[2]);

                Eigen::Vector2d delta_pos = Eigen::Vector2d(0, 0);
                //无点云强化学习处理输出
                //delta_pos = get_delta_pos_RL(pwoi_ori, unreach_num);

                //点云输入强化学习处理
                T_cam_inv = T_cam_main.inverse();
                vector<float> point_in_input, point_out_input;
                point_in_input = ProcessSingleCloud(cloud_filtered_in, T_cam_inv, 256);
                point_out_input = ProcessSingleCloud(cloud_filtered_out, T_cam_inv, 512);
                delta_pos = get_delta_pos_RL_POINT(pwoi_ori, unreach_num, point_in_input, point_out_input);

                double step = 0.08;
                if (abs(delta_pos.norm()) < 1e-5)
                {
                    cout << "强化学习模型输出错误" << endl;
                    RunLogger::LogFailure("rl_output_invalid");
                    RunLogger::EndAvoidance();
                }
                else
                {
                    delta_pos = delta_pos / delta_pos.norm() * step;

                    cout << "before:[" << rt_pyramid.center[0] << "," << rt_pyramid.center[1]
                        << "] after:[" << rt_pyramid.center[0] + delta_pos[0] << "," << rt_pyramid.center[1] + delta_pos[1] << "]" << endl;

                    MOVE_TYPE = MOVE_TYPE_FOCUS;    //将机械臂运动类型置为目标锁定

                    Eigen::Vector2d oa_direction;   //相机运动方向
                    oa_direction[0] = delta_pos[0];
                    oa_direction[1] = delta_pos[1];
                    oa_direction /= oa_direction.norm();    //归一化

                    Eigen::Vector2d occlusion_target;   //相机运动目标位置
                    occlusion_target[0] = pwoi_ori.position.first + delta_pos[0];
                    occlusion_target[1] = pwoi_ori.position.second + delta_pos[1];

                    if (get_traj_kdl(joint_now, traj, oa_direction, true, focus_point, occlusion_target))
                    {
                        unreach_num = 0.0;
                        ARM_STATE = ARM_GET_TRAJ_SUCCESS;   //若求解笛卡尔直线路径成功，将机械臂状态置为ARM_GET_TRAJ_SUCCESS
                    }
                    else
                    {
                        unreach_num += 1.0;
                        RunLogger::LogFailure("trajectory_planning_failed");
                        RunLogger::EndAvoidance();
                    }
                }
            }

        }
    }
}
#pragma endregion
