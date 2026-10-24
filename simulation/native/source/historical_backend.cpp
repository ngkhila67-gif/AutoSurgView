// Binding only: all FK/IK and collision decisions execute the archived C++ code.
#include "chain.hpp"
#include "chainfksolverpos_recursive.hpp"
#include "chainiksolvervel_pinv.hpp"
#include "chainiksolverpos_nr.hpp"
#include "collision.h"
#include <cmath>

#define API extern "C" __declspec(dllexport)

// Same active segments as AnzhenRobotChain() in myenv/Myclass.py.
static KDL::Chain make_chain() {
    KDL::Chain chain;
    for (double length : {0.7, 0.5, 0.5405, 0.0})
        chain.addSegment(KDL::Segment(KDL::Joint(KDL::Joint::RotZ),
                                    KDL::Frame(KDL::Vector(0.0, length, 0.0))));
    return chain;
}

static KDL::JntArray make_q(const double* degrees) {
    KDL::JntArray q(4);
    // Preserve the original Python environment's numerical conversion constant.
    for (int i=0; i<3; ++i) q(i)=degrees[i]/180.0*3.1415926;
    q(3)=0.0;
    return q;
}

API int nc_fk(const double* degrees, double* xy8) {
    auto chain=make_chain();
    KDL::ChainFkSolverPos_recursive fk(chain);
    auto q=make_q(degrees);
    xy8[0]=xy8[1]=0.0;
    for (int segment=1; segment<=3; ++segment) {
        KDL::Frame frame;
        int status=fk.JntToCart(q,frame,segment);
        if(status<0) return status;
        xy8[2*segment]=frame.p[0]; xy8[2*segment+1]=frame.p[1];
    }
    return 0;
}

API int nc_ik(const double* degrees, const double* xy, double* result4,
              int max_iterations, double tolerance) {
    auto chain=make_chain();
    KDL::ChainFkSolverPos_recursive fk(chain);
    KDL::ChainIkSolverVel_pinv velocity(chain);
    KDL::ChainIkSolverPos_NR position(chain,fk,velocity,max_iterations,tolerance);
    auto q=make_q(degrees);
    KDL::Frame target;
    const int fk_status=fk.JntToCart(q,target);
    if(fk_status<0) return fk_status;
    target.p[0]=xy[0]; target.p[1]=xy[1];
    KDL::JntArray result(4);
    const int status=position.CartToJnt(q,target,result);
    for(int i=0;i<4;++i) result4[i]=result(i);
    return status;
}

API int nc_self_collision(const double* degrees) {
    arm_collision_model model;
    model.arm_set(degrees[0],degrees[1],degrees[2]);
    return model.self_collision()?1:0;
}

API int nc_obstacle_collision(const double* degrees,const double* rectangles,int count) {
    arm_collision_model model;
    model.arm_set(degrees[0],degrees[1],degrees[2]);
    std::vector<cv::RotatedRect> obstacles;
    for(int i=0;i<count;++i) {
        const double* r=rectangles+5*i;
        // Public binding uses metres; archived collision.cpp uses millimetres.
        obstacles.emplace_back(cv::Point2f(float(r[0]*1000),float(r[1]*1000)),
            cv::Size2f(float(r[2]*1000),float(r[3]*1000)),float(r[4]));
    }
    return model.obstacle_collision(obstacles)?1:0;
}

API int nc_footprints(const double* degrees,double* records) {
    arm_collision_model model;
    model.arm_set(degrees[0],degrees[1],degrees[2]);
    int index=0;
    for(int group=0;group<4;++group) for(const auto& node:model.arm[group]) {
        double* r=records+8*index++;
        r[0]=group; r[1]=node.type;
        r[2]=node.center.x/1000; r[3]=node.center.y/1000;
        r[4]=node.radius/1000; r[5]=node.length_a/1000;
        r[6]=node.length_b/1000; r[7]=node.angle;
    }
    return index;
}

API int nc_pair_collision(const double* a,const double* b) {
    auto point=[](const double* p){return cv::Point2d(p[1]*1000,p[2]*1000);};
    if(a[0]==0 && b[0]==0)
        return circle_collision(point(a),a[3]*1000,point(b),b[3]*1000);
    if(a[0]==0)
        return rotated_rectangle_circle_collision(point(b),b[3]*1000,b[4]*1000,b[5],point(a),a[3]*1000);
    if(b[0]==0)
        return rotated_rectangle_circle_collision(point(a),a[3]*1000,a[4]*1000,a[5],point(b),b[3]*1000);
    return rotated_rectangle_collision(point(a),a[3]*1000,a[4]*1000,a[5],point(b),b[3]*1000,b[4]*1000,b[5]);
}
