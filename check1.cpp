#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <unistd.h>
#include <bits/stdc++.h>
using namespace std;

// Parameters
const int N=10;
const double Total_mass=1, dt=0.00005, Gravity=9.81, PI=3.14159265;
// const double k_s=1000, k_d=0.5, Total_length=2, Total_width=2;
const double c_x=0, c_y=0, c_z=0, c_r=1;
// Derieved Constants
// const double Mass=Total_mass/((N+2)*(M+2)), distX=Total_length/(M+1), distY=Total_width/(N+1), distXY=sqrt(distX*distX+distY*distY);

Eigen::MatrixXd Pos(N+1,3);
Eigen::MatrixXi F(N,3);

// Eigen::MatrixXd Frc = Eigen::MatrixXd::Zero((N+2)*(M+2),3);
// Eigen::MatrixXd Vel = Eigen::MatrixXd::Zero((N+2)*(M+2),3);

void init_vertices(){
	Pos(0,0)=c_x;
	Pos(0,1)=c_y;
	Pos(0,2)=c_z;
	for(int i=0;i<N;i++){
		Pos(i+1,0)=c_x+c_r*cos(i*2*PI/N);
		Pos(i+1,1)=c_y+c_r*sin(i*2*PI/N);
		Pos(i+1,2)=c_z;
		cout<<i+1<<' '<<Pos(i+1,0)<<' '<<Pos(i+1,1)<<' '<<Pos(i+1,2)<<'\n';
	}
}
void init_faces(){
	for(int i=0;i<N;i++){
		F(i,0)=0;
		F(i,1)=i+1;
		F(i,2)=(i+2)%N;
	}
}

int main(int argc, char *argv[])
{
	init_vertices();
	init_faces();

	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(Pos, F);
	viewer.data().set_face_based(true);

	viewer.launch();
}
