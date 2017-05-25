#include <iostream>
#include<fstream>
#include<opencv2/opencv.hpp>
#include<vector>

using namespace std;

extern "C"{
#include <vl/generic.h>
#include <vl/stringop.h>
#include <vl/pgm.h>
#include <vl/sift.h>
#include <vl/getopt_long.h>
#include <vl/slic.h>
};

struct seg_struct
{
	int i;
	int j;
	int value;
	int sflag;
};

void getgradient(cv::Mat mat)
{
	//转换为灰度图片
	cv::Mat gray;
	cv::cvtColor(mat,gray,CV_BGR2GRAY);
    
	//求得x和y方向的一阶微分
	cv::Mat sobelx;
	cv::Mat sobely;
	cv::Sobel(gray,sobelx,CV_32F,1,0,3);
	cv::Sobel(gray,sobely,CV_32F,0,1,3);

	//求得梯度和方向
	cv::Mat norm;
	cv::Mat dir;
	cv::cartToPolar(sobelx,sobely,norm,dir);
	//转换为8位单通道图像进行显示
	double normMax;
    cv::minMaxLoc(norm,NULL,&normMax);
	cv::Mat grad;
	norm.convertTo(grad,CV_8UC1,255.0/normMax,0);


	double dirMax;
	cv::minMaxLoc(dir,NULL,&dirMax);
	cv::Mat angle;
	dir.convertTo(angle,CV_8UC1,255.0/dirMax,0);

	cv::imwrite("grad.bmp",grad);
	
	cv::imwrite("angle.bmp",angle);
	
}
//superpixel segmentation
void segmentation_fun(vl_uint32* segmentation, cv::Mat mat)
{
	//Convert image to one-dimensional array
	float* image = new float[mat.rows*mat.cols*mat.channels()];
	for(int i=0;i<mat.rows;++i)
	{
		for(int j=0;j<mat.cols;j++)
		{
			image[j+mat.cols*i+mat.cols*mat.rows*0]=mat.at<cv::Vec3b>(i,j)[0];
			image[j+mat.cols*i+mat.cols*mat.rows*1]=mat.at<cv::Vec3b>(i,j)[1];
			image[j+mat.cols*i+mat.cols*mat.rows*2]=mat.at<cv::Vec3b>(i,j)[2];
		}
	}
	//the algorithm will store the final segmentation in a one-dimensional array
	vl_size height = mat.rows;
	vl_size width = mat.cols;
	vl_size channels = mat.channels();

	//The region size defines the number of superpixels obtained.
	//Regularization describes a trade-off between the color term and the spatial term
	vl_size region = 15;
	float regularization = 1000;
	vl_size minRegion = 10;

	vl_slic_segment(segmentation,image,width,height,channels,region,regularization,minRegion);

}
void findEndPoint(cv::Mat mat1,vector<cv::Point2i>& pos)
{
	for(int i=1;i<mat1.rows-1;++i)
	{
		for(int j=1;j<mat1.cols-1;++j)
		{
			if(mat1.at<cv::Vec3b>(i,j) == cv::Vec3b(255,255,255))
			{
				int a = 0, b = 0, c = 0, d = 0;
				int count = 1;
				if (mat1.at<cv::Vec3b>(i-1, j-1)[2] == 255)	{count++; a = 1; c =1; }
				if (mat1.at<cv::Vec3b>(i-1, j  )[2] == 255)	{count++; a = 1;}
				if (mat1.at<cv::Vec3b>(i-1, j+1)[2] == 255)	{count++; a = 1; d =1; }
				if (mat1.at<cv::Vec3b>(i  , j-1)[2] == 255)	{count++; c =1; }
				if (mat1.at<cv::Vec3b>(i  , j+1)[2] == 255)	{count++; d =1; }
				if (mat1.at<cv::Vec3b>(i+1, j-1)[2] == 255)	{count++; b = 1; c =1; }
				if (mat1.at<cv::Vec3b>(i+1, j  )[2] == 255)	{count++; b = 1; }
				if (mat1.at<cv::Vec3b>(i+1, j+1)[2] == 255)	{count++; b = 1; d =1; }
				if (count > 1 && !((a == 1 && b == 1) || (c == 1 && d == 1))) {
					cv::Point2i temp;
					temp.x = j;
					temp.y = i;
					pos.push_back(temp);
				}
			}
		}
	}
}
void candidate_area(cv::Point2i point,vector<cv::Point2i> candiante_point_vec,cv::Mat mat,vl_uint32* segmentation)
{
	for(int i=0;i<mat.rows;i++)
	{
		for(int j=0;j<mat.cols;j++)
		{
			if((int)segmentation[j+mat.cols*i]==(int)segmentation[point.x+mat.cols*point.y]/*&&mat.at<cv::Vec3b>(i,j)==cv::Vec3b(255,255,255)*/)
			{
				cv::Point2i point_temp;
				point_temp.x = j;
				point_temp.y = i;
				candiante_point_vec.push_back(point_temp);
			}
		}
	}
}
int main()
{
	
	cv::Mat mat = cv::imread("image_0.bmp");
	//获得梯度
	getgradient(mat);
	cv::Mat mat_1 = cv::imread("image_1.bmp");
	vector<cv::Point2i> endpoint_vec;
	//获得断点
	findEndPoint(mat_1,endpoint_vec);
	//超像素分割
	vl_uint32* segmentation = new vl_uint32[mat.rows*mat.cols];
	segmentation_fun(segmentation,mat);
	vector<cv::Point2i> shadow_edge_v;
	vector<int> value_vec;
	//将确定是阴影边的点存入vector中（这一步是否有用待定）***********
	for(int i=0;i<mat_1.rows;i++)
	{
		for(int j=0;j<mat_1.cols;j++)
		{
			if(mat_1.at<cv::Vec3b>(i,j)==cv::Vec3b(255,255,255))
			{
			
				cv::Point2i point_temp;
				point_temp.x = j;
				point_temp.y = i;
				value_vec.push_back((int)segmentation[j+mat.cols*i]);
				shadow_edge_v.push_back(point_temp);
			}
		}
	}
	//除去多余的重复的value
	sort(value_vec.begin(),value_vec.end());
	std::vector<int>::iterator it;
	it = std::unique(value_vec.begin(),value_vec.end());
	value_vec.resize(std::distance(value_vec.begin(),it));

	//此时的是未确定的像素(这里是一个区域）
	vector<cv::Point2i> unknown_edge_vec;
	vector<cv::Point2i> shadowedge_candian_vec;
	for(int k=0;k<value_vec.size();k++)
	{
		for(int i=0;i<mat.rows;i++)
		{
			for(int j=0;j<mat.cols;j++)
			{
				if((int)segmentation[j+mat.cols*i]==value_vec[k])
				{
					
					cv::Point2i point_temp;
					point_temp.x = j;
					point_temp.y = i;
					unknown_edge_vec.push_back(point_temp);
				}
				
			}
		}
	}
	//根据邻域得到最后的候选点（就是邻域内如果包含不同的超像素就应该是最后的候选点）
	for(int i = 0;i<unknown_edge_vec.size();i++)
	{
		bool iscannel = false;
		int height = mat.rows;
		int width = mat.cols;
		if(unknown_edge_vec[i].y-1>0&&unknown_edge_vec[i].y+1<height&&unknown_edge_vec[i].x-1>0&&unknown_edge_vec[i].x+1<width)
		{
			int prevalue = (int)segmentation[unknown_edge_vec[i].x+mat.cols*unknown_edge_vec[i].y];
			for(int k=-1;k<2;k++)
			{
				bool cannel  = false;
				for(int j=-1;j<2;j++)
				{
					if((int)segmentation[(unknown_edge_vec[i].x+k)+mat.cols*(unknown_edge_vec[i].y+j)]!=prevalue)
					{
						cannel = true;
						break;
					}
					
				}
				if(cannel)
				{
					iscannel = true;
					break;
				}
			}
			if(iscannel)
			{
				
				cv::Point2i point_temp;
				point_temp.x = unknown_edge_vec[i].x;
				point_temp.y = unknown_edge_vec[i].y;
				shadowedge_candian_vec.push_back(point_temp);
			}
		}
	}
	//查找端点的区域，待定***************
	vector<cv::Point2i> candidate_vec;
	for(int i=0;i<endpoint_vec.size();i++)
	{
		candidate_area(endpoint_vec[i],candidate_vec,mat,segmentation);
	}
	//////////////////////////////测试分类
	cv::Mat image = cv::imread("image_0.bmp");
	for(int i=0;i<candidate_vec.size();i++)
	{
		image.at<cv::Vec3b>(candidate_vec[i].y,candidate_vec[i].x) = cv::Vec3b(255,255,255);
	}
	cv::imshow("image",image);
	cv::waitKey(0);
	/////////////////////////////测试结束
	//全局优化框架来链接点
	{

	}
	
	////////////////////////测试初始分类
	
	/*for(int j=0;j<unknown_edge_vec.size();j++)
	{
	mat.at<cv::Vec3b>(unknown_edge_vec[j].i,unknown_edge_vec[j].j) = cv::Vec3b(0,0,255);
	}
	for(int i=0;i<shadow_edge_v.size();i++)
	{
	mat.at<cv::Vec3b>(shadow_edge_v[i].i,shadow_edge_v[i].j) = cv::Vec3b(255,255,255);
	}*/
	for(int i=0;i<shadowedge_candian_vec.size();i++)
	{
		mat.at<cv::Vec3b>(shadowedge_candian_vec[i].y,shadowedge_candian_vec[i].x) = cv::Vec3b(255,255,255);
	}
	/*for(int k=0;k<nonshadowedge_vec.size();k++)
	{
		mat.at<cv::Vec3b>(nonshadowedge_vec[k].i,nonshadowedge_vec[k].j) = cv::Vec3b(0,0,0);
	}*/

	cv::imshow("mat",mat);
	cv::waitKey(0);
	cv::imwrite("mat.bmp",mat);
	///////////////////////测试结束
	
     ///////////////////////输出
	ofstream seg_vec_file;
	seg_vec_file.open("seg_vec_file.txt");
	for(int i=0;i<shadow_edge_v.size();i++)
	{
		seg_vec_file<<shadow_edge_v[i].y<<" "<<shadow_edge_v[i].x<<endl;
	}
	seg_vec_file.close();
	///////////////////////输出结束


	///////////////////////输出
	for(int i=0;i<value_vec.size();i++)
	{
		cout<<value_vec[i]<<endl;
	}
	///////////////////////输出结束
	system("pause");
    return 0;
}