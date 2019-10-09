// GenHands19H5.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<cmath>
#include<queue>

#include<hdf5.h>
#include<string>
#include<H5Cpp.h>

#include <ctime>
#include <random>
#include <chrono>
#include <fstream>
#define train_phase 
#define train_num 175951

#ifdef train_phase //0-169999
FILE *fout = fopen("../../Task 1/exp/train10fold.txt", "w");
#define rough_bbx_file "../../Task 1/training_bbs.txt"
#define img_cnt 170000
#define train_annotation_file "../../Task 1/training_joint_annotation.txt"
#else
#ifdef val_phase
FILE *fout = fopen("../../Task 1/val.txt", "w");
#define rough_bbx_file "../../Task 1/training_bbs.txt"
#define img_cnt 5951
#define train_annotation_file "../../Task 1/training_joint_annotation.txt"
#else
FILE *fout = fopen("../../Task 1/exp/test10fold.txt", "w");
#define rough_bbx_file "../../Task 1/test_bbs.txt"

#define img_cnt 124999
#endif
#endif 

#define HDF5_DISABLE_VERSION_CHECK 2


//#define N img_cnt
#define N 17000
#define maxlen 111
#define joint_num 21

float gt_joint_3d_global[train_num][joint_num * 3];
float bbx_all[train_num][4];

using namespace std;
using namespace H5;

int t_id_arr[1111111];
char file[maxlen];
int hsh[1111111];
int id_arr[1111111];

//arrays to be written in h5

float imgindex[N][1];
float bbx[N][4];
float gt3d[N][joint_num * 3];
void InitIdArr()
{
	//first auto completion
	//int lb = (img_cnt / N) * N;

	int lb = img_cnt; //124999
	//int ub = (img_cnt / N + 1) * N;  //img_h36m_cnt - 1;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator_rand(seed);
	std::uniform_real_distribution<double> distribution_rand(0.0, 1.0);

	//printf("%d %d\n", lb, ub);
	for (int i = 0; i < lb; i++) t_id_arr[i] = i;

	//add: 124999
	//t_id_arr[124999] = 0; //nothing
	/*for (int j = lb; j < ub; j++)
	{
		//just randomly choose one 
		double rnd_no = distribution_rand(generator_rand) * img_cnt;
		t_id_arr[j] = rnd_no;
	}*/

	//random shuffle
	for (int i = 0; i < lb; i++)
	{
		if (i % 10000 == 0) cout << "Shuffling " << i << "\n";
#ifdef train_phase
		int t = distribution_rand(generator_rand) * lb;
		while (hsh[t]) t = distribution_rand(generator_rand) * lb;
		hsh[t] = 1;
		id_arr[i] = t_id_arr[t];
		//printf("%d\n", id_arr[i]);
#else
#ifdef val_phase
		id_arr[i] = 170000 + t_id_arr[i];
#else
		id_arr[i] = t_id_arr[i];
#endif
#endif
	}

	//id_arr[124999] = t_id_arr[124999];
}




void GenHDF5()
{
	hid_t fileid;
	hid_t imgidsetid, imgidspaceid;
	hid_t bbxid, bbxspaceid;
	hid_t gt3did, gt3dspaceid;
	

	herr_t status;
	fileid = H5Fcreate(file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dimsimgid[2];
	hsize_t dimsbbx[2];
	hsize_t dimsgt3d[2];
	
	dimsimgid[0] = N;
	dimsimgid[1] = 1;

	dimsbbx[0] = N;
	dimsbbx[1] = 4;

	dimsgt3d[0] = N;
	dimsgt3d[1] = joint_num * 3;

	imgidspaceid = H5Screate_simple(2, dimsimgid, NULL);
	imgidsetid = H5Dcreate(fileid, "/image_index", H5T_IEEE_F32LE, imgidspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	gt3dspaceid = H5Screate_simple(2, dimsgt3d, NULL);
	gt3did = H5Dcreate(fileid, "/gt_joint_3d_global", H5T_IEEE_F32LE, gt3dspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	bbxspaceid = H5Screate_simple(2, dimsbbx, NULL);
	bbxid = H5Dcreate(fileid, "/bbx", H5T_IEEE_F32LE, bbxspaceid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	
	H5Dwrite(imgidsetid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, imgindex);
	H5Dwrite(bbxid, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bbx);
	H5Dwrite(gt3did, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, gt3d);
	
	status = H5Dclose(imgidsetid);
	status = H5Sclose(imgidspaceid);
	status = H5Dclose(bbxid);
	status = H5Sclose(bbxspaceid);

	status = H5Dclose(gt3did);
	status = H5Sclose(gt3dspaceid);
	status = H5Fclose(fileid);

}

void LoadInfo()
{
#ifdef train_phase
	ifstream readFile(rough_bbx_file);
#else
	ifstream readFile(rough_bbx_file);
#endif

	//read bbx
	string line;
	string suffix = ".png";
	int index = 0;
	while (getline(readFile, line))
	{
		if (line.empty())
		{
			break;
		}
		index++;

		int rindex = line.find(suffix) + suffix.length();
		string imgname = line.substr(0, rindex);
		//------parse image name to index
		int imgid = -1;
		//e.g. image_D00000001.png
		if (imgname.length() == 19)
		{
			imgid = 0;
			for (int j = 7; j < 15; j++) imgid = imgid * 10 + imgname[j] - '0';
		}
		else continue; //invalid IND

		vector<float> vect;

		line = line.substr(rindex);
		while (line.front() == ' ' || line.front() == '\t')
		{
			line = line.substr(1);
		}
		rindex = 0;
		while (line[rindex] != ' ' && line[rindex] != '\t')
		{
			++rindex;
		}
		//------ First float bbx_x1
		vect.push_back(stof(line.substr(0, rindex)));
		line = line.substr(rindex);
		while (line.front() == ' ' || line.front() == '\t')
		{
			line = line.substr(1);
		}
		rindex = 0;
		while (line[rindex] != ' ' && line[rindex] != '\t')
		{
			++rindex;
		}
		//------ Second float bbx_y1
		vect.push_back(stof(line.substr(0, rindex)));
		line = line.substr(rindex);
		while (line.front() == ' ' || line.front() == '\t')
		{
			line = line.substr(1);
		}
		rindex = 0;
		while (line[rindex] != ' ' && line[rindex] != '\t')
		{
			++rindex;
		}
		//------ Third float bbx_x2
		vect.push_back(stof(line.substr(0, rindex)));
		line = line.substr(rindex);
		while (line.front() == ' ' || line.front() == '\t')
		{
			line = line.substr(1);
		}
		rindex = 0;
		while (line[rindex] != ' ' && line[rindex] != '\t' && line[rindex] != '\0')
		{
			++rindex;
		}
		//------ Fourth float bbx_y2
		vect.push_back(stof(line.substr(0, rindex)));
		if ((index - 1) % 100 == 0)
		{
			cout << "processed " << index << " imgs" << endl;
		}

		//save to array
		bbx_all[index - 1][0] = vect[0];
		bbx_all[index - 1][1] = vect[1];
		bbx_all[index - 1][2] = vect[2];
		bbx_all[index - 1][3] = vect[3];	
	}
	readFile.close();

	//read original annotation
#ifndef test_phase
	FILE *fin_train_annotation = fopen(train_annotation_file, "r");

	char s[maxlen];

	//Note that the joint number is 1-based
	for (int i = 0; i < train_num; i++)
	{
		if (i % 100 == 0) cout << "Reading " << i << "\n";
		fscanf(fin_train_annotation, "%s", s);
		//cout << s << "\n";
		for (int j = 0; j < joint_num; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				fscanf(fin_train_annotation, "%f", &gt_joint_3d_global[i][j * 3 + k]);
				//cout << gt_joint_3d_global[i][j * 3 + k] << " ";
			}
		}
		//cout << "\n";
	}
	cout << "Reading original 3D ground truth annotation on training set Done!!!\n";
	fclose(fin_train_annotation);
#endif

}


void Init(int step)
{
	for (int i = 0; i < N; i++)
	{
		int id = id_arr[step * N + i];
		//cout << i << " " << id << "\n";
		//cout << step << " " << i << "\n";
		for (int j = 0; j < joint_num; j++)
		{
			for (int k = 0; k < 3; k++)
			{
#ifndef test_phase
				gt3d[i][j * 3 + k] = gt_joint_3d_global[id][j * 3 + k];
#else
				//dummy
				if (k == 2) gt3d[i][j * 3 + k] = 500; // only mean_depth
#endif
			}
		}

		bbx[i][0] = bbx_all[id][0];
		bbx[i][1] = bbx_all[id][1];
		bbx[i][2] = bbx_all[id][2];
		bbx[i][3] = bbx_all[id][3];
		imgindex[i][0] = id;
	}
}


int main()
{
	InitIdArr();
	printf("Img number %10d\n", img_cnt);
	LoadInfo();
	
	//for (int step = 0; step < 1; step++)
	for (int step = 0; step < 10; step++)
	{
		cout << step << "\n";
		char file_true[maxlen];

#ifdef train_phase
		sprintf(file, "%s%d%s", "../../Task 1/training/d10fold_", step, ".h5");
		sprintf(file_true, "%s%d%s", "../h5/training/d10fold_", step, ".h5");
#else
#ifdef val_phase
		sprintf(file, "%s%d%s", "../../Task 1/val/d_", step, ".h5");
#else 
		sprintf(file, "%s%d%s", "../../Task 1/test/dn10fold_", step, ".h5");
#endif
#endif
		Init(step);
		printf("loading done %4d\n", step);
		GenHDF5();
		

		fprintf(fout, "%s\n", file_true);

	}
	fclose(fout);
	return 0;
}