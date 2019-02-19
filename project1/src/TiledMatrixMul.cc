//#include<cuda_runtime.h>
#include<iostream>
#include<string>
#include<sstream>

#include<parseOprand.hpp>
#include<spdlog/spdlog.h>
using std::string;
using std::cout;
using std::endl;
using std::stringstream;


int main(int argc, char const *argv[])
{
    int row,col;
    spdlog::set_pattern("[%c] [%s] [%^---%L---%$] [thread %t] %v");
    if(0!=parseOpt(argc,argv,row,col)){
        SPDLOG_ERROR("parseOpt false");
        return -1;
    }
}
