#include<cuda_runtime.h>
#include<iostream>
#include<string>
#include<sstream>

#include<parseOprand.hpp>

using std::string;
using std::cout;
using std::endl;
using std::stringstream;


int main(int argc, char const *argv[])
{
    int row,col;
    if(0!=parseOpt(argc,argv,row,col)){
        cout<<"parseOpt false"<<endl;
    }
}
