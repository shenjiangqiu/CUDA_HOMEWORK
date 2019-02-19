#include<parseOprand.hpp>
#include<iostream>
#include<string>
#include<sstream>
#include<spdlog/spdlog.h>
using namespace std;


int parseOpt(int argc,const char**argv,int &row,int &col){
    if(argc!=4){
        SPDLOG_ERROR("Usage: -i <rowDim> <colDim>");
        return -1;
    }
    
    
    string a1(argv[1]);
    string a2(argv[2]);
    string a3(argv[3]);
    if(a1!="-i"){
        SPDLOG_ERROR("Usage: -i <rowDim> <colDim>");
        return -1;
    }
    
    stringstream stream1;
    stringstream stream2;
    try
    {
        stream1=stringstream(a2);
        stream2=stringstream(a3);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        SPDLOG_ERROR("can not build stringsteam");
        return -1;
    }
    
    try
    {
        stream1.exceptions(std::ios_base::failbit);
        stream2.exceptions(std::ios_base::failbit);
        stream1>>row;
        stream2>>col;
        if(stream1.fail() or stream2.fail()){
            throw std::runtime_error("can't read from stringstream");
        }
    }
    catch(const std::exception& e)
    {
        SPDLOG_ERROR( e.what() );
        SPDLOG_ERROR("cannot read oprand,may be you need to input a number");
        return -1;
    }
    
    return 0;


}