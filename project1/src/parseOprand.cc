#include<parseOprand.hpp>
#include<iostream>
#include<string>
#include<sstream>
#include<spdlog/spdlog.h>
using namespace std;


int parseOpt(int argc,const char**argv,int &row){
    if(argc!=3){
        SPDLOG_ERROR("Usage: -i <row>");
        return -1;
    }
    string op(argv[1]);
    string row_str(argv[2]);
    
    if(op!="-i"){
        SPDLOG_ERROR("Usage: -i <rowDim> <colDim>");
        return -1;
    }
    
    stringstream stream(row_str);
    try
    {
        stream.exceptions(std::ios_base::failbit);
        stream>>row;
        if(stream.rdbuf()->in_avail()!=0||stream.fail()){
            throw std::runtime_error("parse error");
        }
    }
    catch(const std::exception& e)
    {
        SPDLOG_ERROR( e.what());
        return -1;
    }
    
    if(row<=0){
        SPDLOG_ERROR("dim can't less than 1");
        return -1;
    }
    
    return 0;


}