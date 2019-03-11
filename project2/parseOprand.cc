#include<parseOprand.hpp>
#include<iostream>
#include<string>
#include<sstream>
#include<log.hpp>
using namespace std;


int parseOpt(int argc,const char**argv,int &row){
    QDEBUG(argc);
    if(argc!=3){
        QERROR("Usage: -i <Dim>");
        return -1;
    }
    string op(argv[1]);
    string row_str(argv[2]);
    
    if(op!="-i"){
        QERROR("Usage: -i <Dim>");
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
        QERROR( e.what());
        return -1;
    }
    
    if(row<=0){
        QERROR("dim can't less than 1");
        return -1;
    }
    
    return 0;


}