#include<parseOprand.hpp>
#include<iostream>
#include<string>
#include<sstream>
#include<log.hpp>
using namespace std;
//#include<tuple>

int parseOpt(int argc,const char**argv,int &row,int &col,int &mask){
    QDEBUG(argc);
    if(argc!=7){
        QERROR("Usage: -i <dimX> -j <dimY> -k <dimK>");
        return -1;
    }
    string op1(argv[1]);
    string row_str(argv[2]);
    string op2(argv[3]);
    string Y(argv[4]);
    string op3(argv[5]);
    string Z(argv[6]);

    
    if(op1!="-i"||op2!="-j"||op3!="-k"){
        QERROR("Usage: -i <dimX> -j <dimY> -k <dimK>");
        return -1;
    }
    
    stringstream stream1(row_str);
    stringstream stream2(Y);
    stringstream stream3(Z);
    ////auto streams=make_tuple(stream1,stream2,stream3);
    try
    {
        stream1.exceptions(std::ios_base::failbit);
        stream2.exceptions(std::ios_base::failbit);
        stream3.exceptions(std::ios_base::failbit);
        stream1>>row;
        if(stream1.rdbuf()->in_avail()!=0||stream1.fail()){
            throw std::runtime_error("parse error");
        }
        stream2>>col;
        if(stream2.rdbuf()->in_avail()!=0||stream2.fail()){
            throw std::runtime_error("parse error");
        }
        stream3>>mask;
        if(stream3.rdbuf()->in_avail()!=0||stream3.fail()){
            throw std::runtime_error("parse error");
        }
    }
    catch(const std::exception& e)
    {
        QERROR( e.what());
        return -1;
    }
    QDEBUG(mask);
    QDEBUG((mask&1U==0U))
    if(row<=0 or col <=0 or mask <=0 or (mask%2==0U) ){
        QERROR("dim can't less than 1, or the mask cant be even number");
        return -1;
    }
    
    return 0;


}