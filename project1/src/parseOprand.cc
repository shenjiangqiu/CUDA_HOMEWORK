#include<parseOprand.hpp>
#include<iostream>
#include<string>
#include<sstream>
using namespace std;


int parseOpt(int argc,const char**argv,int &row,int &col){
    if(argc!=4){
        cout<<"Usage: -i <rowDim> <colDim>"<<endl;
        return -1;
    }
    
    
    string a1(argv[1]);
    string a2(argv[2]);
    string a3(argv[3]);
    if(a1!="-i"){
        cout<<"Usage: -i <rowDim> <colDim>"<<endl;
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
        std::cerr <<"can not build stringsteam"<<std::endl;
        return -1;
    }
    
    try
    {
        stream1.exceptions(std::ios_base::failbit);
        stream2.exceptions(std::ios_base::failbit);
        stream1>>row;
        stream2>>col;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::cerr <<"cannot read oprand"<<std::endl;
        return -1;
    }
    
    return 0;


}