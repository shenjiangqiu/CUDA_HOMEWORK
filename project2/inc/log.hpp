#pragma once
#include<iostream>

#define QERROR(x) std::cout<<"ERROR: at file "<<__FILE__<<":"<<__LINE__<<", "<<x<<std::endl;
#define QINFO(x) std::cout<<"INFO : at file "<<__FILE__<<":"<<__LINE__<<", "<<x<<std::endl;
#define QDEBUG(x) std::cout<<"DEBUG: at file "<<__FILE__<<":"<<__LINE__<<", "<<x<<std::endl;
