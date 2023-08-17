#pragma once

#include <exception>
#include <stdexcept>
#include <string>


#define _STRING_LINE_(s) #s
#define _STRING_LINE2_(s) _STRING_LINE_(s)
#define __LINESTR__ _STRING_LINE2_(__LINE__)
#define FileAndLine __FILE__ ":" __LINESTR__

//#define FileAndLine __FILE__ 

//This has been changed on July 2012 to enforce informative error messages.
//The error message can also be something like "gv37530970n9876" as long as it is unique.
static void Verify(bool expression,const char* msg){
	if(!expression)
	{
		if(msg==0)
			throw std::runtime_error("Verify failed.");
		else{
			//std::cout<<"Error "<<msg<<std::endl;
			throw std::runtime_error(msg);
		}
	}	
}

//This is slower...
static void Verify(bool expression,const std::string& msg){
	Verify(expression,msg.c_str());
}
