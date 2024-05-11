#pragma once

#define USE_BOOST_SHARED_PTR

#ifdef USE_BOOST_SHARED_PTR
#include <boost/shared_ptr.hpp>
#define dskjhlkh344343 boost
#else
#include <memory>
#define dskjhlkh344343 std
#endif

class TrackedObject{
	dskjhlkh344343::shared_ptr<void> ptr;	
public:
	TrackedObject(){}
	explicit TrackedObject(dskjhlkh344343::shared_ptr<void> ptr):ptr(ptr){}
	//This c'tor takes over ownership
	template<typename T>
	explicit TrackedObject(T* theObject):ptr(dskjhlkh344343::shared_ptr<T>(theObject)){}
	void* Pointer(){
		return ptr.get();
	}	
};
