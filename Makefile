.PHONY: configure build clean

build: configure
	cmake --build build -j

configure:
	cmake -B build -DALPHA_BUILD_PLAIN=1

clean:
	rm -rf build
