.PHONY: configure build clean

build: configure
	cmake --build build -j

configure:
	cmake -B build

clean:
	rm -rf build
