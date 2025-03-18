#build.sh
echo "This script should build your project now..."

mkdir -p build
cd build

#generate makefile
cmake ..

#create executable
make