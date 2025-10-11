==================================
?? SOFTWARE CONFIGURATION MANAGEMENT (SCM)
Project: AlgorithmCore (Full-Stack Sorting Demo)
Maintainer: Pablo P.
Last Updated: 2025-04-05
==================================

?? GOAL
Document all commands, configurations, and procedures needed to:
- Set up the development environment
- Build native C++ components
- Integrate with .NET Core
- Distribute binaries
- Maintain version control

----------------------------------
?? DEVELOPMENT ENVIRONMENT SETUP
----------------------------------
# Required Tools
- MSYS2 UCRT64 or MINGW64
- g++ (MinGW-w64)
- make
- pkg-config
- .NET SDK 8.0
- Node.js + Angular CLI

# Install MSYS2 & Required Packages
> Download from https://www.msys2.org/
> Run MSYS2 UCRT64 terminal

pacman -Syu
pacman -S mingw-w64-ucrt-x86_64-gcc \
          mingw-w64-ucrt-x86_64-make \
          mingw-w64-ucrt-x86_64-pkgconf \
          mingw-w64-ucrt-x86_64-opencv \
          git \
          vim

# Add to PATH (Windows)
C:\msys64\ucrt64\bin;C:\msys64\usr\bin

----------------------------------
?? C++ DEPENDENCIES & LIBRARIES
----------------------------------
# OpenCV
pkg-config --cflags --libs opencv4

# Static linking (if available)
pkg-config --cflags --static --libs opencv4

# Manual flags if pkg-config fails
-I/ucrt64/include/opencv4 -lopencv_core -lopencv_imgcodecs ...

----------------------------------
?? BUILD SYSTEM CONFIGURATION
----------------------------------
# Makefile Location
/src/native/Makefile

# Expected Variables
CXX = g++
CXXFLAGS = -std=c++11 -I/ucrt64/include/opencv4
LDFLAGS = $(pkg-config --cflags --static --libs opencv4)

# Custom Libraries
LIBS = -lopencv_gapi -lz -ltbb -lgdi32 -luser32 -lole32

----------------------------------
?? COMPILATION COMMANDS
----------------------------------
# Compile DLL with OpenCV support
g++ -shared -o AlgorithmCore.dll \
    SortingMethods.cpp NeuralNetwork.cpp \
    -I/ucrt64/include/opencv4 \
    $(pkg-config --cflags --static --libs opencv4) \
    -static-libgcc -static-libstdc++ \
    -Wl,--start-group \
    -lopencv_gapi -lz -ltbb -llibjpeg -llibpng \
    -lopengl32 -lglu32 -lgdi32 -luser32 -lole32 \
    -Wl,--end-group

# Or using Makefile
make clean && make

----------------------------------
?? DOTNET INTEGRATION
----------------------------------
# Copy DLL before running .NET app
cp AlgorithmCore.dll ../ASP_NET_CORE_CPP_ENTRY/bin/Debug/net8.0/

# DllImport usage in C#
[DllImport("AlgorithmCore.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr QuickSortWithHistory(double[] data, int size);

----------------------------------
?? DISTRIBUTION & DEPLOYMENT
----------------------------------
# Required DLLs to bundle with .NET app
- AlgorithmCore.dll
- libgcc_s_seh-1.dll
- libstdc++-6.dll
- libwinpthread-1.dll
# Optional: OpenCV modules if not statically linked

# Folder structure after publish
bin/
+-- AlgorithmCore.dll
+-- libgcc_s_seh-1.dll
+-- libstdc++-6.dll
+-- libwinpthread-1.dll

----------------------------------
?? VERSION CONTROL (GIT)
----------------------------------
# Initialize repo
git init
git add .
git commit -m "Initial commit"

# Branching Strategy
git checkout -b feature/neural-sort
git checkout -b release/v1.0

# Sync with remote
git remote add origin https://github.com/pablo/AlgorithmCore.git
git push -u origin main

# Tag stable versions
git tag v1.0.0
git push origin v1.0.0

----------------------------------
?? TROUBLESHOOTING
----------------------------------
# Error: "cannot find -ltbb"
? Solution: Rebuild OpenCV with TBB disabled or install tbb-static

# Error: "The specified module could not be found"
? Use Dependencies (https://github.com/lucasg/Dependencies) to check missing DLLs

# Makefile: *** missing separator
? Replace spaces with tabs before command lines

# P/Invoke returns garbage
? Ensure string memory is managed correctly (use FreeString())