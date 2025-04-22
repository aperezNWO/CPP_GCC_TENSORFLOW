/*

TOOLCHAIN : C:\msys64\ucrt64\bin.

gcc -o TensorFlowAppC.exe tf_dll_test.c -I"include" -L"lib" -ltensorflow

*/
#include <stdio.h>
#include <windows.h>

typedef const char* (*GetTFVersionFunc)();  // Define function pointer type

int main() {
    // Load the DLL
    HMODULE hDLL = LoadLibrary("TensorFlowAppC.dll");
    if (hDLL == NULL) {
        printf("Failed to load DLL.\n");
        return 1;
    }

    // Get the function from the DLL
    GetTFVersionFunc GetTensorFlowVersion = (GetTFVersionFunc)GetProcAddress(hDLL, "GetTensorFlowVersion");
    if (!GetTensorFlowVersion) {
        printf("Could not locate the function.\n");
        FreeLibrary(hDLL);
        return 1;
    }

    // Call the function
    const char* version = GetTensorFlowVersion();
    printf("Returning Message : %s\n", version);

    // Clean up
    FreeLibrary(hDLL);
    return 0;
}


