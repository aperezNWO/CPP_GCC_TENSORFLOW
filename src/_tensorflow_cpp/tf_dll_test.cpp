/*

1) TOOLCHAIN : C:\msys64\ucrt64\bin (CHANGE PATH)

2) g++ -o TensorFlowAppCPP.exe tf_dll_test.cpp 

3) UTILIZAR PROYECDTO CPP_GCC_TENSORFLOW.DEV (Embarcadero Dev C++) PROVISIONALMENTE PARA 
   
   A) VISUALIZAR Y EDITAR ARCHIVOS.
   B) COMPILAR CON LINEA DE COMANDOS.
   C) EJECUTAR COMANDOS GIT
   
*/

#include <stdio.h>
#include <windows.h>

typedef const char* (*GetTensorFlowAPIVersionFunc)();  // Define function pointer type
typedef const char* (*GetTensorFlowAPPVersionFunc)();  // Define function pointer type


int main() {
    // Load the DLL
    HMODULE hDLL = LoadLibrary("TensorFlowAppCPP.dll");
    if (hDLL == NULL) {
        printf("Failed to load DLL.\n");
        return 1;
    }
    
	//////////////////////////////////////////////////////
    //GetTensorFlowAPIVersion
    //////////////////////////////////////////////////////
    
    GetTensorFlowAPIVersionFunc GetTensorFlowAPIVersion = (GetTensorFlowAPIVersionFunc)GetProcAddress(hDLL, "GetTensorFlowAPIVersion");
    if (!GetTensorFlowAPIVersion) {
        printf("Could not locate the function 'GetTensorFlowAPIVersion'.\n");
        FreeLibrary(hDLL);
        return 1;
    }

    // Call the function
    const char* apiVersion = GetTensorFlowAPIVersion();
    printf("'GetTensorFlowAPIVersion' : %s\n", apiVersion);

	//////////////////////////////////////////////////////
    //GetTensorFlowAPPVersion
    //////////////////////////////////////////////////////
    
    
    GetTensorFlowAPPVersionFunc GetTensorFlowAPPVersion = (GetTensorFlowAPPVersionFunc)GetProcAddress(hDLL, "GetTensorFlowAppVersion");
    if (!GetTensorFlowAPIVersion) {
        printf("Could not locate the function 'GetTensorFlowAPPVersion'.\n");
        FreeLibrary(hDLL);
        return 1;
    }

    // Call the function
    const char* appVersion = GetTensorFlowAPPVersion();
    printf("'GetTensorFlowAPPVersion' : %s\n", appVersion);
    
    // Clean up
    FreeLibrary(hDLL);
    return 0;
}


