/*

//
TOOLCHAIN : C:\msys64\uctr64\bin (CHANGE PATH)

// INSTALLERS
https://www.tensorflow.org/install/lang_c?hl=es-419


// COMPILE OK - NON STATIC

g++ -I"include" -L"lib" -shared -m64 -o TensorFlowAppCPP.dll tensorFlowApp.cpp -ltensorflow  -Wl,--subsystem,windows 


// UNABLE TO COMPILE AS STATIC
gcc -I"include" -L"lib" -shared -static -static-libgcc -static-libstdc++ -m64 -o TensorFlowAppC.dll tf_dll_gen.c -ltensorflow -Wl,--subsystem,console 


3) UTILIZAR PROYECDTO CPP_GCC_TENSORFLOW.DEV (Embarcadero Dev C++) PROVISIONALMENTE PARA 
   
   A) VISUALIZAR Y EDITAR ARCHIVOS.
   B) COMPILAR CON LINEA DE COMANDOS.
   C) EJECUTAR COMANDOS GIT
   
*/


#include <tensorflow/c/c_api.h>
#include "include/tensorFlowApp.h"

//
TensorFlowApp::TensorFlowApp()
{
     //
     ReadConfigFile();
}
//
TensorFlowApp::~TensorFlowApp()
{
    //
}

//
int          TensorFlowApp::ReadConfigFile()
{
    // Open the configuration file
    std::ifstream configFile("tensorflow.ini");

    // Check if the file is opened successfully
    if (!configFile.is_open()) {
        std::cerr << "Error opening the configuration file." << std::endl;
        return 1;
    }

    // Read the file line by line
    std::string line = "";
    while (std::getline(configFile, line)) {
        // Skip empty lines or lines starting with '#' (comments)
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Split the line into key and value
        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value))
        {
            // Trim leading and trailing whitespaces from key and value
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            // Insert key-value pair into the map
            this->configMap[key] = value;
        }
    }

    // Close the configuration file
    configFile.close();

    //
    return 0;
} 

//
const char*  TensorFlowApp::GetTensorFlowAPIVersion()
{
	//
  	return TF_Version(); // Return the TensorFlow version directly;
}

//
std::string TensorFlowApp::GetTensorFlowAppVersion()
{
    auto it = configMap.find("DLL_VERSION");
    if (it != configMap.end()) {
        return it->second;
    }
    return "UNKNOWN"; 
}



/////////////////////////////////////////////////////////////////////
// DLL ENTRY POINTS
/////////////////////////////////////////////////////////////////////

DLL_EXPORT const char* GetTensorFlowAPIVersion() 
{
    static std::string versionCache;
    {
        std::unique_ptr<TensorFlowApp> app = std::make_unique<TensorFlowApp>();
        versionCache = app->GetTensorFlowAPIVersion(); // TF_Version()
    }
    return versionCache.c_str();
}

DLL_EXPORT const char* GetTensorFlowAppVersion() {
    static std::string version;
    if (version.empty()) {
        TensorFlowApp app;
        version = app.GetTensorFlowAppVersion();
    }
    return version.c_str();
}



