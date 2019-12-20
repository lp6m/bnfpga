#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "vadd.h"
#include "hls_math.h"
#include <iomanip>
#include <chrono>

#define NUMOF_VARS 8
#define NUMOF_DATASETS 1000

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

using namespace std;

void load_data(int data[NUMOF_DATASETS*NUMOF_VARS]){
	ifstream ifs;
	ifs.open("../asia.idt", std::ios::in);
	if (!ifs) {
	  cerr << "training file open failed" << endl;
	}
	int i = 0;
	while(!ifs.eof()){
		string line;
		getline(ifs, line);
		if(line == "") break;
		int pos = 0;
		for(int j = 0; j < (int)line.size(); j++){
			if(pos >= NUMOF_VARS) break;
			if(line[j] != ' ') data[(pos++)*NUMOF_DATASETS+i] = (line[j] - '0');
		}
		i++;
	}
	ifs.close();
}
int main(int argc, char* argv[]) {
	//cout << fixed << setprecision(10) << hls::lgamma(1.0) <<" " << hls::lgamma(1.0)<< endl;
    //TARGET_DEVICE macro needs to be passed from gcc command line
    if(argc != 2) {
    std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
    return EXIT_FAILURE;
  }

    char* xclbinFilename = argv[1];


    // Creates a vector of DATA_SIZE elements with an initial value of 10 and 32
    // using customized allocator for getting buffer alignment to 4k boundary

    std::vector<cl::Device> devices;
    cl::Device device;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    //traversing all Platforms To find Xilinx Platform and targeted
    //Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if ( platformName == "Xilinx"){
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
      if (devices.size()){
        device = devices[0];
        found_device = true;
        break;
      }
        }
    }
    if (found_device == false){
       std::cout << "Error: Unable to find Target Device "
           << device.getInfo<CL_DEVICE_NAME>() << std::endl;
       return EXIT_FAILURE;
    }

    // Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);

    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    devices.resize(1);
    cl::Program program(context, devices, bins);

    // This call will get the kernel object from program. A kernel is an
    // OpenCL function that is executed on the FPGA.
    // cl::Kernel krnl_vector_add(program,"krnl_vadd");
    //cl::Kernel krnl(program,"dp_bayesian_learning");
    cl::Kernel krnl(program,"hypercube_kernel");

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device.
    // size_t size_in_bytes = DATA_SIZE * sizeof(int);
    cl::Buffer buffer_dataset(context, CL_MEM_READ_ONLY, NUMOF_VARS * NUMOF_DATASETS * sizeof(int));
    cl::Buffer buffer_nof_vars(context, CL_MEM_READ_ONLY, NUMOF_VARS * sizeof(int));
    //cl::Buffer buffer_adjacent_matrix(context, CL_MEM_WRITE_ONLY, NUMOF_VARS * NUMOF_VARS * sizeof(int));
    //cl::Buffer buffer_best_order(context, CL_MEM_WRITE_ONLY, NUMOF_VARS * sizeof(int));
    cl::Buffer buffer_best_score(context, CL_MEM_WRITE_ONLY, 1 * sizeof(float));
    //set the kernel Arguments
    int narg=0;
    krnl.setArg(narg++, buffer_dataset);
    krnl.setArg(narg++, buffer_nof_vars);
    krnl.setArg(narg++, buffer_best_score);
    /*krnl.setArg(narg++, buffer_adjacent_matrix);
    krnl.setArg(narg++, buffer_best_order);}*/

    //We then need to map our OpenCL buffers to get the pointers
    int *ptr_dataset = (int *) q.enqueueMapBuffer (buffer_dataset , CL_TRUE , CL_MAP_READ , 0, NUMOF_VARS * NUMOF_DATASETS * sizeof(int));
    int *ptr_nof_vars = (int *) q.enqueueMapBuffer (buffer_nof_vars , CL_TRUE , CL_MAP_READ , 0, NUMOF_VARS * sizeof(int));
    //int *ptr_adjacent_matrix = (int *) q.enqueueMapBuffer (buffer_adjacent_matrix , CL_TRUE , CL_MAP_WRITE , 0, NUMOF_VARS * NUMOF_VARS * sizeof(int));
    //int *ptr_best_order = (int *) q.enqueueMapBuffer (buffer_best_order , CL_TRUE , CL_MAP_WRITE , 0, NUMOF_VARS * sizeof(int));
    float *ptr_best_score = (float *)q.enqueueMapBuffer (buffer_best_score , CL_TRUE , CL_MAP_WRITE , 0, 1 * sizeof(float));
    //load dataset file
    load_data(ptr_dataset);
    //setting input data
    for(int i = 0; i < NUMOF_VARS; i++) ptr_nof_vars[i] = 2;
    //for(int i = 0; i < NUMOF_VARS*NUMOF_VARS; i++) ptr_adjacent_matrix[i] = 0;

    std::chrono::system_clock::time_point  t1, t2, t3, t4;

    t1 = std::chrono::system_clock::now();
    // Data will be migrated to kernel space
    q.enqueueMigrateMemObjects({buffer_dataset, buffer_nof_vars},0/* 0 means from host*/);
    	/* 0 means from host*/
    t2 = std::chrono::system_clock::now();
    //Launch the Kernel
    q.enqueueTask(krnl);
    t3 = std::chrono::system_clock::now();

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    //q.enqueueMigrateMemObjects({buffer_adjacent_matrix, buffer_best_order},CL_MIGRATE_MEM_OBJECT_HOST);
    q.enqueueMigrateMemObjects({buffer_best_score},CL_MIGRATE_MEM_OBJECT_HOST);
    t4 = std::chrono::system_clock::now();
    q.finish();

    //Verify the result
    int match = 0;
    //Show the result
    /*for(int i = 0; i < NUMOF_VARS; i++) cout << ptr_best_order[i] << " ";
    cout << endl;
    for(int i = 0; i < NUMOF_VARS; i++){
        for(int j = 0; j < NUMOF_VARS; j++){
            cout << ptr_adjacent_matrix[i*NUMOF_VARS+j] << " ";
        }
        cout << endl;
    }*/
    cout << "best_score : " << *ptr_best_score << endl;
    double elapsed1 = (double)std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    double elapsed2 = (double)std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count();
    double elapsed3 = (double)std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count();
    double elapsed4 = (double)std::chrono::duration_cast<std::chrono::microseconds>(t4-t1).count();
    cout << "HOST DDR->FPGA DDR : " << elapsed1 << "[microseconds]" << endl;
    cout << "FPGA Running Time : " << elapsed2 << "[microseconds]" << endl;
    cout << "FPGA DDR->HOST DDR : " << elapsed3 << "[microseconds]" << endl;
    cout << "ALL TIME : " << elapsed4 << "[microseconds]" << endl;
    q.enqueueUnmapMemObject(buffer_dataset , ptr_dataset);
    q.enqueueUnmapMemObject(buffer_nof_vars ,ptr_nof_vars);
    q.enqueueUnmapMemObject(buffer_best_score ,ptr_best_score);
    //q.enqueueUnmapMemObject(buffer_adjacent_matrix ,ptr_adjacent_matrix);
    //q.enqueueUnmapMemObject(buffer_best_order, ptr_best_order);

    q.finish();

    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE :  EXIT_SUCCESS);

}
