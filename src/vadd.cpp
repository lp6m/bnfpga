#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "vadd.h"
#include "hls_math.h"
#include <iomanip>
#include <chrono>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>

#define NUMOF_DATASETS 1000
#define MAXOF_VARS 10
#define DATA_BIT 8
typedef ap_uint<DATA_BIT> data_t;
static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

using namespace std;

void load_data(int NUMOF_VARS, ap_uint<32> data[NUMOF_DATASETS], string filepath){
	ifstream ifs;
	ifs.open(filepath, std::ios::in);
	if (!ifs) {
	  cerr << "training file open failed" << endl;
	}
	int i = 0;
	while(!ifs.eof()){
		string line;
		getline(ifs, line);
		if(line == "") break;
		int pos = 0;
		ap_uint<32> d = 0;
		for(int j = 0; j < (int)line.size(); j++){
			if(pos >= NUMOF_VARS) break;
			if(line[j] != ' '){
				unsigned int tmp_d = ((unsigned int)(line[j] - '0'));
				if(tmp_d > 3) cerr << "Warning: the value of dataset must be less than 4. line = " <<  i << "tmp_d = " << tmp_d << endl;
				d |= (tmp_d << pos*2);
				pos++;
			}
		}
		data[i] = d;
		i++;
	}
	ifs.close();
}
int main(int argc, char* argv[]) {
    //TARGET_DEVICE macro needs to be passed from gcc command line
    if(argc != 4) {
    std::cout << "Usage: " << argv[0] <<" <xclbin> <NUMOF_VARS> <dataset filepath>" << std::endl;
    return EXIT_FAILURE;
  }

    char* xclbinFilename = argv[1];
    int NUMOF_VARS = atoi(argv[2]);
    string dataset_filepath = string(argv[3]);

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
    std::chrono::system_clock::time_point  t1, t2, t3, t4;
    t1 = std::chrono::system_clock::now();
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
    cl::Kernel krnl(program,"hypercube_kernel");

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device.
    cl::Buffer buffer_nof_vars(context, CL_MEM_READ_ONLY, 1 * sizeof(int));
    cl::Buffer buffer_dataset(context, CL_MEM_READ_ONLY, NUMOF_DATASETS * sizeof(ap_uint<32>));
    cl::Buffer buffer_max_vals(context, CL_MEM_READ_ONLY, sizeof(ap_uint<32>));
    cl::Buffer buffer_best_matrix(context, CL_MEM_WRITE_ONLY, MAXOF_VARS / 2 * sizeof(ap_uint<32>));
    cl::Buffer buffer_best_score(context, CL_MEM_WRITE_ONLY, 1 * sizeof(float));
    //set the kernel Arguments
    int narg=0;
    krnl.setArg(narg++, buffer_nof_vars);
    krnl.setArg(narg++, buffer_dataset);
    krnl.setArg(narg++, buffer_max_vals);
    krnl.setArg(narg++, buffer_best_score);
    krnl.setArg(narg++, buffer_best_matrix);

    //We then need to map our OpenCL buffers to get the pointers
    int *ptr_nof_vars = (int*)q.enqueueMapBuffer(buffer_nof_vars, CL_TRUE, CL_MAP_READ, 0, 1 * sizeof(int));
    ap_uint<32> *ptr_dataset = (ap_uint<32>*) q.enqueueMapBuffer (buffer_dataset , CL_TRUE , CL_MAP_READ , 0, NUMOF_DATASETS * sizeof(ap_uint<32>));
    ap_uint<32> *ptr_max_vals = (ap_uint<32>*) q.enqueueMapBuffer (buffer_max_vals , CL_TRUE , CL_MAP_READ , 0, sizeof(ap_uint<32>));
    ap_uint<32> *ptr_best_matrix = (ap_uint<32>*) q.enqueueMapBuffer (buffer_best_matrix , CL_TRUE , CL_MAP_WRITE , 0, MAXOF_VARS / 2 * sizeof(ap_uint<32>));
    float *ptr_best_score = (float *)q.enqueueMapBuffer (buffer_best_score , CL_TRUE , CL_MAP_WRITE , 0, 1 * sizeof(float));

    *ptr_nof_vars = NUMOF_VARS;
    //load dataset file
    load_data(NUMOF_VARS, ptr_dataset, dataset_filepath);
    //setting input data
    *ptr_max_vals = 0;
    for(int i = 0; i < NUMOF_VARS; i++){
    	ap_uint<32> max_val = 1;
    	*ptr_max_vals |= (max_val << i*2);
    }
    //for(int i = 0; i < NUMOF_VARS*NUMOF_VARS; i++) ptr_adjacent_matrix[i] = 0;

    // Data will be migrated to kernel space
    q.enqueueMigrateMemObjects({buffer_dataset, buffer_max_vals},0/* 0 means from host*/);
    	/* 0 means from host*/

    //Launch the Kernel
    q.enqueueTask(krnl);

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    q.enqueueMigrateMemObjects({buffer_best_score, buffer_best_matrix},CL_MIGRATE_MEM_OBJECT_HOST);
    t2 = std::chrono::system_clock::now();
    // Wait kernel execution complete
    q.finish();
    t3 = std::chrono::system_clock::now();
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
    cout << "best_matrix" << endl;
    for(int i = 0; i < NUMOF_VARS/2; i++){
    	ap_uint<16> firstdata = ptr_best_matrix[i] >> 16;
    	ap_uint<16> seconddata = ptr_best_matrix[i].range(15, 0);
    	for(int j = 0; j < NUMOF_VARS; j++){
    		bool elm = (firstdata >> j & 1U);
    		cout << elm << " ";
    	}
    	cout << endl;
    	if(i * 2 + 1 < NUMOF_VARS){
				for(int j = 0; j < NUMOF_VARS; j++){
					bool elm = (seconddata >> j & 1U);
					cout << elm << " ";
				}
				cout << endl;
    	}
    }
    double elapsed1 = (double)std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    double elapsed2 = (double)std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count();
    double elapsed3 = (double)std::chrono::duration_cast<std::chrono::microseconds>(t3-t1).count();
    //double elapsed4 = (double)std::chrono::duration_cast<std::chrono::microseconds>(t4-t1).count();
    cout << "Loading Time : " << elapsed1 << "[us]" << endl;
    cout << "FPGA Running Time : " << elapsed2 << "[us]" << endl;
    cout << "ALL TIME : " << elapsed3 << "[us]" << endl;
    q.enqueueUnmapMemObject(buffer_dataset , ptr_dataset);
    q.enqueueUnmapMemObject(buffer_max_vals ,ptr_max_vals);
    q.enqueueUnmapMemObject(buffer_best_score ,ptr_best_score);
    q.enqueueUnmapMemObject(buffer_best_matrix ,ptr_best_matrix);

    q.finish();

    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE :  EXIT_SUCCESS);

}
