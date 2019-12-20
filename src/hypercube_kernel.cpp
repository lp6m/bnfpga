#include <iostream>
#include <string.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <iomanip>
// #include "my_lgamma.h"
typedef ap_fixed<64, 32> score_t;
//typedef ap_fixed<32,10> score_t;
//typedef double score_t;
typedef unsigned int varset_t;
//#define MAXOF_VARS 8
#define NUMOF_VARS 8
#define PE_NUM 70
#define NUMOF_DATASETS 1000
//#define MINUS_INF (-1*1e-15);
#define DEBUGOUT false

using namespace std;

namespace mylib{
  score_t lgamma_int(int k){;
    //log((k-1)!)
    score_t res = 0;
    for(int i = 2; i <= (k - 1); i++){
  #pragma HLS PIPELINE//UNROLL?
      res += hls::log((score_t)i);
    }
    return res;
  }
  score_t calc_bdeu_local_score(int child, varset_t parents, int dataset[NUMOF_VARS][NUMOF_DATASETS], int nof_vars[NUMOF_VARS]){
    score_t local_score = 0;
    //const score_t ess = 10.0;
    int itr_num = 1;
    int refresh_timing[NUMOF_VARS];
    int counters[NUMOF_VARS];
    int now_comb[NUMOF_VARS];
  #pragma HLS ARRAY_PARTITION variable=refresh_timing complete dim = 1
  #pragma HLS ARRAY_PARTITION variable=counters complete dim = 1
  #pragma HLS ARRAY_PARTITION variable=now_comb complete dim = 1

    int next_timing = 1;
    for(int i = 0; i < NUMOF_VARS; i++){
  #pragma HLS PIPELINE
      counters[i] = 0;
      now_comb[i] = 0;
      if(((parents >> i) & 1) || i == child){
        itr_num *= nof_vars[i];
        refresh_timing[i] = next_timing;
        next_timing *= nof_vars[i];
      }else{
        refresh_timing[i] = 0;//no need
      }
    }
    int status = 0;
    //score_t nijk_prime = (score_t)ess/(score_t)itr_num;
    //score_t nik_prime = nijk_prime * (score_t)nof_vars[child];

    combination_loop:for(int i = 0; i < itr_num; i++){
  #pragma HLS loop_tripcount min=1 max=256 avg=34
      //show combination
      //calc nik, nijk
      int nik = 0;
      int nijk = 0;
      for(int j = 0; j < NUMOF_DATASETS; j++){
        bool nik_match = true;
        bool nijk_match = true;
        for(int k = 0; k < NUMOF_VARS; k++){
  #pragma HLS UNROLL
          int d = dataset[k][j];
          if(((parents >> k) & 1) || k == child){
            if(d != now_comb[k]){
              nijk_match = false;
              if(k != child) nik_match = false;
            }
          }
        }
        if(nijk_match) nijk++;
        if(nik_match) nik++;
      }
      //calc score
  //#pragma HLS allocation instances = hls::my_lgamma limit=1 function
      //if(now_comb[child] == 0 && nik > 0) local_score += hls::my_lgamma((score_t)nik_prime/(score_t)((score_t)nik + nik_prime));
      //if(nijk > 0) local_score += hls::my_lgamma((score_t)((score_t)nijk + nijk_prime)/(score_t)(nijk_prime));

      //K2Score
      //The reason of the condition : now_comb[child] == 0 is, there is only one time when child value is 0 for each parents combination.
      int r = nof_vars[child];
#pragma HLS allocation instances=lgamma_int limit=1 function
      score_t tmp = (lgamma_int(r) - lgamma_int(nik + r));
      if(now_comb[child] == 0 && nik > 0) local_score += (lgamma_int(r) - lgamma_int(nik + r));
      if(nijk > 0) local_score += lgamma_int(nijk + 1);


      //next combination
      for(int k = 0; k < NUMOF_VARS; k++){
  #pragma HLS UNROLL
        if(((parents >> k) & 1) || k == child){
          counters[k]++;
          if(counters[k] == refresh_timing[k]){
            now_comb[k] = (now_comb[k] == nof_vars[k] - 1) ? 0 : (now_comb[k]+1);
            counters[k] = 0;
          }
        }
      }
    }
    return local_score;
  }
}
extern "C" {

typedef struct{
  score_t q;
  score_t f[NUMOF_VARS];
} pe_out;

void PE(int A, int stage,
  int dataset[NUMOF_VARS][NUMOF_DATASETS], int nof_vars[NUMOF_VARS],
  pe_out indata[NUMOF_VARS], pe_out& outdata){

  if(stage == 0 || A < 0) return;

  pe_out tmp_outdata;
  int cnt = 0;
  bool firstflag = true;
  int indexcnt[NUMOF_VARS];
  unsigned int indexbit = (1U << (stage - 1)) - 1;
  for(int i = 0; i < NUMOF_VARS; i++){
#pragma HLS UNROLL
    indexcnt[i] = 0;
  }
  for(int i = 0; i < NUMOF_VARS; i++){
    if((A >> i) & 1U){
      varset_t parents = (A^(1U << i));
      score_t s = mylib::calc_bdeu_local_score(i, parents, dataset, nof_vars);
      score_t f = s;
      for(int k = 0; k < NUMOF_VARS; k++){
        if((indexbit >> k) & 1U){
          f = max(f, indata[k].f[indexcnt[k]]);
        }
      }
      //next status
      for(int k = 0; k < NUMOF_VARS; k++){
#pragma HLS UNROLL
        if((indexbit >> k) & 1U) indexcnt[k]++;
      }
      indexbit = (indexbit >> 1) | (1U << (stage - 1));

      score_t nowq = f + indata[stage-1-cnt].q;
      tmp_outdata.f[cnt] = f;
      tmp_outdata.q = firstflag ? nowq : max(tmp_outdata.q, nowq);
      firstflag = false;
      cnt++;
    }
  }
  outdata = tmp_outdata;
}

int nck(int n, int k){
  int ans = 1;
  for(int i = 0; i < k; i++){
    ans *= (n - i);
  }
  for(int i = 1; i <= k; i++){
    ans /= i;
  }
  return ans;
}

//K: popcount of val
int val2index(int val, int K){
  int res = 0;
  int onecnt = 0;
  for(int i = 0; i < NUMOF_VARS; i++){
    if((val >> i) & 1U){
      onecnt++;
    }else{
      int n = NUMOF_VARS - 1 - i;
      int k = K - 1 - onecnt;
      if(n > 0 && k >= 0) res += nck(n, k);
    }
  }
  return res;
}

//top function
void hypercube_kernel(
    int *p_dataset,
    int *p_nof_vars, float* p_best_score){

#pragma HLS INTERFACE m_axi port=p_dataset offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=p_nof_vars offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=p_best_score offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=p_dataset bundle=control
#pragma HLS INTERFACE s_axilite port=p_nof_vars bundle=control
#pragma HLS INTERFACE s_axilite port=p_best_score bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	int dataset[NUMOF_VARS][NUMOF_DATASETS];
	int dataset2[NUMOF_VARS][NUMOF_DATASETS];
	int dataset3[NUMOF_VARS][NUMOF_DATASETS];
	int dataset4[NUMOF_VARS][NUMOF_DATASETS];
	int dataset5[NUMOF_VARS][NUMOF_DATASETS];
	int dataset6[NUMOF_VARS][NUMOF_DATASETS];
	int dataset7[NUMOF_VARS][NUMOF_DATASETS];
	int dataset8[NUMOF_VARS][NUMOF_DATASETS];
	int dataset9[NUMOF_VARS][NUMOF_DATASETS];
	int dataset10[NUMOF_VARS][NUMOF_DATASETS];
	int dataset11[NUMOF_VARS][NUMOF_DATASETS];
	int dataset12[NUMOF_VARS][NUMOF_DATASETS];
	int dataset13[NUMOF_VARS][NUMOF_DATASETS];
	int dataset14[NUMOF_VARS][NUMOF_DATASETS];
	int dataset15[NUMOF_VARS][NUMOF_DATASETS];
	int dataset16[NUMOF_VARS][NUMOF_DATASETS];
	int dataset17[NUMOF_VARS][NUMOF_DATASETS];
	int dataset18[NUMOF_VARS][NUMOF_DATASETS];
	int dataset19[NUMOF_VARS][NUMOF_DATASETS];
	int dataset20[NUMOF_VARS][NUMOF_DATASETS];
	int dataset21[NUMOF_VARS][NUMOF_DATASETS];
	int dataset22[NUMOF_VARS][NUMOF_DATASETS];
	int dataset23[NUMOF_VARS][NUMOF_DATASETS];
	int dataset24[NUMOF_VARS][NUMOF_DATASETS];
	int dataset25[NUMOF_VARS][NUMOF_DATASETS];
	int dataset26[NUMOF_VARS][NUMOF_DATASETS];
	int dataset27[NUMOF_VARS][NUMOF_DATASETS];
	int dataset28[NUMOF_VARS][NUMOF_DATASETS];
	int dataset29[NUMOF_VARS][NUMOF_DATASETS];
	int dataset30[NUMOF_VARS][NUMOF_DATASETS];
	int dataset31[NUMOF_VARS][NUMOF_DATASETS];
	int dataset32[NUMOF_VARS][NUMOF_DATASETS];
	int dataset33[NUMOF_VARS][NUMOF_DATASETS];
	int dataset34[NUMOF_VARS][NUMOF_DATASETS];
	int dataset35[NUMOF_VARS][NUMOF_DATASETS];
	int dataset36[NUMOF_VARS][NUMOF_DATASETS];
	int dataset37[NUMOF_VARS][NUMOF_DATASETS];
	int dataset38[NUMOF_VARS][NUMOF_DATASETS];
	int dataset39[NUMOF_VARS][NUMOF_DATASETS];
	int dataset40[NUMOF_VARS][NUMOF_DATASETS];
	int dataset41[NUMOF_VARS][NUMOF_DATASETS];
	int dataset42[NUMOF_VARS][NUMOF_DATASETS];
	int dataset43[NUMOF_VARS][NUMOF_DATASETS];
	int dataset44[NUMOF_VARS][NUMOF_DATASETS];
	int dataset45[NUMOF_VARS][NUMOF_DATASETS];
	int dataset46[NUMOF_VARS][NUMOF_DATASETS];
	int dataset47[NUMOF_VARS][NUMOF_DATASETS];
	int dataset48[NUMOF_VARS][NUMOF_DATASETS];
	int dataset49[NUMOF_VARS][NUMOF_DATASETS];
	int dataset50[NUMOF_VARS][NUMOF_DATASETS];
	int dataset51[NUMOF_VARS][NUMOF_DATASETS];
	int dataset52[NUMOF_VARS][NUMOF_DATASETS];
	int dataset53[NUMOF_VARS][NUMOF_DATASETS];
	int dataset54[NUMOF_VARS][NUMOF_DATASETS];
	int dataset55[NUMOF_VARS][NUMOF_DATASETS];
	int dataset56[NUMOF_VARS][NUMOF_DATASETS];
	int dataset57[NUMOF_VARS][NUMOF_DATASETS];
	int dataset58[NUMOF_VARS][NUMOF_DATASETS];
	int dataset59[NUMOF_VARS][NUMOF_DATASETS];
	int dataset60[NUMOF_VARS][NUMOF_DATASETS];
	int dataset61[NUMOF_VARS][NUMOF_DATASETS];
	int dataset62[NUMOF_VARS][NUMOF_DATASETS];
	int dataset63[NUMOF_VARS][NUMOF_DATASETS];
	int dataset64[NUMOF_VARS][NUMOF_DATASETS];
	int dataset65[NUMOF_VARS][NUMOF_DATASETS];
	int dataset66[NUMOF_VARS][NUMOF_DATASETS];
	int dataset67[NUMOF_VARS][NUMOF_DATASETS];
	int dataset68[NUMOF_VARS][NUMOF_DATASETS];
	int dataset69[NUMOF_VARS][NUMOF_DATASETS];
	int dataset70[NUMOF_VARS][NUMOF_DATASETS];
	#pragma HLS ARRAY_PARTITION variable=dataset complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset2 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset3 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset4 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset5 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset6 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset7 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset8 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset9 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset10 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset11 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset12 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset13 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset14 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset15 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset16 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset17 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset18 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset19 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset20 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset21 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset22 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset23 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset24 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset25 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset26 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset27 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset28 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset29 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset30 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset31 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset32 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset33 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset34 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset35 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset36 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset37 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset38 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset39 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset40 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset41 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset42 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset43 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset44 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset45 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset46 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset47 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset48 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset49 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset50 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset51 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset52 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset53 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset54 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset55 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset56 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset57 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset58 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset59 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset60 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset61 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset62 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset63 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset64 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset65 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset66 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset67 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset68 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset69 complete dim = 1
	#pragma HLS ARRAY_PARTITION variable=dataset70 complete dim = 1

  int nof_vars[PE_NUM][NUMOF_VARS];
#pragma HLS ARRAY_PARTITION variable=nof_vars complete dim = 1
#pragma HLS RESOURCE variable=nof_vars core=RAM_1P_LUTRAM

/*#pragma HLS stable variable=dataset
#pragma HLS stable variable=dataset2
#pragma HLS stable variable=dataset3
#pragma HLS stable variable=dataset4
#pragma HLS stable variable=dataset5
#pragma HLS stable variable=dataset6*/

  for(int i = 0; i < NUMOF_VARS; i++){
    int offset = i * NUMOF_DATASETS;
    memcpy(dataset[i], p_dataset + offset, NUMOF_DATASETS * sizeof(int));
  }
  for(int i = 0; i < NUMOF_VARS; i++){
    for(int j = 0; j < NUMOF_DATASETS; j++){
      int d = dataset[i][j];
      dataset2[i][j] = d; dataset3[i][j] = d; dataset4[i][j] = d;
      dataset5[i][j] = d; dataset6[i][j] = d; dataset7[i][j] = d;
      dataset8[i][j] = d; dataset9[i][j] = d; dataset10[i][j] = d;
      dataset11[i][j] = d; dataset12[i][j] = d; dataset13[i][j] = d;
      dataset14[i][j] = d; dataset15[i][j] = d; dataset16[i][j] = d;
      dataset17[i][j] = d; dataset18[i][j] = d; dataset19[i][j] = d;
      dataset20[i][j] = d; dataset21[i][j] = d; dataset22[i][j] = d;
      dataset23[i][j] = d; dataset24[i][j] = d; dataset25[i][j] = d;
      dataset26[i][j] = d; dataset27[i][j] = d; dataset28[i][j] = d;
      dataset29[i][j] = d; dataset30[i][j] = d; dataset31[i][j] = d;
      dataset32[i][j] = d; dataset33[i][j] = d; dataset34[i][j] = d;
      dataset35[i][j] = d; dataset36[i][j] = d; dataset37[i][j] = d;
      dataset38[i][j] = d; dataset39[i][j] = d; dataset40[i][j] = d;
      dataset41[i][j] = d; dataset42[i][j] = d; dataset43[i][j] = d;
      dataset44[i][j] = d; dataset45[i][j] = d; dataset46[i][j] = d;
      dataset47[i][j] = d; dataset48[i][j] = d; dataset49[i][j] = d;
      dataset50[i][j] = d; dataset51[i][j] = d; dataset52[i][j] = d;
      dataset53[i][j] = d; dataset54[i][j] = d; dataset55[i][j] = d;
      dataset56[i][j] = d; dataset57[i][j] = d; dataset58[i][j] = d;
      dataset59[i][j] = d; dataset60[i][j] = d; dataset61[i][j] = d;
      dataset62[i][j] = d; dataset63[i][j] = d; dataset64[i][j] = d;
      dataset65[i][j] = d; dataset66[i][j] = d; dataset67[i][j] = d;
      dataset68[i][j] = d; dataset69[i][j] = d; dataset70[i][j] = d;
    }
  }

  for(int i = 0; i < PE_NUM; i++){
    int offset = i * NUMOF_VARS;
    memcpy(nof_vars[i], p_nof_vars, NUMOF_VARS * sizeof(int));
  }
  const int A[NUMOF_VARS + 1][PE_NUM] = {
  {0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 2, 4, 8, 16, 32, 64, 128, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 5, 9, 17, 33, 65, 129, 6, 10, 18, 34, 66, 130, 12, 20, 36, 68, 132, 24, 40, 72, 136, 48, 80, 144, 96, 160, 192, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38, 70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56, 88, 152, 104, 168, 200, 112, 176, 208, 224, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {15, 23, 39, 71, 135, 27, 43, 75, 139, 51, 83, 147, 99, 163, 195, 29, 45, 77, 141, 53, 85, 149, 101, 165, 197, 57, 89, 153, 105, 169, 201, 113, 177, 209, 225, 30, 46, 78, 142, 54, 86, 150, 102, 166, 198, 58, 90, 154, 106, 170, 202, 114, 178, 210, 226, 60, 92, 156, 108, 172, 204, 116, 180, 212, 228, 120, 184, 216, 232, 240},
  {31, 47, 79, 143, 55, 87, 151, 103, 167, 199, 59, 91, 155, 107, 171, 203, 115, 179, 211, 227, 61, 93, 157, 109, 173, 205, 117, 181, 213, 229, 121, 185, 217, 233, 241, 62, 94, 158, 110, 174, 206, 118, 182, 214, 230, 122, 186, 218, 234, 242, 124, 188, 220, 236, 244, 248, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {63, 95, 159, 111, 175, 207, 119, 183, 215, 231, 123, 187, 219, 235, 243, 125, 189, 221, 237, 245, 249, 126, 190, 222, 238, 246, 250, 252, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {127, 191, 223, 239, 247, 251, 253, 254, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {255, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
  };

#pragma HLS ARRAY_PARTITION variable=A complete dim=1
#pragma HLS RESOURCE variable=A core=ROM_1P_BRAM

  int a[PE_NUM] = {0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
#pragma HLS ARRAY_PARTITION variable=a complete dim=1
//#pragma HLS RESOURCE variable=a core=RAM_1P_LUTRAM

  pe_out indata[PE_NUM][NUMOF_VARS];
  pe_out outdata[PE_NUM];
  //initialize (output of stage 0)
  for(int i = 0; i < PE_NUM; i++){
	outdata[i].q = 0;
    for(int j = 0; j < NUMOF_VARS; j++){
      outdata[i].f[j] = 0;
    }
  }
#pragma HLS ARRAY_PARTITION variable=indata complete dim=1
//#pragma HLS ARRAY_PARTITION variable=indata->f complete dim=1

#pragma HLS ARRAY_PARTITION variable=outdata complete dim=1
//#pragma HLS ARRAY_PARTITION variable=outdata->f complete dim=1
  score_t res = 0;
  //main function
  for(int stage = 0; stage <= NUMOF_VARS; stage++){
    PE(a[0], stage, dataset, nof_vars[0], indata[0], outdata[0]);
    PE(a[1], stage, dataset2, nof_vars[1], indata[1], outdata[1]);
    PE(a[2], stage, dataset3, nof_vars[2], indata[2], outdata[2]);
    PE(a[3], stage, dataset4, nof_vars[3], indata[3], outdata[3]);
    PE(a[4], stage, dataset5, nof_vars[4], indata[4], outdata[4]);
    PE(a[5], stage, dataset6, nof_vars[5], indata[5], outdata[5]);
    PE(a[6], stage, dataset7, nof_vars[6], indata[6], outdata[6]);
    PE(a[7], stage, dataset8, nof_vars[7], indata[7], outdata[7]);
    PE(a[8], stage, dataset9, nof_vars[8], indata[8], outdata[8]);
    PE(a[9], stage, dataset10, nof_vars[9], indata[9], outdata[9]);
    PE(a[10], stage, dataset11, nof_vars[10], indata[10], outdata[10]);
    PE(a[11], stage, dataset12, nof_vars[11], indata[11], outdata[11]);
    PE(a[12], stage, dataset13, nof_vars[12], indata[12], outdata[12]);
    PE(a[13], stage, dataset14, nof_vars[13], indata[13], outdata[13]);
    PE(a[14], stage, dataset15, nof_vars[14], indata[14], outdata[14]);
    PE(a[15], stage, dataset16, nof_vars[15], indata[15], outdata[15]);
    PE(a[16], stage, dataset17, nof_vars[16], indata[16], outdata[16]);
    PE(a[17], stage, dataset18, nof_vars[17], indata[17], outdata[17]);
    PE(a[18], stage, dataset19, nof_vars[18], indata[18], outdata[18]);
    PE(a[19], stage, dataset20, nof_vars[19], indata[19], outdata[19]);
    PE(a[20], stage, dataset21, nof_vars[20], indata[20], outdata[20]);
    PE(a[21], stage, dataset22, nof_vars[21], indata[21], outdata[21]);
    PE(a[22], stage, dataset23, nof_vars[22], indata[22], outdata[22]);
    PE(a[23], stage, dataset24, nof_vars[23], indata[23], outdata[23]);
    PE(a[24], stage, dataset25, nof_vars[24], indata[24], outdata[24]);
    PE(a[25], stage, dataset26, nof_vars[25], indata[25], outdata[25]);
    PE(a[26], stage, dataset27, nof_vars[26], indata[26], outdata[26]);
    PE(a[27], stage, dataset28, nof_vars[27], indata[27], outdata[27]);
    PE(a[28], stage, dataset29, nof_vars[28], indata[28], outdata[28]);
    PE(a[29], stage, dataset30, nof_vars[29], indata[29], outdata[29]);
    PE(a[30], stage, dataset31, nof_vars[30], indata[30], outdata[30]);
    PE(a[31], stage, dataset32, nof_vars[31], indata[31], outdata[31]);
    PE(a[32], stage, dataset33, nof_vars[32], indata[32], outdata[32]);
    PE(a[33], stage, dataset34, nof_vars[33], indata[33], outdata[33]);
    PE(a[34], stage, dataset35, nof_vars[34], indata[34], outdata[34]);
    PE(a[35], stage, dataset36, nof_vars[35], indata[35], outdata[35]);
    PE(a[36], stage, dataset37, nof_vars[36], indata[36], outdata[36]);
    PE(a[37], stage, dataset38, nof_vars[37], indata[37], outdata[37]);
    PE(a[38], stage, dataset39, nof_vars[38], indata[38], outdata[38]);
    PE(a[39], stage, dataset40, nof_vars[39], indata[39], outdata[39]);
    PE(a[40], stage, dataset41, nof_vars[40], indata[40], outdata[40]);
    PE(a[41], stage, dataset42, nof_vars[41], indata[41], outdata[41]);
    PE(a[42], stage, dataset43, nof_vars[42], indata[42], outdata[42]);
    PE(a[43], stage, dataset44, nof_vars[43], indata[43], outdata[43]);
    PE(a[44], stage, dataset45, nof_vars[44], indata[44], outdata[44]);
    PE(a[45], stage, dataset46, nof_vars[45], indata[45], outdata[45]);
    PE(a[46], stage, dataset47, nof_vars[46], indata[46], outdata[46]);
    PE(a[47], stage, dataset48, nof_vars[47], indata[47], outdata[47]);
    PE(a[48], stage, dataset49, nof_vars[48], indata[48], outdata[48]);
    PE(a[49], stage, dataset50, nof_vars[49], indata[49], outdata[49]);
    PE(a[50], stage, dataset51, nof_vars[50], indata[50], outdata[50]);
    PE(a[51], stage, dataset52, nof_vars[51], indata[51], outdata[51]);
    PE(a[52], stage, dataset53, nof_vars[52], indata[52], outdata[52]);
    PE(a[53], stage, dataset54, nof_vars[53], indata[53], outdata[53]);
    PE(a[54], stage, dataset55, nof_vars[54], indata[54], outdata[54]);
    PE(a[55], stage, dataset56, nof_vars[55], indata[55], outdata[55]);
    PE(a[56], stage, dataset57, nof_vars[56], indata[56], outdata[56]);
    PE(a[57], stage, dataset58, nof_vars[57], indata[57], outdata[57]);
    PE(a[58], stage, dataset59, nof_vars[58], indata[58], outdata[58]);
    PE(a[59], stage, dataset60, nof_vars[59], indata[59], outdata[59]);
    PE(a[60], stage, dataset61, nof_vars[60], indata[60], outdata[60]);
    PE(a[61], stage, dataset62, nof_vars[61], indata[61], outdata[61]);
    PE(a[62], stage, dataset63, nof_vars[62], indata[62], outdata[62]);
    PE(a[63], stage, dataset64, nof_vars[63], indata[63], outdata[63]);
    PE(a[64], stage, dataset65, nof_vars[64], indata[64], outdata[64]);
    PE(a[65], stage, dataset66, nof_vars[65], indata[65], outdata[65]);
    PE(a[66], stage, dataset67, nof_vars[66], indata[66], outdata[66]);
    PE(a[67], stage, dataset68, nof_vars[67], indata[67], outdata[67]);
    PE(a[68], stage, dataset69, nof_vars[68], indata[68], outdata[68]);
    PE(a[69], stage, dataset70, nof_vars[69], indata[69], outdata[69]);

    if(stage < NUMOF_VARS){
      //read Next A
      loop_read_nextA:for(int i = 0; i < PE_NUM; i++){
#pragma HLS UNROLL
        a[i] = A[stage+1][i];
      }
      //prepare next input
      int indexcnt[PE_NUM];
      loop_initialize_indexcnt:for(int i = 0; i < PE_NUM; i++){
#pragma HLS UNROLL
        indexcnt[i] = 0;
      }
      for(int j = 0; j < PE_NUM; j++){//nC(stage+1)
        //for each j, all "index" values calculated by k are different
        for(int k = NUMOF_VARS - 1; k >= 0; k--){
#pragma HLS UNROLL
#pragma HLS dependence variable=outdata inter false
          if(a[j] < 0) continue;
          if((a[j] >> k) & 1U){
            int tmp = a[j]^(1U << k);
            //cout << j << " " << a[j] << " " << tmp << endl;
            int index = val2index(a[j]^(1U << k), stage);
            pe_out data = outdata[index];
            //cout << "indata[" << j << "][" << indexcnt[j] << "] = " << "outdata[" << index << "]" << " " << data.q.to_string(10) << endl;
            indata[j][indexcnt[j]] = data;
            indexcnt[j]++;
          }
        }
      }
    }else{
      //final output
      res = outdata[0].q;
    }
  }

  *p_best_score = (float)res;
}

}
