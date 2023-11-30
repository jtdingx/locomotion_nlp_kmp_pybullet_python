
/*
Auto-generated by CVXPYgen on November 29, 2023 at 21:15:35.
Content: Function definitions.
*/

#include "cpg_solve.h"
#include "cpg_workspace.h"

static c_int i;
static c_int j;
static c_int initialized = 0;

// Update user-defined parameters
void cpg_update_X_ref(c_int idx, c_float val){
  cpg_params_vec[idx+0] = val;
  Canon_Outdated.b = 1;
  Canon_Outdated.h = 1;
}

void cpg_update_x_init(c_int idx, c_float val){
  cpg_params_vec[idx+84] = val;
  Canon_Outdated.b = 1;
}

void cpg_update_A_dyn(c_int idx, c_float val){
  cpg_params_vec[idx+96] = val;
  Canon_Outdated.A = 1;
}

void cpg_update_Inertial_matrix(c_int idx, c_float val){
  cpg_params_vec[idx+1104] = val;
  Canon_Outdated.A = 1;
  Canon_Outdated.G = 1;
}

// Map user-defined to canonical parameters
void cpg_canonicalize_A(){
  for(i=0; i<1251; i++){
    Canon_Params.A->x[i] = 0;
    for(j=canon_A_map.p[i]; j<canon_A_map.p[i+1]; j++){
      Canon_Params.A->x[i] += canon_A_map.x[j]*cpg_params_vec[canon_A_map.i[j]];
    }
  }
}

void cpg_canonicalize_b(){
  for(i=0; i<117; i++){
    Canon_Params.b[i] = 0;
    for(j=canon_b_map.p[i]; j<canon_b_map.p[i+1]; j++){
      Canon_Params.b[i] += canon_b_map.x[j]*cpg_params_vec[canon_b_map.i[j]];
    }
  }
}

void cpg_canonicalize_G(){
  for(i=0; i<202; i++){
    Canon_Params.G->x[i] = 0;
    for(j=canon_G_map.p[i]; j<canon_G_map.p[i+1]; j++){
      Canon_Params.G->x[i] += canon_G_map.x[j]*cpg_params_vec[canon_G_map.i[j]];
    }
  }
}

void cpg_canonicalize_h(){
  for(i=0; i<202; i++){
    Canon_Params.h[i] = 0;
    for(j=canon_h_map.p[i]; j<canon_h_map.p[i+1]; j++){
      Canon_Params.h[i] += canon_h_map.x[j]*cpg_params_vec[canon_h_map.i[j]];
    }
  }
}

// Retrieve primal solution in terms of user-defined variables
void cpg_retrieve_prim(){
  CPG_Prim.X[0] = ecos_workspace->x[3];
  CPG_Prim.X[1] = ecos_workspace->x[4];
  CPG_Prim.X[2] = ecos_workspace->x[5];
  CPG_Prim.X[3] = ecos_workspace->x[6];
  CPG_Prim.X[4] = ecos_workspace->x[7];
  CPG_Prim.X[5] = ecos_workspace->x[8];
  CPG_Prim.X[6] = ecos_workspace->x[9];
  CPG_Prim.X[7] = ecos_workspace->x[10];
  CPG_Prim.X[8] = ecos_workspace->x[11];
  CPG_Prim.X[9] = ecos_workspace->x[12];
  CPG_Prim.X[10] = ecos_workspace->x[13];
  CPG_Prim.X[11] = ecos_workspace->x[14];
  CPG_Prim.X[12] = ecos_workspace->x[15];
  CPG_Prim.X[13] = ecos_workspace->x[16];
  CPG_Prim.X[14] = ecos_workspace->x[17];
  CPG_Prim.X[15] = ecos_workspace->x[18];
  CPG_Prim.X[16] = ecos_workspace->x[19];
  CPG_Prim.X[17] = ecos_workspace->x[20];
  CPG_Prim.X[18] = ecos_workspace->x[21];
  CPG_Prim.X[19] = ecos_workspace->x[22];
  CPG_Prim.X[20] = ecos_workspace->x[23];
  CPG_Prim.X[21] = ecos_workspace->x[24];
  CPG_Prim.X[22] = ecos_workspace->x[25];
  CPG_Prim.X[23] = ecos_workspace->x[26];
  CPG_Prim.X[24] = ecos_workspace->x[27];
  CPG_Prim.X[25] = ecos_workspace->x[28];
  CPG_Prim.X[26] = ecos_workspace->x[29];
  CPG_Prim.X[27] = ecos_workspace->x[30];
  CPG_Prim.X[28] = ecos_workspace->x[31];
  CPG_Prim.X[29] = ecos_workspace->x[32];
  CPG_Prim.X[30] = ecos_workspace->x[33];
  CPG_Prim.X[31] = ecos_workspace->x[34];
  CPG_Prim.X[32] = ecos_workspace->x[35];
  CPG_Prim.X[33] = ecos_workspace->x[36];
  CPG_Prim.X[34] = ecos_workspace->x[37];
  CPG_Prim.X[35] = ecos_workspace->x[38];
  CPG_Prim.X[36] = ecos_workspace->x[39];
  CPG_Prim.X[37] = ecos_workspace->x[40];
  CPG_Prim.X[38] = ecos_workspace->x[41];
  CPG_Prim.X[39] = ecos_workspace->x[42];
  CPG_Prim.X[40] = ecos_workspace->x[43];
  CPG_Prim.X[41] = ecos_workspace->x[44];
  CPG_Prim.X[42] = ecos_workspace->x[45];
  CPG_Prim.X[43] = ecos_workspace->x[46];
  CPG_Prim.X[44] = ecos_workspace->x[47];
  CPG_Prim.X[45] = ecos_workspace->x[48];
  CPG_Prim.X[46] = ecos_workspace->x[49];
  CPG_Prim.X[47] = ecos_workspace->x[50];
  CPG_Prim.X[48] = ecos_workspace->x[51];
  CPG_Prim.X[49] = ecos_workspace->x[52];
  CPG_Prim.X[50] = ecos_workspace->x[53];
  CPG_Prim.X[51] = ecos_workspace->x[54];
  CPG_Prim.X[52] = ecos_workspace->x[55];
  CPG_Prim.X[53] = ecos_workspace->x[56];
  CPG_Prim.X[54] = ecos_workspace->x[57];
  CPG_Prim.X[55] = ecos_workspace->x[58];
  CPG_Prim.X[56] = ecos_workspace->x[59];
  CPG_Prim.X[57] = ecos_workspace->x[60];
  CPG_Prim.X[58] = ecos_workspace->x[61];
  CPG_Prim.X[59] = ecos_workspace->x[62];
  CPG_Prim.X[60] = ecos_workspace->x[63];
  CPG_Prim.X[61] = ecos_workspace->x[64];
  CPG_Prim.X[62] = ecos_workspace->x[65];
  CPG_Prim.X[63] = ecos_workspace->x[66];
  CPG_Prim.X[64] = ecos_workspace->x[67];
  CPG_Prim.X[65] = ecos_workspace->x[68];
  CPG_Prim.X[66] = ecos_workspace->x[69];
  CPG_Prim.X[67] = ecos_workspace->x[70];
  CPG_Prim.X[68] = ecos_workspace->x[71];
  CPG_Prim.X[69] = ecos_workspace->x[72];
  CPG_Prim.X[70] = ecos_workspace->x[73];
  CPG_Prim.X[71] = ecos_workspace->x[74];
  CPG_Prim.X[72] = ecos_workspace->x[75];
  CPG_Prim.X[73] = ecos_workspace->x[76];
  CPG_Prim.X[74] = ecos_workspace->x[77];
  CPG_Prim.X[75] = ecos_workspace->x[78];
  CPG_Prim.X[76] = ecos_workspace->x[79];
  CPG_Prim.X[77] = ecos_workspace->x[80];
  CPG_Prim.X[78] = ecos_workspace->x[81];
  CPG_Prim.X[79] = ecos_workspace->x[82];
  CPG_Prim.X[80] = ecos_workspace->x[83];
  CPG_Prim.X[81] = ecos_workspace->x[84];
  CPG_Prim.X[82] = ecos_workspace->x[85];
  CPG_Prim.X[83] = ecos_workspace->x[86];
  CPG_Prim.X[84] = ecos_workspace->x[87];
  CPG_Prim.X[85] = ecos_workspace->x[88];
  CPG_Prim.X[86] = ecos_workspace->x[89];
  CPG_Prim.X[87] = ecos_workspace->x[90];
  CPG_Prim.X[88] = ecos_workspace->x[91];
  CPG_Prim.X[89] = ecos_workspace->x[92];
  CPG_Prim.X[90] = ecos_workspace->x[93];
  CPG_Prim.X[91] = ecos_workspace->x[94];
  CPG_Prim.X[92] = ecos_workspace->x[95];
  CPG_Prim.X[93] = ecos_workspace->x[96];
  CPG_Prim.X[94] = ecos_workspace->x[97];
  CPG_Prim.X[95] = ecos_workspace->x[98];
  CPG_Prim.U[0] = ecos_workspace->x[99];
  CPG_Prim.U[1] = ecos_workspace->x[100];
  CPG_Prim.U[2] = ecos_workspace->x[101];
  CPG_Prim.U[3] = ecos_workspace->x[102];
  CPG_Prim.U[4] = ecos_workspace->x[103];
  CPG_Prim.U[5] = ecos_workspace->x[104];
  CPG_Prim.U[6] = ecos_workspace->x[105];
  CPG_Prim.U[7] = ecos_workspace->x[106];
  CPG_Prim.U[8] = ecos_workspace->x[107];
  CPG_Prim.U[9] = ecos_workspace->x[108];
  CPG_Prim.U[10] = ecos_workspace->x[109];
  CPG_Prim.U[11] = ecos_workspace->x[110];
  CPG_Prim.U[12] = ecos_workspace->x[111];
  CPG_Prim.U[13] = ecos_workspace->x[112];
  CPG_Prim.U[14] = ecos_workspace->x[113];
  CPG_Prim.U[15] = ecos_workspace->x[114];
  CPG_Prim.U[16] = ecos_workspace->x[115];
  CPG_Prim.U[17] = ecos_workspace->x[116];
  CPG_Prim.U[18] = ecos_workspace->x[117];
  CPG_Prim.U[19] = ecos_workspace->x[118];
  CPG_Prim.U[20] = ecos_workspace->x[119];
  CPG_Prim.U[21] = ecos_workspace->x[120];
  CPG_Prim.U[22] = ecos_workspace->x[121];
  CPG_Prim.U[23] = ecos_workspace->x[122];
  CPG_Prim.U[24] = ecos_workspace->x[123];
  CPG_Prim.U[25] = ecos_workspace->x[124];
  CPG_Prim.U[26] = ecos_workspace->x[125];
  CPG_Prim.U[27] = ecos_workspace->x[126];
  CPG_Prim.U[28] = ecos_workspace->x[127];
  CPG_Prim.U[29] = ecos_workspace->x[128];
  CPG_Prim.U[30] = ecos_workspace->x[129];
  CPG_Prim.U[31] = ecos_workspace->x[130];
  CPG_Prim.U[32] = ecos_workspace->x[131];
  CPG_Prim.U[33] = ecos_workspace->x[132];
  CPG_Prim.U[34] = ecos_workspace->x[133];
  CPG_Prim.U[35] = ecos_workspace->x[134];
  CPG_Prim.U[36] = ecos_workspace->x[135];
  CPG_Prim.U[37] = ecos_workspace->x[136];
  CPG_Prim.U[38] = ecos_workspace->x[137];
  CPG_Prim.U[39] = ecos_workspace->x[138];
  CPG_Prim.U[40] = ecos_workspace->x[139];
  CPG_Prim.U[41] = ecos_workspace->x[140];
  CPG_Prim.X_cmp[0] = ecos_workspace->x[141];
  CPG_Prim.X_cmp[1] = ecos_workspace->x[142];
  CPG_Prim.X_cmp[2] = ecos_workspace->x[143];
  CPG_Prim.X_cmp[3] = ecos_workspace->x[144];
  CPG_Prim.X_cmp[4] = ecos_workspace->x[145];
  CPG_Prim.X_cmp[5] = ecos_workspace->x[146];
  CPG_Prim.X_cmp[6] = ecos_workspace->x[147];
  CPG_Prim.X_cmp[7] = ecos_workspace->x[148];
  CPG_Prim.X_cmp[8] = ecos_workspace->x[149];
  CPG_Prim.X_cmp[9] = ecos_workspace->x[150];
  CPG_Prim.X_cmp[10] = ecos_workspace->x[151];
  CPG_Prim.X_cmp[11] = ecos_workspace->x[152];
  CPG_Prim.X_cmp[12] = ecos_workspace->x[153];
  CPG_Prim.X_cmp[13] = ecos_workspace->x[154];
  CPG_Prim.X_cmp[14] = ecos_workspace->x[155];
  CPG_Prim.X_cmp[15] = ecos_workspace->x[156];
}

// Retrieve dual solution in terms of user-defined constraints
void cpg_retrieve_dual(){
  CPG_Dual.d0[0] = ecos_workspace->y[0];
  CPG_Dual.d0[1] = ecos_workspace->y[1];
  CPG_Dual.d0[2] = ecos_workspace->y[2];
  CPG_Dual.d0[3] = ecos_workspace->y[3];
  CPG_Dual.d0[4] = ecos_workspace->y[4];
  CPG_Dual.d0[5] = ecos_workspace->y[5];
  CPG_Dual.d0[6] = ecos_workspace->y[6];
  CPG_Dual.d0[7] = ecos_workspace->y[7];
  CPG_Dual.d0[8] = ecos_workspace->y[8];
  CPG_Dual.d0[9] = ecos_workspace->y[9];
  CPG_Dual.d0[10] = ecos_workspace->y[10];
  CPG_Dual.d0[11] = ecos_workspace->y[11];
  CPG_Dual.d1[0] = ecos_workspace->y[12];
  CPG_Dual.d1[1] = ecos_workspace->y[13];
  CPG_Dual.d1[2] = ecos_workspace->y[14];
  CPG_Dual.d1[3] = ecos_workspace->y[15];
  CPG_Dual.d1[4] = ecos_workspace->y[16];
  CPG_Dual.d1[5] = ecos_workspace->y[17];
  CPG_Dual.d1[6] = ecos_workspace->y[18];
  CPG_Dual.d1[7] = ecos_workspace->y[19];
  CPG_Dual.d1[8] = ecos_workspace->y[20];
  CPG_Dual.d1[9] = ecos_workspace->y[21];
  CPG_Dual.d1[10] = ecos_workspace->y[22];
  CPG_Dual.d1[11] = ecos_workspace->y[23];
  CPG_Dual.d2 = ecos_workspace->y[24];
  CPG_Dual.d3 = ecos_workspace->y[25];
  CPG_Dual.d4 = ecos_workspace->y[0];
  CPG_Dual.d5 = ecos_workspace->y[1];
  CPG_Dual.d6 = ecos_workspace->y[2];
  CPG_Dual.d7 = ecos_workspace->y[3];
  CPG_Dual.d8 = ecos_workspace->y[4];
  CPG_Dual.d9 = ecos_workspace->y[5];
  CPG_Dual.d10 = ecos_workspace->y[6];
  CPG_Dual.d11 = ecos_workspace->y[7];
  CPG_Dual.d12 = ecos_workspace->y[26];
  CPG_Dual.d13[0] = ecos_workspace->y[27];
  CPG_Dual.d13[1] = ecos_workspace->y[28];
  CPG_Dual.d13[2] = ecos_workspace->y[29];
  CPG_Dual.d13[3] = ecos_workspace->y[30];
  CPG_Dual.d13[4] = ecos_workspace->y[31];
  CPG_Dual.d13[5] = ecos_workspace->y[32];
  CPG_Dual.d13[6] = ecos_workspace->y[33];
  CPG_Dual.d13[7] = ecos_workspace->y[34];
  CPG_Dual.d13[8] = ecos_workspace->y[35];
  CPG_Dual.d13[9] = ecos_workspace->y[36];
  CPG_Dual.d13[10] = ecos_workspace->y[37];
  CPG_Dual.d13[11] = ecos_workspace->y[38];
  CPG_Dual.d14 = ecos_workspace->y[39];
  CPG_Dual.d15 = ecos_workspace->y[40];
  CPG_Dual.d16 = ecos_workspace->y[8];
  CPG_Dual.d17 = ecos_workspace->y[9];
  CPG_Dual.d18 = ecos_workspace->y[10];
  CPG_Dual.d19 = ecos_workspace->y[11];
  CPG_Dual.d20 = ecos_workspace->y[12];
  CPG_Dual.d21 = ecos_workspace->y[13];
  CPG_Dual.d22 = ecos_workspace->y[14];
  CPG_Dual.d23 = ecos_workspace->y[15];
  CPG_Dual.d24 = ecos_workspace->y[41];
  CPG_Dual.d25[0] = ecos_workspace->y[42];
  CPG_Dual.d25[1] = ecos_workspace->y[43];
  CPG_Dual.d25[2] = ecos_workspace->y[44];
  CPG_Dual.d25[3] = ecos_workspace->y[45];
  CPG_Dual.d25[4] = ecos_workspace->y[46];
  CPG_Dual.d25[5] = ecos_workspace->y[47];
  CPG_Dual.d25[6] = ecos_workspace->y[48];
  CPG_Dual.d25[7] = ecos_workspace->y[49];
  CPG_Dual.d25[8] = ecos_workspace->y[50];
  CPG_Dual.d25[9] = ecos_workspace->y[51];
  CPG_Dual.d25[10] = ecos_workspace->y[52];
  CPG_Dual.d25[11] = ecos_workspace->y[53];
  CPG_Dual.d26 = ecos_workspace->y[54];
  CPG_Dual.d27 = ecos_workspace->y[55];
  CPG_Dual.d28 = ecos_workspace->y[16];
  CPG_Dual.d29 = ecos_workspace->z[17];
  CPG_Dual.d30 = ecos_workspace->z[18];
  CPG_Dual.d31 = ecos_workspace->z[19];
  CPG_Dual.d32 = ecos_workspace->z[20];
  CPG_Dual.d33 = ecos_workspace->z[21];
  CPG_Dual.d34 = ecos_workspace->z[22];
  CPG_Dual.d35 = ecos_workspace->z[23];
  CPG_Dual.d36 = ecos_workspace->z[56];
  CPG_Dual.d37[0] = ecos_workspace->z[57];
  CPG_Dual.d37[1] = ecos_workspace->z[58];
  CPG_Dual.d37[2] = ecos_workspace->z[59];
  CPG_Dual.d37[3] = ecos_workspace->z[60];
  CPG_Dual.d37[4] = ecos_workspace->z[61];
  CPG_Dual.d37[5] = ecos_workspace->z[62];
  CPG_Dual.d37[6] = ecos_workspace->z[63];
  CPG_Dual.d37[7] = ecos_workspace->z[64];
  CPG_Dual.d37[8] = ecos_workspace->z[65];
  CPG_Dual.d37[9] = ecos_workspace->z[66];
  CPG_Dual.d37[10] = ecos_workspace->z[67];
  CPG_Dual.d37[11] = ecos_workspace->z[68];
  CPG_Dual.d38 = ecos_workspace->z[69];
  CPG_Dual.d39 = ecos_workspace->z[70];
  CPG_Dual.d40 = ecos_workspace->z[24];
  CPG_Dual.d41 = ecos_workspace->z[25];
  CPG_Dual.d42 = ecos_workspace->z[26];
  CPG_Dual.d43 = ecos_workspace->z[27];
  CPG_Dual.d44 = ecos_workspace->z[28];
  CPG_Dual.d45 = ecos_workspace->z[29];
  CPG_Dual.d46 = ecos_workspace->z[30];
  CPG_Dual.d47 = ecos_workspace->z[31];
  CPG_Dual.d48 = ecos_workspace->z[71];
  CPG_Dual.d49[0] = ecos_workspace->z[72];
  CPG_Dual.d49[1] = ecos_workspace->z[73];
  CPG_Dual.d49[2] = ecos_workspace->z[74];
  CPG_Dual.d49[3] = ecos_workspace->z[75];
  CPG_Dual.d49[4] = ecos_workspace->z[76];
  CPG_Dual.d49[5] = ecos_workspace->z[77];
  CPG_Dual.d49[6] = ecos_workspace->z[78];
  CPG_Dual.d49[7] = ecos_workspace->z[79];
  CPG_Dual.d49[8] = ecos_workspace->z[80];
  CPG_Dual.d49[9] = ecos_workspace->z[81];
  CPG_Dual.d49[10] = ecos_workspace->z[82];
  CPG_Dual.d49[11] = ecos_workspace->z[83];
  CPG_Dual.d50 = ecos_workspace->z[84];
  CPG_Dual.d51 = ecos_workspace->z[85];
  CPG_Dual.d52 = ecos_workspace->z[32];
  CPG_Dual.d53 = ecos_workspace->z[33];
  CPG_Dual.d54 = ecos_workspace->z[34];
  CPG_Dual.d55 = ecos_workspace->z[35];
  CPG_Dual.d56 = ecos_workspace->z[36];
  CPG_Dual.d57 = ecos_workspace->z[37];
  CPG_Dual.d58 = ecos_workspace->z[38];
  CPG_Dual.d59 = ecos_workspace->z[39];
  CPG_Dual.d60 = ecos_workspace->z[86];
  CPG_Dual.d61[0] = ecos_workspace->z[87];
  CPG_Dual.d61[1] = ecos_workspace->z[88];
  CPG_Dual.d61[2] = ecos_workspace->z[89];
  CPG_Dual.d61[3] = ecos_workspace->z[90];
  CPG_Dual.d61[4] = ecos_workspace->z[91];
  CPG_Dual.d61[5] = ecos_workspace->z[92];
  CPG_Dual.d61[6] = ecos_workspace->z[93];
  CPG_Dual.d61[7] = ecos_workspace->z[94];
  CPG_Dual.d61[8] = ecos_workspace->z[95];
  CPG_Dual.d61[9] = ecos_workspace->z[96];
  CPG_Dual.d61[10] = ecos_workspace->z[97];
  CPG_Dual.d61[11] = ecos_workspace->z[98];
  CPG_Dual.d62 = ecos_workspace->z[99];
  CPG_Dual.d63 = ecos_workspace->z[100];
  CPG_Dual.d64 = ecos_workspace->z[40];
  CPG_Dual.d65 = ecos_workspace->z[41];
  CPG_Dual.d66 = ecos_workspace->z[42];
  CPG_Dual.d67 = ecos_workspace->z[43];
  CPG_Dual.d68 = ecos_workspace->z[44];
  CPG_Dual.d69 = ecos_workspace->z[45];
  CPG_Dual.d70 = ecos_workspace->z[46];
  CPG_Dual.d71 = ecos_workspace->z[47];
  CPG_Dual.d72 = ecos_workspace->z[101];
  CPG_Dual.d73[0] = ecos_workspace->z[102];
  CPG_Dual.d73[1] = ecos_workspace->z[103];
  CPG_Dual.d73[2] = ecos_workspace->z[104];
  CPG_Dual.d73[3] = ecos_workspace->z[105];
  CPG_Dual.d73[4] = ecos_workspace->z[106];
  CPG_Dual.d73[5] = ecos_workspace->z[107];
  CPG_Dual.d73[6] = ecos_workspace->z[108];
  CPG_Dual.d73[7] = ecos_workspace->z[109];
  CPG_Dual.d73[8] = ecos_workspace->z[110];
  CPG_Dual.d73[9] = ecos_workspace->z[111];
  CPG_Dual.d73[10] = ecos_workspace->z[112];
  CPG_Dual.d73[11] = ecos_workspace->z[113];
  CPG_Dual.d74 = ecos_workspace->z[114];
  CPG_Dual.d75 = ecos_workspace->z[115];
  CPG_Dual.d76 = ecos_workspace->z[48];
  CPG_Dual.d77 = ecos_workspace->z[49];
  CPG_Dual.d78 = ecos_workspace->z[50];
  CPG_Dual.d79 = ecos_workspace->z[51];
  CPG_Dual.d80 = ecos_workspace->z[52];
  CPG_Dual.d81 = ecos_workspace->z[53];
  CPG_Dual.d82 = ecos_workspace->z[54];
  CPG_Dual.d83 = ecos_workspace->z[55];
  CPG_Dual.d84 = ecos_workspace->z[116];
}

// Retrieve solver info
void cpg_retrieve_info(){
  CPG_Info.obj_val = ecos_workspace->info->pcost;
  CPG_Info.iter = ecos_workspace->info->iter;
  CPG_Info.status = ecos_flag;
  CPG_Info.pri_res = ecos_workspace->info->pres;
  CPG_Info.dua_res = ecos_workspace->info->dres;
}

// Solve via canonicalization, canonical solve, retrieval
void cpg_solve(){
  // Canonicalize if necessary
  if (Canon_Outdated.A) {
    cpg_canonicalize_A();
  }
  if (Canon_Outdated.b) {
    cpg_canonicalize_b();
  }
  if (Canon_Outdated.G) {
    cpg_canonicalize_G();
  }
  if (Canon_Outdated.h) {
    cpg_canonicalize_h();
  }
  for (i=0; i<157; i++){
    Canon_Params_ECOS.c[i] = Canon_Params.c[i];
  }
  Canon_Params_ECOS.d = Canon_Params.d;
  for (i=0; i<1251; i++){
    Canon_Params_ECOS.A->x[i] = Canon_Params.A->x[i];
  }
  for (i=0; i<117; i++){
    Canon_Params_ECOS.b[i] = Canon_Params.b[i];
  }
  for (i=0; i<202; i++){
    Canon_Params_ECOS.G->x[i] = Canon_Params.G->x[i];
  }
  for (i=0; i<202; i++){
    Canon_Params_ECOS.h[i] = Canon_Params.h[i];
  }
  // Initialize / update ECOS workspace and settings
  if (!initialized) {
    ecos_workspace = ECOS_setup(157, 202, 117, 56, 3, (int *) &ecos_q, 0, Canon_Params_ECOS.G->x, Canon_Params_ECOS.G->p, Canon_Params_ECOS.G->i, Canon_Params_ECOS.A->x, Canon_Params_ECOS.A->p, Canon_Params_ECOS.A->i, Canon_Params_ECOS.c, Canon_Params_ECOS.h, Canon_Params_ECOS.b);
    initialized = 1;
  } else {
    if (Canon_Outdated.G || Canon_Outdated.A || Canon_Outdated.b) {
      ECOS_updateData(ecos_workspace, Canon_Params_ECOS.G->x, Canon_Params_ECOS.A->x, Canon_Params_ECOS.c, Canon_Params_ECOS.h, Canon_Params_ECOS.b);
    } else {
      if (Canon_Outdated.h) {
        for (i=0; i<202; i++){
          ecos_updateDataEntry_h(ecos_workspace, i, Canon_Params_ECOS.h[i]);
        }
      }
      if (Canon_Outdated.c) {
        for (i=0; i<157; i++){
          ecos_updateDataEntry_c(ecos_workspace, i, Canon_Params_ECOS.c[i]);
        }
      }
    }
  }
  ecos_workspace->stgs->feastol = Canon_Settings.feastol;
  ecos_workspace->stgs->abstol = Canon_Settings.abstol;
  ecos_workspace->stgs->reltol = Canon_Settings.reltol;
  ecos_workspace->stgs->feastol_inacc = Canon_Settings.feastol_inacc;
  ecos_workspace->stgs->abstol_inacc = Canon_Settings.abstol_inacc;
  ecos_workspace->stgs->reltol_inacc = Canon_Settings.reltol_inacc;
  ecos_workspace->stgs->maxit = Canon_Settings.maxit;
  // Solve with ECOS
  ecos_flag = ECOS_solve(ecos_workspace);
  // Retrieve results
  cpg_retrieve_prim();
  cpg_retrieve_dual();
  cpg_retrieve_info();
  // Reset flags for outdated canonical parameters
  Canon_Outdated.c = 0;
  Canon_Outdated.d = 0;
  Canon_Outdated.A = 0;
  Canon_Outdated.b = 0;
  Canon_Outdated.G = 0;
  Canon_Outdated.h = 0;
}

// Update solver settings
void cpg_set_solver_default_settings(){
  Canon_Settings.feastol = 1e-8;
  Canon_Settings.abstol = 1e-8;
  Canon_Settings.reltol = 1e-8;
  Canon_Settings.feastol_inacc = 1e-4;
  Canon_Settings.abstol_inacc = 5e-5;
  Canon_Settings.reltol_inacc = 5e-5;
  Canon_Settings.maxit = 100;
}

void cpg_set_solver_feastol(c_float feastol_new){
  Canon_Settings.feastol = feastol_new;
}

void cpg_set_solver_abstol(c_float abstol_new){
  Canon_Settings.abstol = abstol_new;
}

void cpg_set_solver_reltol(c_float reltol_new){
  Canon_Settings.reltol = reltol_new;
}

void cpg_set_solver_feastol_inacc(c_float feastol_inacc_new){
  Canon_Settings.feastol_inacc = feastol_inacc_new;
}

void cpg_set_solver_abstol_inacc(c_float abstol_inacc_new){
  Canon_Settings.abstol_inacc = abstol_inacc_new;
}

void cpg_set_solver_reltol_inacc(c_float reltol_inacc_new){
  Canon_Settings.reltol_inacc = reltol_inacc_new;
}

void cpg_set_solver_maxit(c_int maxit_new){
  Canon_Settings.maxit = maxit_new;
}
