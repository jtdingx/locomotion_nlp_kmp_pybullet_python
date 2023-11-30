
/*
Auto-generated by CVXPYgen on November 29, 2023 at 21:15:35.
Content: Python binding with pybind11.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ctime>
#include "cpg_module.hpp"

extern "C" {
    #include "include/cpg_workspace.h"
    #include "include/cpg_solve.h"
}

namespace py = pybind11;

static int i;

CPG_Result_cpp_t solve_cpp(struct CPG_Updated_cpp_t& CPG_Updated_cpp, struct CPG_Params_cpp_t& CPG_Params_cpp){

    // Pass changed user-defined parameter values to the solver
    if (CPG_Updated_cpp.X_ref) {
        for(i=0; i<84; i++) {
            cpg_update_X_ref(i, CPG_Params_cpp.X_ref[i]);
        }
    }
    if (CPG_Updated_cpp.x_init) {
        for(i=0; i<12; i++) {
            cpg_update_x_init(i, CPG_Params_cpp.x_init[i]);
        }
    }
    if (CPG_Updated_cpp.A_dyn) {
        for(i=0; i<1008; i++) {
            cpg_update_A_dyn(i, CPG_Params_cpp.A_dyn[i]);
        }
    }
    if (CPG_Updated_cpp.Inertial_matrix) {
        for(i=0; i<9; i++) {
            cpg_update_Inertial_matrix(i, CPG_Params_cpp.Inertial_matrix[i]);
        }
    }

    // Solve
    std::clock_t ASA_start = std::clock();
    cpg_solve();
    std::clock_t ASA_end = std::clock();

    // Arrange and return results
    CPG_Prim_cpp_t CPG_Prim_cpp {};
    for(i=0; i<96; i++) {
        CPG_Prim_cpp.X[i] = CPG_Prim.X[i];
    }
    for(i=0; i<42; i++) {
        CPG_Prim_cpp.U[i] = CPG_Prim.U[i];
    }
    for(i=0; i<16; i++) {
        CPG_Prim_cpp.X_cmp[i] = CPG_Prim.X_cmp[i];
    }
    CPG_Dual_cpp_t CPG_Dual_cpp {};
    for(i=0; i<12; i++) {
        CPG_Dual_cpp.d0[i] = CPG_Dual.d0[i];
    }
    for(i=0; i<12; i++) {
        CPG_Dual_cpp.d1[i] = CPG_Dual.d1[i];
    }
    CPG_Dual_cpp.d2 = CPG_Dual.d2;
    CPG_Dual_cpp.d3 = CPG_Dual.d3;
    CPG_Dual_cpp.d4 = CPG_Dual.d4;
    CPG_Dual_cpp.d5 = CPG_Dual.d5;
    CPG_Dual_cpp.d6 = CPG_Dual.d6;
    CPG_Dual_cpp.d7 = CPG_Dual.d7;
    CPG_Dual_cpp.d8 = CPG_Dual.d8;
    CPG_Dual_cpp.d9 = CPG_Dual.d9;
    CPG_Dual_cpp.d10 = CPG_Dual.d10;
    CPG_Dual_cpp.d11 = CPG_Dual.d11;
    CPG_Dual_cpp.d12 = CPG_Dual.d12;
    for(i=0; i<12; i++) {
        CPG_Dual_cpp.d13[i] = CPG_Dual.d13[i];
    }
    CPG_Dual_cpp.d14 = CPG_Dual.d14;
    CPG_Dual_cpp.d15 = CPG_Dual.d15;
    CPG_Dual_cpp.d16 = CPG_Dual.d16;
    CPG_Dual_cpp.d17 = CPG_Dual.d17;
    CPG_Dual_cpp.d18 = CPG_Dual.d18;
    CPG_Dual_cpp.d19 = CPG_Dual.d19;
    CPG_Dual_cpp.d20 = CPG_Dual.d20;
    CPG_Dual_cpp.d21 = CPG_Dual.d21;
    CPG_Dual_cpp.d22 = CPG_Dual.d22;
    CPG_Dual_cpp.d23 = CPG_Dual.d23;
    CPG_Dual_cpp.d24 = CPG_Dual.d24;
    for(i=0; i<12; i++) {
        CPG_Dual_cpp.d25[i] = CPG_Dual.d25[i];
    }
    CPG_Dual_cpp.d26 = CPG_Dual.d26;
    CPG_Dual_cpp.d27 = CPG_Dual.d27;
    CPG_Dual_cpp.d28 = CPG_Dual.d28;
    CPG_Dual_cpp.d29 = CPG_Dual.d29;
    CPG_Dual_cpp.d30 = CPG_Dual.d30;
    CPG_Dual_cpp.d31 = CPG_Dual.d31;
    CPG_Dual_cpp.d32 = CPG_Dual.d32;
    CPG_Dual_cpp.d33 = CPG_Dual.d33;
    CPG_Dual_cpp.d34 = CPG_Dual.d34;
    CPG_Dual_cpp.d35 = CPG_Dual.d35;
    CPG_Dual_cpp.d36 = CPG_Dual.d36;
    for(i=0; i<12; i++) {
        CPG_Dual_cpp.d37[i] = CPG_Dual.d37[i];
    }
    CPG_Dual_cpp.d38 = CPG_Dual.d38;
    CPG_Dual_cpp.d39 = CPG_Dual.d39;
    CPG_Dual_cpp.d40 = CPG_Dual.d40;
    CPG_Dual_cpp.d41 = CPG_Dual.d41;
    CPG_Dual_cpp.d42 = CPG_Dual.d42;
    CPG_Dual_cpp.d43 = CPG_Dual.d43;
    CPG_Dual_cpp.d44 = CPG_Dual.d44;
    CPG_Dual_cpp.d45 = CPG_Dual.d45;
    CPG_Dual_cpp.d46 = CPG_Dual.d46;
    CPG_Dual_cpp.d47 = CPG_Dual.d47;
    CPG_Dual_cpp.d48 = CPG_Dual.d48;
    for(i=0; i<12; i++) {
        CPG_Dual_cpp.d49[i] = CPG_Dual.d49[i];
    }
    CPG_Dual_cpp.d50 = CPG_Dual.d50;
    CPG_Dual_cpp.d51 = CPG_Dual.d51;
    CPG_Dual_cpp.d52 = CPG_Dual.d52;
    CPG_Dual_cpp.d53 = CPG_Dual.d53;
    CPG_Dual_cpp.d54 = CPG_Dual.d54;
    CPG_Dual_cpp.d55 = CPG_Dual.d55;
    CPG_Dual_cpp.d56 = CPG_Dual.d56;
    CPG_Dual_cpp.d57 = CPG_Dual.d57;
    CPG_Dual_cpp.d58 = CPG_Dual.d58;
    CPG_Dual_cpp.d59 = CPG_Dual.d59;
    CPG_Dual_cpp.d60 = CPG_Dual.d60;
    for(i=0; i<12; i++) {
        CPG_Dual_cpp.d61[i] = CPG_Dual.d61[i];
    }
    CPG_Dual_cpp.d62 = CPG_Dual.d62;
    CPG_Dual_cpp.d63 = CPG_Dual.d63;
    CPG_Dual_cpp.d64 = CPG_Dual.d64;
    CPG_Dual_cpp.d65 = CPG_Dual.d65;
    CPG_Dual_cpp.d66 = CPG_Dual.d66;
    CPG_Dual_cpp.d67 = CPG_Dual.d67;
    CPG_Dual_cpp.d68 = CPG_Dual.d68;
    CPG_Dual_cpp.d69 = CPG_Dual.d69;
    CPG_Dual_cpp.d70 = CPG_Dual.d70;
    CPG_Dual_cpp.d71 = CPG_Dual.d71;
    CPG_Dual_cpp.d72 = CPG_Dual.d72;
    for(i=0; i<12; i++) {
        CPG_Dual_cpp.d73[i] = CPG_Dual.d73[i];
    }
    CPG_Dual_cpp.d74 = CPG_Dual.d74;
    CPG_Dual_cpp.d75 = CPG_Dual.d75;
    CPG_Dual_cpp.d76 = CPG_Dual.d76;
    CPG_Dual_cpp.d77 = CPG_Dual.d77;
    CPG_Dual_cpp.d78 = CPG_Dual.d78;
    CPG_Dual_cpp.d79 = CPG_Dual.d79;
    CPG_Dual_cpp.d80 = CPG_Dual.d80;
    CPG_Dual_cpp.d81 = CPG_Dual.d81;
    CPG_Dual_cpp.d82 = CPG_Dual.d82;
    CPG_Dual_cpp.d83 = CPG_Dual.d83;
    CPG_Dual_cpp.d84 = CPG_Dual.d84;
    CPG_Info_cpp_t CPG_Info_cpp {};
    CPG_Info_cpp.obj_val = CPG_Info.obj_val;
    CPG_Info_cpp.iter = CPG_Info.iter;
    CPG_Info_cpp.status = CPG_Info.status;
    CPG_Info_cpp.pri_res = CPG_Info.pri_res;
    CPG_Info_cpp.dua_res = CPG_Info.dua_res;
    CPG_Info_cpp.time = 1.0*(ASA_end-ASA_start) / CLOCKS_PER_SEC;
    CPG_Result_cpp_t CPG_Result_cpp {};
    CPG_Result_cpp.prim = CPG_Prim_cpp;
    CPG_Result_cpp.dual = CPG_Dual_cpp;
    CPG_Result_cpp.info = CPG_Info_cpp;
    return CPG_Result_cpp;

}

PYBIND11_MODULE(cpg_module, m) {

    py::class_<CPG_Params_cpp_t>(m, "cpg_params")
            .def(py::init<>())
            .def_readwrite("X_ref", &CPG_Params_cpp_t::X_ref)
            .def_readwrite("x_init", &CPG_Params_cpp_t::x_init)
            .def_readwrite("A_dyn", &CPG_Params_cpp_t::A_dyn)
            .def_readwrite("Inertial_matrix", &CPG_Params_cpp_t::Inertial_matrix)
            ;

    py::class_<CPG_Updated_cpp_t>(m, "cpg_updated")
            .def(py::init<>())
            .def_readwrite("X_ref", &CPG_Updated_cpp_t::X_ref)
            .def_readwrite("x_init", &CPG_Updated_cpp_t::x_init)
            .def_readwrite("A_dyn", &CPG_Updated_cpp_t::A_dyn)
            .def_readwrite("Inertial_matrix", &CPG_Updated_cpp_t::Inertial_matrix)
            ;

    py::class_<CPG_Prim_cpp_t>(m, "cpg_prim")
            .def(py::init<>())
            .def_readwrite("X", &CPG_Prim_cpp_t::X)
            .def_readwrite("U", &CPG_Prim_cpp_t::U)
            .def_readwrite("X_cmp", &CPG_Prim_cpp_t::X_cmp)
            ;

    py::class_<CPG_Dual_cpp_t>(m, "cpg_dual")
            .def(py::init<>())
            .def_readwrite("d0", &CPG_Dual_cpp_t::d0)
            .def_readwrite("d1", &CPG_Dual_cpp_t::d1)
            .def_readwrite("d2", &CPG_Dual_cpp_t::d2)
            .def_readwrite("d3", &CPG_Dual_cpp_t::d3)
            .def_readwrite("d4", &CPG_Dual_cpp_t::d4)
            .def_readwrite("d5", &CPG_Dual_cpp_t::d5)
            .def_readwrite("d6", &CPG_Dual_cpp_t::d6)
            .def_readwrite("d7", &CPG_Dual_cpp_t::d7)
            .def_readwrite("d8", &CPG_Dual_cpp_t::d8)
            .def_readwrite("d9", &CPG_Dual_cpp_t::d9)
            .def_readwrite("d10", &CPG_Dual_cpp_t::d10)
            .def_readwrite("d11", &CPG_Dual_cpp_t::d11)
            .def_readwrite("d12", &CPG_Dual_cpp_t::d12)
            .def_readwrite("d13", &CPG_Dual_cpp_t::d13)
            .def_readwrite("d14", &CPG_Dual_cpp_t::d14)
            .def_readwrite("d15", &CPG_Dual_cpp_t::d15)
            .def_readwrite("d16", &CPG_Dual_cpp_t::d16)
            .def_readwrite("d17", &CPG_Dual_cpp_t::d17)
            .def_readwrite("d18", &CPG_Dual_cpp_t::d18)
            .def_readwrite("d19", &CPG_Dual_cpp_t::d19)
            .def_readwrite("d20", &CPG_Dual_cpp_t::d20)
            .def_readwrite("d21", &CPG_Dual_cpp_t::d21)
            .def_readwrite("d22", &CPG_Dual_cpp_t::d22)
            .def_readwrite("d23", &CPG_Dual_cpp_t::d23)
            .def_readwrite("d24", &CPG_Dual_cpp_t::d24)
            .def_readwrite("d25", &CPG_Dual_cpp_t::d25)
            .def_readwrite("d26", &CPG_Dual_cpp_t::d26)
            .def_readwrite("d27", &CPG_Dual_cpp_t::d27)
            .def_readwrite("d28", &CPG_Dual_cpp_t::d28)
            .def_readwrite("d29", &CPG_Dual_cpp_t::d29)
            .def_readwrite("d30", &CPG_Dual_cpp_t::d30)
            .def_readwrite("d31", &CPG_Dual_cpp_t::d31)
            .def_readwrite("d32", &CPG_Dual_cpp_t::d32)
            .def_readwrite("d33", &CPG_Dual_cpp_t::d33)
            .def_readwrite("d34", &CPG_Dual_cpp_t::d34)
            .def_readwrite("d35", &CPG_Dual_cpp_t::d35)
            .def_readwrite("d36", &CPG_Dual_cpp_t::d36)
            .def_readwrite("d37", &CPG_Dual_cpp_t::d37)
            .def_readwrite("d38", &CPG_Dual_cpp_t::d38)
            .def_readwrite("d39", &CPG_Dual_cpp_t::d39)
            .def_readwrite("d40", &CPG_Dual_cpp_t::d40)
            .def_readwrite("d41", &CPG_Dual_cpp_t::d41)
            .def_readwrite("d42", &CPG_Dual_cpp_t::d42)
            .def_readwrite("d43", &CPG_Dual_cpp_t::d43)
            .def_readwrite("d44", &CPG_Dual_cpp_t::d44)
            .def_readwrite("d45", &CPG_Dual_cpp_t::d45)
            .def_readwrite("d46", &CPG_Dual_cpp_t::d46)
            .def_readwrite("d47", &CPG_Dual_cpp_t::d47)
            .def_readwrite("d48", &CPG_Dual_cpp_t::d48)
            .def_readwrite("d49", &CPG_Dual_cpp_t::d49)
            .def_readwrite("d50", &CPG_Dual_cpp_t::d50)
            .def_readwrite("d51", &CPG_Dual_cpp_t::d51)
            .def_readwrite("d52", &CPG_Dual_cpp_t::d52)
            .def_readwrite("d53", &CPG_Dual_cpp_t::d53)
            .def_readwrite("d54", &CPG_Dual_cpp_t::d54)
            .def_readwrite("d55", &CPG_Dual_cpp_t::d55)
            .def_readwrite("d56", &CPG_Dual_cpp_t::d56)
            .def_readwrite("d57", &CPG_Dual_cpp_t::d57)
            .def_readwrite("d58", &CPG_Dual_cpp_t::d58)
            .def_readwrite("d59", &CPG_Dual_cpp_t::d59)
            .def_readwrite("d60", &CPG_Dual_cpp_t::d60)
            .def_readwrite("d61", &CPG_Dual_cpp_t::d61)
            .def_readwrite("d62", &CPG_Dual_cpp_t::d62)
            .def_readwrite("d63", &CPG_Dual_cpp_t::d63)
            .def_readwrite("d64", &CPG_Dual_cpp_t::d64)
            .def_readwrite("d65", &CPG_Dual_cpp_t::d65)
            .def_readwrite("d66", &CPG_Dual_cpp_t::d66)
            .def_readwrite("d67", &CPG_Dual_cpp_t::d67)
            .def_readwrite("d68", &CPG_Dual_cpp_t::d68)
            .def_readwrite("d69", &CPG_Dual_cpp_t::d69)
            .def_readwrite("d70", &CPG_Dual_cpp_t::d70)
            .def_readwrite("d71", &CPG_Dual_cpp_t::d71)
            .def_readwrite("d72", &CPG_Dual_cpp_t::d72)
            .def_readwrite("d73", &CPG_Dual_cpp_t::d73)
            .def_readwrite("d74", &CPG_Dual_cpp_t::d74)
            .def_readwrite("d75", &CPG_Dual_cpp_t::d75)
            .def_readwrite("d76", &CPG_Dual_cpp_t::d76)
            .def_readwrite("d77", &CPG_Dual_cpp_t::d77)
            .def_readwrite("d78", &CPG_Dual_cpp_t::d78)
            .def_readwrite("d79", &CPG_Dual_cpp_t::d79)
            .def_readwrite("d80", &CPG_Dual_cpp_t::d80)
            .def_readwrite("d81", &CPG_Dual_cpp_t::d81)
            .def_readwrite("d82", &CPG_Dual_cpp_t::d82)
            .def_readwrite("d83", &CPG_Dual_cpp_t::d83)
            .def_readwrite("d84", &CPG_Dual_cpp_t::d84)
            ;

    py::class_<CPG_Info_cpp_t>(m, "cpg_info")
            .def(py::init<>())
            .def_readwrite("obj_val", &CPG_Info_cpp_t::obj_val)
            .def_readwrite("iter", &CPG_Info_cpp_t::iter)
            .def_readwrite("status", &CPG_Info_cpp_t::status)
            .def_readwrite("pri_res", &CPG_Info_cpp_t::pri_res)
            .def_readwrite("dua_res", &CPG_Info_cpp_t::dua_res)
            .def_readwrite("time", &CPG_Info_cpp_t::time)
            ;

    py::class_<CPG_Result_cpp_t>(m, "cpg_result")
            .def(py::init<>())
            .def_readwrite("cpg_prim", &CPG_Result_cpp_t::prim)
            .def_readwrite("cpg_dual", &CPG_Result_cpp_t::dual)
            .def_readwrite("cpg_info", &CPG_Result_cpp_t::info)
            ;

    m.def("solve", &solve_cpp);

    m.def("set_solver_default_settings", &cpg_set_solver_default_settings);
    m.def("set_solver_feastol", &cpg_set_solver_feastol);
    m.def("set_solver_abstol", &cpg_set_solver_abstol);
    m.def("set_solver_reltol", &cpg_set_solver_reltol);
    m.def("set_solver_feastol_inacc", &cpg_set_solver_feastol_inacc);
    m.def("set_solver_abstol_inacc", &cpg_set_solver_abstol_inacc);
    m.def("set_solver_reltol_inacc", &cpg_set_solver_reltol_inacc);
    m.def("set_solver_maxit", &cpg_set_solver_maxit);

}
