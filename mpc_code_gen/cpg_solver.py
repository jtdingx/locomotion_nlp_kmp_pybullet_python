
"""
Auto-generated by CVXPYgen on November 30, 2023 at 11:36:07.
Content: Custom solve method for CVXPY interface.
"""

import time
import warnings
import numpy as np
from cvxpy.reductions import Solution
from cvxpy.problems.problem import SolverStats
from mpc_code_gen import cpg_module


def cpg_solve(prob, updated_params=None, **kwargs):

    # set flags for updated parameters
    upd = cpg_module.cpg_updated()
    if updated_params is None:
        updated_params = ["X_ref", "L_ref", "x_init", "A_dyn", "Z_c", "J_ini"]
    for p in updated_params:
        try:
            setattr(upd, p, True)
        except AttributeError:
            raise(AttributeError("%s is not a parameter." % p))

    # set solver settings
    cpg_module.set_solver_default_settings()
    for key, value in kwargs.items():
        try:
            eval('cpg_module.set_solver_%s(value)' % key)
        except AttributeError:
            raise(AttributeError('Solver setting "%s" not available.' % key))

    # set parameter values
    par = cpg_module.cpg_params()
    par.X_ref = list(prob.param_dict['X_ref'].value.flatten(order='F'))
    par.L_ref = list(prob.param_dict['L_ref'].value.flatten(order='F'))
    par.x_init = list(prob.param_dict['x_init'].value.flatten(order='F'))
    par.A_dyn = list(prob.param_dict['A_dyn'].value.flatten(order='F'))
    par.Z_c = prob.param_dict['Z_c'].value
    par.J_ini = list(prob.param_dict['J_ini'].value.flatten(order='F'))

    # solve
    t0 = time.time()
    res = cpg_module.solve(upd, par)
    t1 = time.time()

    # store solution in problem object
    prob._clear_solution()
    prob.var_dict['X'].value = np.array(res.cpg_prim.X).reshape((12, 8), order='F')
    prob.var_dict['U'].value = np.array(res.cpg_prim.U).reshape((6, 7), order='F')
    prob.var_dict['X_cmp'].value = np.array(res.cpg_prim.X_cmp).reshape((2, 8), order='F')
    prob.constraints[0].save_dual_value(np.array(res.cpg_dual.d0).reshape(12))
    prob.constraints[1].save_dual_value(np.array(res.cpg_dual.d1).reshape(12))
    prob.constraints[2].save_dual_value(np.array(res.cpg_dual.d2).reshape(1))
    prob.constraints[3].save_dual_value(np.array(res.cpg_dual.d3).reshape(1))
    prob.constraints[4].save_dual_value(np.array(res.cpg_dual.d4))
    prob.constraints[5].save_dual_value(np.array(res.cpg_dual.d5))
    prob.constraints[6].save_dual_value(np.array(res.cpg_dual.d6))
    prob.constraints[7].save_dual_value(np.array(res.cpg_dual.d7))
    prob.constraints[8].save_dual_value(np.array(res.cpg_dual.d8))
    prob.constraints[9].save_dual_value(np.array(res.cpg_dual.d9))
    prob.constraints[10].save_dual_value(np.array(res.cpg_dual.d10))
    prob.constraints[11].save_dual_value(np.array(res.cpg_dual.d11))
    prob.constraints[12].save_dual_value(np.array(res.cpg_dual.d12))
    prob.constraints[13].save_dual_value(np.array(res.cpg_dual.d13).reshape(12))
    prob.constraints[14].save_dual_value(np.array(res.cpg_dual.d14).reshape(1))
    prob.constraints[15].save_dual_value(np.array(res.cpg_dual.d15).reshape(1))
    prob.constraints[16].save_dual_value(np.array(res.cpg_dual.d16))
    prob.constraints[17].save_dual_value(np.array(res.cpg_dual.d17))
    prob.constraints[18].save_dual_value(np.array(res.cpg_dual.d18))
    prob.constraints[19].save_dual_value(np.array(res.cpg_dual.d19))
    prob.constraints[20].save_dual_value(np.array(res.cpg_dual.d20))
    prob.constraints[21].save_dual_value(np.array(res.cpg_dual.d21))
    prob.constraints[22].save_dual_value(np.array(res.cpg_dual.d22))
    prob.constraints[23].save_dual_value(np.array(res.cpg_dual.d23))
    prob.constraints[24].save_dual_value(np.array(res.cpg_dual.d24))
    prob.constraints[25].save_dual_value(np.array(res.cpg_dual.d25).reshape(12))
    prob.constraints[26].save_dual_value(np.array(res.cpg_dual.d26).reshape(1))
    prob.constraints[27].save_dual_value(np.array(res.cpg_dual.d27).reshape(1))
    prob.constraints[28].save_dual_value(np.array(res.cpg_dual.d28))
    prob.constraints[29].save_dual_value(np.array(res.cpg_dual.d29))
    prob.constraints[30].save_dual_value(np.array(res.cpg_dual.d30))
    prob.constraints[31].save_dual_value(np.array(res.cpg_dual.d31))
    prob.constraints[32].save_dual_value(np.array(res.cpg_dual.d32))
    prob.constraints[33].save_dual_value(np.array(res.cpg_dual.d33))
    prob.constraints[34].save_dual_value(np.array(res.cpg_dual.d34))
    prob.constraints[35].save_dual_value(np.array(res.cpg_dual.d35))
    prob.constraints[36].save_dual_value(np.array(res.cpg_dual.d36))
    prob.constraints[37].save_dual_value(np.array(res.cpg_dual.d37).reshape(12))
    prob.constraints[38].save_dual_value(np.array(res.cpg_dual.d38).reshape(1))
    prob.constraints[39].save_dual_value(np.array(res.cpg_dual.d39).reshape(1))
    prob.constraints[40].save_dual_value(np.array(res.cpg_dual.d40))
    prob.constraints[41].save_dual_value(np.array(res.cpg_dual.d41))
    prob.constraints[42].save_dual_value(np.array(res.cpg_dual.d42))
    prob.constraints[43].save_dual_value(np.array(res.cpg_dual.d43))
    prob.constraints[44].save_dual_value(np.array(res.cpg_dual.d44))
    prob.constraints[45].save_dual_value(np.array(res.cpg_dual.d45))
    prob.constraints[46].save_dual_value(np.array(res.cpg_dual.d46))
    prob.constraints[47].save_dual_value(np.array(res.cpg_dual.d47))
    prob.constraints[48].save_dual_value(np.array(res.cpg_dual.d48))
    prob.constraints[49].save_dual_value(np.array(res.cpg_dual.d49).reshape(12))
    prob.constraints[50].save_dual_value(np.array(res.cpg_dual.d50).reshape(1))
    prob.constraints[51].save_dual_value(np.array(res.cpg_dual.d51).reshape(1))
    prob.constraints[52].save_dual_value(np.array(res.cpg_dual.d52))
    prob.constraints[53].save_dual_value(np.array(res.cpg_dual.d53))
    prob.constraints[54].save_dual_value(np.array(res.cpg_dual.d54))
    prob.constraints[55].save_dual_value(np.array(res.cpg_dual.d55))
    prob.constraints[56].save_dual_value(np.array(res.cpg_dual.d56))
    prob.constraints[57].save_dual_value(np.array(res.cpg_dual.d57))
    prob.constraints[58].save_dual_value(np.array(res.cpg_dual.d58))
    prob.constraints[59].save_dual_value(np.array(res.cpg_dual.d59))
    prob.constraints[60].save_dual_value(np.array(res.cpg_dual.d60))
    prob.constraints[61].save_dual_value(np.array(res.cpg_dual.d61).reshape(12))
    prob.constraints[62].save_dual_value(np.array(res.cpg_dual.d62).reshape(1))
    prob.constraints[63].save_dual_value(np.array(res.cpg_dual.d63).reshape(1))
    prob.constraints[64].save_dual_value(np.array(res.cpg_dual.d64))
    prob.constraints[65].save_dual_value(np.array(res.cpg_dual.d65))
    prob.constraints[66].save_dual_value(np.array(res.cpg_dual.d66))
    prob.constraints[67].save_dual_value(np.array(res.cpg_dual.d67))
    prob.constraints[68].save_dual_value(np.array(res.cpg_dual.d68))
    prob.constraints[69].save_dual_value(np.array(res.cpg_dual.d69))
    prob.constraints[70].save_dual_value(np.array(res.cpg_dual.d70))
    prob.constraints[71].save_dual_value(np.array(res.cpg_dual.d71))
    prob.constraints[72].save_dual_value(np.array(res.cpg_dual.d72))
    prob.constraints[73].save_dual_value(np.array(res.cpg_dual.d73).reshape(12))
    prob.constraints[74].save_dual_value(np.array(res.cpg_dual.d74).reshape(1))
    prob.constraints[75].save_dual_value(np.array(res.cpg_dual.d75).reshape(1))
    prob.constraints[76].save_dual_value(np.array(res.cpg_dual.d76))
    prob.constraints[77].save_dual_value(np.array(res.cpg_dual.d77))
    prob.constraints[78].save_dual_value(np.array(res.cpg_dual.d78))
    prob.constraints[79].save_dual_value(np.array(res.cpg_dual.d79))
    prob.constraints[80].save_dual_value(np.array(res.cpg_dual.d80))
    prob.constraints[81].save_dual_value(np.array(res.cpg_dual.d81))
    prob.constraints[82].save_dual_value(np.array(res.cpg_dual.d82))
    prob.constraints[83].save_dual_value(np.array(res.cpg_dual.d83))
    prob.constraints[84].save_dual_value(np.array(res.cpg_dual.d84))

    # store additional solver information in problem object
    prob._status = res.cpg_info.status
    if abs(res.cpg_info.obj_val) == 1e30:
        prob._value = np.sign(res.cpg_info.obj_val)*np.inf
    else:
        prob._value = res.cpg_info.obj_val
    primal_vars = {var.id: var.value for var in prob.variables()}
    dual_vars = {c.id: c.dual_value for c in prob.constraints}
    solver_specific_stats = {'obj_val': res.cpg_info.obj_val,
                             'status': prob._status,
                             'iter': res.cpg_info.iter,
                             'pri_res': res.cpg_info.pri_res,
                             'dua_res': res.cpg_info.dua_res,
                             'time': res.cpg_info.time}
    attr = {'solve_time': t1-t0, 'solver_specific_stats': solver_specific_stats, 'num_iters': res.cpg_info.iter}
    prob._solution = Solution(prob.status, prob.value, primal_vars, dual_vars, attr)
    results_dict = {'solver_specific_stats': solver_specific_stats,
                    'num_iters': res.cpg_info.iter,
                    'solve_time': t1-t0}
    prob._solver_stats = SolverStats(results_dict, 'OSQP')

    return prob.value
