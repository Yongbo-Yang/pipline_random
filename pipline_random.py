#-*- coding:utf-8 -*-
'''小李师兄的多跨方法'''
import numpy as np
from numpy import pi, exp, sqrt, linalg


def support_distance(sup_local):
    '''确定支座相对距离'''
    n = len(sup_local)
    l_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            l_mat[i, j] = abs(sup_local[i] - sup_local[j])
    return l_mat


def clough_power_spectrum(omega):
    '''
    生成地震功率谱
    Kanai-Tajimi谱
    '''
    nf = len(omega)
    omega_g1 = 5 * pi
    omega_g2 = 0.1 * omega_g1
    zeta_g1 = 0.6
    zeta_g2 = 0.6
    s0 = 0.025
    spectrum = np.zeros((1, nf))
    for i in range(nf):
        spectrum[0, i] = s0 * (1 + 4 * zeta_g1 ** 2 * (omega[i] / omega_g1) ** 2) / (
                    (1 - (omega[i] / omega_g1) ** 2) ** 2 + 4 * zeta_g1 ** 2 * (omega[i] / omega_g1) ** 2) * (
                                  omega[i] / omega_g2) ** 4 / (
                                  (1 - (omega[i] / omega_g2) ** 2) ** 2 + 4 * zeta_g2 ** 2 * (
                                      omega[i] / omega_g2) ** 2)
    return spectrum


def spatial_correlation_cofficient(gamma, lmat, omega, vs):
    '''空间相关函数'''
    nf = len(omega)
    a,b = lmat.shape
    rho = np.empty((a, b, nf))
    for i in range(nf):
        rho[:, :, i] = exp(-(gamma * lmat * omega[i] / 2 / pi / vs))
    return rho


def wave_passage_effect(sup_local, omega, vs):
    '''行波效应影响矩阵'''
    mat_b = exp(-1j * omega * sup_local / vs)
    mat_b = np.diag(mat_b)
    return mat_b


def e_y_theta_m(r0, sup_local, e0, iner):
    '''根据特解确定边界条件的形式'''
    e_y = exp(r0 * sup_local)
    e_theta = r0 @ exp(r0 * sup_local)
    e_m = e0 * iner * r0 ** 2 @ exp(r0 * sup_local)
    return e_y, e_theta, e_m


def first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc, loca, style):
    '''求响应对随机参数的一阶导数'''
    if style == 'Y':
        d_e_x_dm = exp(r0 * loca) * loca @ d_r_dm
        d_e_x_de = exp(r0 * loca) * loca @ d_r_de
        d_e_x_dc = exp(r0 * loca) * loca @ d_r_dc
    elif style == 'Theta':
        d_e_x_dm = (1 + loca * r0) @ exp(r0 * loca) @ d_r_dm
        d_e_x_de = (1 + loca * r0) @ exp(r0 * loca) @ d_r_de
        d_e_x_dc = (1 + loca * r0) @ exp(r0 * loca) @ d_r_dc
    elif style == 'M':
        d_e_x_dm = e0 * iner * r0 @ exp(loca * r0) @ (2 + loca @ r0) @ d_r_dm
        d_e_x_de = e0 * iner * r0 @ exp(loca * r0) @ (2 + loca @ r0) @ d_r_de + iner * r0 ** 2 @ exp(loca * r0)
        d_e_x_dc = e0 * iner * r0 @ exp(loca * r0) @ (2 + loca @ r0) @ d_r_dc
    dx_d1 = {}
    dx_d1['d_e_x_dm'] = d_e_x_dm
    dx_d1['d_e_x_de'] = d_e_x_de
    dx_d1['d_e_x_dc'] = d_e_x_dc
    return dx_d1


def second_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc, d_2r_dmdm, d_2r_dmde, d_2r_dmdc, d_2r_dede, d_2r_dcde, d_2r_dcdc, loca, style):
    '''求响应关于随机参数的二阶导数'''
    if style == 'Y':
        d_2e_x_dmdm = exp(loca * r0) * loca @ (loca * d_r_dm ** 2 + d_2r_dmdm)
        d_2e_x_dmdc = exp(loca * r0) * loca @ (loca * d_r_dm @ d_r_dc + d_2r_dmdc)
        d_2e_x_dmde = exp(loca * r0) * loca @ (loca * d_r_dm @ d_r_de + d_2r_dmde)
        d_2e_x_dcdc = exp(loca * r0) * loca @ (loca * d_r_dc ** 2 + d_2r_dcdc)
        d_2e_x_dedc = exp(loca * r0) * loca @ (loca * d_r_dc @ d_r_de + d_2r_dcde)
        d_2e_x_dede = exp(loca * r0) * loca @ (loca * d_r_de ** 2 + d_2r_dede)
    elif style == 'Theta':
        d_2e_x_dmdm = ((2 + loca * r0) * loca @ d_r_dm ** 2 + (1 + loca * r0) @ d_2r_dmdm) @ exp(loca * r0)
        d_2e_x_dmdc = ((2 + loca * r0) * loca @ d_r_dm @ d_r_dc + (1 + loca * r0) @ d_2r_dmdc) @ exp(loca * r0)
        d_2e_x_dmde = ((2 + loca * r0) * loca @ d_r_dm @ d_r_de + (1 + loca * r0) @ d_2r_dmde) @ exp(loca * r0)
        d_2e_x_dcdc = ((2 + loca * r0) * loca @ d_r_dc ** 2 + (1 + loca * r0) @ d_2r_dcdc) @ exp(loca * r0)
        d_2e_x_dedc = ((2 + loca * r0) * loca @ d_r_dc @ d_r_de + (1 + loca * r0) @ d_2r_dcde) @ exp(loca * r0)
        d_2e_x_dede = ((2 + loca * r0) * loca @ d_r_de ** 2 + (1 + loca * r0) @ d_2r_dede) @ exp(loca * r0)
    elif style == 'M':
        d_2e_x_dmdm = e0*iner*((2+4*loca*r0 + loca**2*r0**2)@d_r_dm**2+(2*r0+loca*r0**2)@d_2r_dmdm)@exp(loca*r0)
        d_2e_x_dmdc = e0*iner*((2+4*loca*r0 + loca**2*r0**2)@d_r_dm@d_r_dc+(2*r0+loca*r0**2)@d_2r_dmdc)@exp(loca*r0)
        d_2e_x_dmde = e0*iner*((2+4*loca*r0 + loca**2*r0**2)@d_r_dm@d_r_de+(2*r0+loca*r0**2)@d_2r_dmde)@exp(loca*r0)
        +iner*r0@exp(loca*r0)@(2+loca*r0)@d_r_dm
        d_2e_x_dcdc = e0*iner*((2+4*loca*r0 + loca**2*r0**2)@d_r_dc**2+(2*r0+loca*r0**2)@d_2r_dcdc)@exp(loca*r0)
        d_2e_x_dedc = e0*iner*((2+4*loca*r0 + loca**2*r0**2)@d_r_dc@d_r_de+(2*r0+loca*r0**2)@d_2r_dcde)@exp(loca*r0)
        +iner*r0@exp(loca*r0)@(2+loca*r0)@d_r_dc
        d_2e_x_dede = e0*iner*((2+4*loca*r0 + loca**2*r0**2)@d_r_de**2+(2*r0+loca*r0**2)@d_2r_dede)@exp(loca*r0)
    return d_2e_x_dmdm, d_2e_x_dmdc, d_2e_x_dmde, d_2e_x_dcdc, d_2e_x_dedc, d_2e_x_dede

n_sup = 5  # 支座个数
sup_local = np.array([0, 40, 90, 140, 180])  # 支座位置
# 结构随机参数
b = 1
h = 1  # 截面性质
iner = b * h ** 3 / 12
e0 = 2e11
m0 = b * h * 7800
c0 = 500
# 变量标准差
sigma_alpha_e = 0.01
sigma_alpha_c = 0
sigma_alpha_m = 0
cov_e = (e0 * sigma_alpha_e) ** 2
cov_m = (e0 * sigma_alpha_m) ** 2
cov_c = (e0 * sigma_alpha_c) ** 2
#空间多点效应表征
gamma = 0.1  # 非相干因子
vs = 2000  # 地震波视波速
lmat = support_distance(sup_local)  # 生成支座相对距离
# 随机地震载荷
df = 0.1  # 频率步长
f = np.arange(df, 15 + df, df)  # 频点数组
nf = len(f)  # 频点数
omega = 2 * pi * f  # 转换为圆频率
dx = 1  # 座标尺度
x1 = np.arange(0, sup_local[1] + dx, dx)  # 第一跨座标
n_x1 = len(x1)  # 坐标个数
x2 = np.arange(sup_local[1] + dx, sup_local[2] + dx, dx)  # 第二跨座标
n_x2 = len(x2)  # 坐标个数
x3 = np.arange(sup_local[2] + dx, sup_local[3] + dx, dx)  # 第三跨座标
n_x3 = len(x3)  # 坐标个数
x4 = np.arange(sup_local[3] + dx, sup_local[4] + dx, dx)  # 第四跨座标
n_x4 = len(x4)  # 坐标个数
n_x = n_x1 + n_x2 + n_x3 + n_x4  # 总做标书
x = np.hstack((x1, x2, x3, x4))  # 总座标数组
# 初始化数据
d_y_dm_x1 = np.zeros((n_x1, n_sup))
d_y_dc_x1 = np.zeros((n_x1, n_sup))
d_y_de_x1 = np.zeros((n_x1, n_sup))
d_2y_dmdm_x1 = np.zeros((n_x1, n_sup))
d_2y_dmdc_x1 = np.zeros((n_x1, n_sup))
d_2y_dmde_x1 = np.zeros((n_x1, n_sup))
d_2y_dcdc_x1 = np.zeros((n_x1, n_sup))
d_2y_dcde_x1 = np.zeros((n_x1, n_sup))
d_2y_dede_x1 = np.zeros((n_x1, n_sup))
y0_x1 = np.zeros((n_x1, n_sup))

d_y_dm_x2 = np.zeros((n_x2, n_sup))
d_y_dc_x2 = np.zeros((n_x2, n_sup))
d_y_de_x2 = np.zeros((n_x2, n_sup))
d_2y_dmdm_x2 = np.zeros((n_x2, n_sup))
d_2y_dmdc_x2 = np.zeros((n_x2, n_sup))
d_2y_dmde_x2 = np.zeros((n_x2, n_sup))
d_2y_dcdc_x2 = np.zeros((n_x2, n_sup))
d_2y_dcde_x2 = np.zeros((n_x2, n_sup))
d_2y_dede_x2 = np.zeros((n_x2, n_sup))
y0_x2 = np.zeros((n_x2, n_sup))
# 生成地震功率谱
spectrum = clough_power_spectrum(omega)  # 生成功率谱
rho = spatial_correlation_cofficient(gamma, lmat, omega, vs)  # 生成相干函数
for ff in range(nf):
    mat_b = wave_passage_effect(sup_local, omega[ff], vs)  # 行波效应矩阵
    cholskey = linalg.cholesky(rho[:, :, ff])  # 对相干矩阵进行cholesky分解
    a_psudo = sqrt(spectrum[0, ff]) * mat_b @ np.eye(n_sup) @ cholskey  # 生成加速度功率谱
    a0_psudo = a_psudo / omega[ff] ** 2
    for gg in range(n_sup):
        a1_psudo = a0_psudo[0, gg]
        a2_psudo = a0_psudo[1, gg]
        a3_psudo = a0_psudo[2, gg]
        a4_psudo = a0_psudo[3, gg]
        a5_psudo = a0_psudo[4, gg]


    # 求解
    s0 = ((omega[ff] ** 2 * m0 - 1j * omega[ff] * c0) / e0 / iner) ** (1/4)  # 特征值
    s_r = np.array([1, -1, 1j, -1j])
    r0 = s_r * s0  # 特征值数组
    # 各段边界条件表达形式
    e_y_sup1, e_theta_sup1, e_m_sup1 = e_y_theta_m(r0, sup_local[0], e0, iner)
    e_y_sup2, e_theta_sup2, e_m_sup2 = e_y_theta_m(r0, sup_local[1], e0, iner)
    e_y_sup3, e_theta_sup3, e_m_sup3 = e_y_theta_m(r0, sup_local[2], e0, iner)
    e_y_sup4, e_theta_sup4, e_m_sup4 = e_y_theta_m(r0, sup_local[3], e0, iner)
    e_y_sup5, e_theta_sup5, e_m_sup5 = e_y_theta_m(r0, sup_local[4], e0, iner)
    # 特征值数组关于随机变量e、c、m的一阶导数
    d_r_dm = omega[ff] ** 2 * (e0 * iner) ** (-1/4) / 4 * (omega[ff] ** 2 * m0 - 1j * omega[ff] * c0) ** (-3/4) * s_r
    d_r_dc = -1j * omega[ff] * (e0 * iner) ** (-1/4) / 4 * (omega[ff] ** 2 * m0 - 1j * omega[ff] * c0) ** (-3/4) * s_r
    d_r_de = -e0 ** (-5/4) * iner ** (-1/4) / 4 * (omega[ff] ** 2 * m0 - 1j * omega[ff] * c0) ** (1/4) * s_r
    # 特征值数组关于随机变量e、c、m的二阶导数
    d_2r_dmdm = -3 * omega[ff] ** 4 * (e0 * iner) ** (-1 / 4) / 16 * (omega[ff] ** 2 * m0 - 1j * omega[ff] * c0) ** (
            -7 / 4) * s_r
    d_2r_dmdc = 3 * 1j * omega[ff] ** 3 * (e0 * iner) ** (-1 / 4) / 16 * (
            omega[ff] ** 2 * m0 - 1j * omega[ff] * c0) ** (-7 / 4) * s_r
    d_2r_dmde = -e0 ** (-5 / 4) * iner ** (-1 / 4) * omega[ff] ** 2 / 16 * (
                omega[ff] ** 2 * m0 - 1j * omega[ff] * c0) ** (-3 / 4) * s_r
    d_2r_dcdc = 3 * omega[ff] ** 2 * (e0 * iner) ** (-1 / 4) / 16 * (omega[ff] ** 2 * m0 - 1j * omega[ff] * c0) ** (
                -7 / 4) * s_r
    d_2r_dcde = e0 ** (-5 / 4) * iner ** (-1 / 4) * (1j * omega[ff]) / 4 * (
                omega[ff] ** 2 * m0 - 1j * omega[ff] * c0) ** (
                        -3 / 4) * s_r
    d_2r_dede = 5 * e0 ** (-9 / 4) * iner ** (-1 / 4) / 16 * (omega[ff] ** 2 * m0 - 1j * omega[ff] * c0) ** (
                1 / 4) * s_r
    # e_x的一阶偏导
    d_e_y_sup1, d_e_y_sup1, d_e_y_sup1 = first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc,
                                                                      sup_local[0], 'Y')
    d_e_y_dm_sup2, d_e_y_de_sup2, d_e_y_de_sup2 = first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc,
                                                                      sup_local[1], 'Y')
    d_e_y_dm_sup3, d_e_y_de_sup3, d_e_y_de_sup3 = first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc,
                                                                      sup_local[2], 'Y')
    d_e_y_dm_sup4, d_e_y_de_sup4, d_e_y_de_sup4 = first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc,
                                                                      sup_local[3], 'Y')
    d_e_y_dm_sup5, d_e_y_de_sup5, d_e_y_de_sup5 = first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc,
                                                                      sup_local[4], 'Y')

    d_e_theta_dm_sup2, d_e_theta_de_sup2, d_e_theta_de_sup2 = first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc,
                                                                                  sup_local[1], 'theta')
    d_e_theta_dm_sup3, d_e_theta_de_sup3, d_e_theta_de_sup3 = first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc,
                                                                                  sup_local[2], 'theta')
    d_e_theta_dm_sup4, d_e_theta_de_sup4, d_e_theta_de_sup4 = first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc,
                                                                                  sup_local[3], 'theta')

    d_e_m_dm_sup1, d_e_m_de_sup1, d_e_m_de_sup1 = first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc,
                                                                      sup_local[0], 'M')
    d_e_m_dm_sup2, d_e_m_de_sup2, d_e_m_de_sup2 = first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc,
                                                                      sup_local[2], 'M')
    d_e_m_dm_sup3, d_e_m_de_sup3, d_e_m_de_sup3 = first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc,
                                                                      sup_local[3], 'M')
    d_e_m_dm_sup4, d_e_m_de_sup4, d_e_m_de_sup4 = first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc,
                                                                      sup_local[4], 'M')
    d_e_m_dm_sup5, d_e_m_de_sup5, d_e_m_de_sup5 = first_order_partial(e0, iner, r0, d_r_dm, d_r_de, d_r_dc,
                                                                      sup_local[5], 'M')
    # e_x的二阶偏导
    d_2e_x_dmdm, d_2e_x_dmdc, d_2e_x_dmde, d_2e_x_dcdc, d_2e_x_dedc, d_2e_x_dede = second_order_partial(e0, iner, r0,
         d_r_dm, d_r_de, d_r_dc, d_2r_dmdm, d_2r_dmde, d_2r_dmdc, d_2r_dede, d_2r_dcde, d_2r_dcdc, sup_local[0], 'Y')

    print(1)