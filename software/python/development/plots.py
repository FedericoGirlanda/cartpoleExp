import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
import numpy as np

trajDirtran_path = "results/trajExpDirtran_dist.csv"
trajOpt_path = "results/trajExpRtc_dist.csv"

# load trajectories from csv file
trajDirtran = np.loadtxt(trajDirtran_path, skiprows=1, delimiter=",")
dirtran_time_list = trajDirtran.T[0].T  
dirtran_cart_pos_list = trajDirtran.T[1].T  
dirtran_pend_pos_list = trajDirtran.T[2].T  
dirtran_cart_vel_list = trajDirtran.T[3].T  
dirtran_pend_vel_list = trajDirtran.T[4].T  
dirtran_force_list = trajDirtran.T[5].T  

trajOpt = np.loadtxt(trajOpt_path, skiprows=1, delimiter=",")
rtc_time_list = trajOpt.T[0].T  
rtc_cart_pos_list = trajOpt.T[1].T  
rtc_pend_pos_list = trajOpt.T[2].T  
rtc_cart_vel_list = trajOpt.T[3].T  
rtc_pend_vel_list = trajOpt.T[4].T  
rtc_force_list = trajOpt.T[5].T 

 # Plot the results
fig_test, ax_test = plt.subplots(2,2, figsize = (8, 8))
fig_test.suptitle(f"Real System's trajectory stabilization: optimized(blue) vs dirtran(orange)")
ax_test[0][0].plot(rtc_time_list, rtc_cart_pos_list)
ax_test[0][1].plot(rtc_time_list, rtc_pend_pos_list)
ax_test[1][0].plot(rtc_time_list, rtc_cart_vel_list)
ax_test[1][1].plot(rtc_time_list, rtc_pend_vel_list)
ax_test[0][0].plot(dirtran_time_list, dirtran_cart_pos_list, linestyle = "--")
ax_test[0][1].plot(dirtran_time_list, dirtran_pend_pos_list, linestyle = "--")
ax_test[1][0].plot(dirtran_time_list, dirtran_cart_vel_list, linestyle = "--")
ax_test[1][1].plot(dirtran_time_list, dirtran_pend_vel_list, linestyle = "--")
ax_test[0][0].hlines(np.vstack((np.ones((len(dirtran_time_list),1)),-np.ones((len(dirtran_time_list),1))))*0.3,dirtran_time_list[0], dirtran_time_list[-1])
ax_test[0][0].vlines(np.array([2.51,2.52,2.53,2.54]),-0.25,0.6)
ax_test[1][0].vlines(np.array([2.51,2.52,2.53,2.54]),-1,1)
ax_test[0][1].vlines(np.array([2.51,2.52,2.53,2.54]),-5,6)
ax_test[1][1].vlines(np.array([2.51,2.52,2.53,2.54]),-15,15)
ax_test[0][0].set_xlabel("x0(x_cart)")
ax_test[0][1].set_xlabel("x1(theta)")
ax_test[1][0].set_xlabel("x2(x_cart_dot)")
ax_test[1][1].set_xlabel("x3(theta_dot)")

fig_test, ax_test = plt.subplots(1,1, figsize = (8, 8))
ax_test.plot(rtc_time_list,rtc_force_list)
ax_test.plot(dirtran_time_list,dirtran_force_list, linestyle = "--")
ax_test.hlines(np.vstack((np.ones((len(dirtran_time_list),1)),-np.ones((len(dirtran_time_list),1))))*6,dirtran_time_list[0], dirtran_time_list[-1])
ax_test.vlines(np.array([2.51,2.52,2.53,2.54]),-5.5,5.5)
ax_test.set_xlabel("u(force)")
plt.show()