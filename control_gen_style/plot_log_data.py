import numpy as np
import re
import matplotlib.pyplot as plt

def plot_log_data(log_file_name):

    style_loss_list = []
    style_accu_list = []
    pt_loss_list = []
    recon_loss_list = []
    kld_list = []
    ind_loss_list = []
    style_recon_loss_list = []

    with open(log_file_name, "r") as log_file:

        style_loss_re = r'style_loss: (\d+\.?\d+|nan)'
        style_accu_re = r'style_accu: (\d+\.?\d+|nan)'
        pt_loss_re = r'pt_loss: (\d+\.?\d+|nan)'
        recon_loss_re = r' recon_loss: (\d+\.?\d+|nan)'
        kld_re = r'kld: (\d+\.?\d+|nan)'
        ind_loss_re = r'ind_loss: (\d+\.?\d+|nan)'
        style_recon_loss_re = r'style_recon_loss: (\d+\.?\d+|nan)'

        i = 0
        for line in log_file:

        	print(i)
        	i += 1

        	if len(style_loss_list) > 400:
        		break

	        style_loss = re.findall(style_loss_re, line)
	        if not len(style_loss) > 0: # Continue for the lines which don't have the losses printed on them
	        	continue

	        style_loss = style_loss[0]
	        style_accu = re.findall(style_accu_re, line)[0]
	        pt_loss = re.findall(pt_loss_re, line)[0]
	        recon_loss = re.findall(recon_loss_re, line)[0]
	        kld = re.findall(kld_re, line)[0]
	        ind_loss = re.findall(ind_loss_re, line)[0]
	        style_recon_loss = re.findall(style_recon_loss_re, line)[0]

	        print("recons: ", recon_loss)

	        style_loss_list.append(style_loss)
	        style_accu_list.append(style_accu)
	        pt_loss_list.append(pt_loss)
	        recon_loss_list.append(recon_loss)
	        kld_list.append(kld)
	        ind_loss_list.append(ind_loss)
	        style_recon_loss_list.append(style_recon_loss)

	         


    return style_loss_list, style_accu_list, pt_loss_list, recon_loss_list, kld_list, ind_loss_list, style_recon_loss_list
	        


ex_line = "INFO:root:iter: 1, style_loss: nan, style_accu: 0.4941 pt_loss: nan, recon_loss: 0.5111, kld: nan, ind_loss: 0.4542, style_recon_loss: 0.3292, temp_o: 1.0000"
# style_loss_re = ".* style_loss: \d+"
# style_loss = re.findall(style_loss_re, line)
# style_loss = ex_line.split("style_loss: ", 1)
# print("style loss: ", style_loss)

n01 = "results/normal_0_1_45578047-85c4-46a2-acbd-c4cb95021a8e.ctrl_style/log.txt"
n02 = "results/normal_0_2_2e735956-6b65-4e03-9c42-2d6c2c0c1f55.ctrl_style/log.txt"
b12 = "results/beta_1_2_6ea98694-8d14-4049-aca0-20cb9f149564.ctrl_style/log.txt"
e12 = "results/exp_1_2_8dca0bb8-cf87-4f79-b2e2-1b14b678cd4d.ctrl_style/log.txt"

n01_results = plot_log_data(n01)
n02_results = plot_log_data(n02)
b12_results = plot_log_data(b12)
e12_results = plot_log_data(e12)

results = {
	"Norma(0, 1)": n01_results,
	"Normal(0, 2)": n02_results,
	"Beta(1, 2)": b12_results,
	"Exp(1, 2)": e12_results
}

print("Finished getting data")
# print(results[3])
style_loss_ind = 0
style_accu_ind = 1
pt_loss_ind = 2
recon_loss_ind = 3
kld_ind = 4
ind_loss_ind = 5
style_recon_loss_ind = 6

num_rows = len(n01_results[0])
x_vals = np.arange(num_rows)

print(len(n01_results[recon_loss_ind]))
print(len(e12_results[recon_loss_ind]))



def recon_loss_plot(results_dict):

	for (distr, results) in results_dict.items():
		plt.plot(results[recon_loss_ind], label=distr)

	plt.xlabel("Iterations")
	plt.ylabel("Reconstruction loss")
	plt.title("Reconstruction loss")
	plt.legend()
	plt.show()


def kld_plot(results_dict):

	for (distr, results) in results_dict.items():
		plt.plot(results[kld_ind], label=distr)

	plt.xlabel("Iterations")
	plt.ylabel("KLD")
	plt.title("KLD")
	plt.legend()
	plt.show()

def style_accu_plot(results_dict):

	for (distr, results) in results_dict.items():
		plt.plot(results[style_accu_ind], label=distr)

	plt.xlabel("Iterations")
	plt.ylabel("Style accuracy")
	plt.title("Style accuracy")
	plt.legend()
	plt.show()

def ind_loss_plot(results_dict):

	for (distr, results) in results_dict.items():
		plt.plot(results[ind_loss_ind], label=distr)

	plt.xlabel("Iterations")
	plt.ylabel("Ind loss")
	plt.title("Ind loss")
	plt.legend()
	plt.show()

def style_recon_loss_plot(results_dict):

	for (distr, results) in results_dict.items():
		plt.plot(results[style_recon_loss_ind], label=distr)

	plt.xlabel("Iterations")
	plt.ylabel("Style reconstruction loss")
	plt.title("Style reconstruction loss")
	plt.legend()
	plt.show()

recon_loss_plot(results)
kld_plot(results)
style_accu_plot(results)
ind_loss_plot(results)
style_recon_loss_plot(results)



