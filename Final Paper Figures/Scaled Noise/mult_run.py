import numpy as np
import matplotlib.pyplot as plt
import utils.utils_eps as ute
import utils.forward_models as g
import scipy.stats as st
import warnings
#import h5py
from mxmc import Optimizer
from mxmc import Estimator
from mxmc import OutputProcessor
#%%
if __name__ == '__main__':
    #  set up problem parameters, data structures
    def xuns_function(theta, d):
        first_term = theta**3*d**2
        second_term = theta*np.exp(-1*np.abs(0.2-d))
        return first_term + second_term
    
    def xun2(theta, d):
        first_term = (0.5**0.5)*(1.**0.25)*theta**2.5*d**1.75
        second_term = theta*np.exp(-1*np.abs(0.2-d))
        return first_term + second_term
    
    def xun3(theta, d):
        first_term = 0.5*(1.**0.5)*theta**2*d**1.5
        second_term = theta*np.exp(-1*np.abs(0.2-d))
        return first_term + second_term
    
    lb = 0; ub = 1 # bounds on design space, d
    lb_theta = 0.1; ub_theta = 1 # bounds on uniform prior of theta
    n_param = 1 # dimension of theta
    n_LF = 2 # number of LF models
    g_func = [xuns_function, xun2, xun3]
    target_cost = 100 # total budget
    
    #%% run OED using optimal ACV estimator
    dcount=0
    n_d=40
    d_vals = np.linspace(lb, ub, n_d+1)
    totalRuns = 50
    #n_in = [2500,221,733] # reuse0 case
    #n_in = [2500,1586,562] # reuse1 case
    #n_in = [2500,903,733] # reuse2 case
    u_d = np.zeros((totalRuns,n_d+1,3)); u_d_DNMC = np.zeros((totalRuns,n_d+1,3));
    reuse=0
    # 0: random sampling; 1: outer and inner thetas are identical; 
    # 2: random sampling but inner are same across fidelities; 
    
    #d_test = np.random.uniform(lb, ub, test_size)#1*np.ones((test_size,))#np.random.uniform(lb, ub, test_size)#np.linspace(lb, ub, test_size)
    intervals=n_d; perInt = 500; d_ints = np.linspace(lb,ub,intervals)
    test_size = intervals*perInt
    thetas_test = ute.sample_prior(test_size, n_param, lb_theta, ub_theta).flatten()
    eps_test = ute.sample_epsilon(test_size)
    covs_all = np.zeros((n_LF+1,n_LF+1,intervals))
    avgs = np.zeros((n_LF+1,intervals))
    d_test = np.zeros((test_size,))
    for j in range(intervals):
        d_test[j*perInt:(j+1)*perInt] = d_ints[j]*np.ones((perInt,))
    y = np.zeros((n_LF+1, test_size))
    
    for reuse in [1]: # 0 means no reuse, 1 means across-fid reuse, 2 means inner/outer reuse
        count2=0
        print("reuse identifier # "+str(reuse))
        n_in = [2500,2500,2500]#!!!
        w = np.array([1, 0.1, 0.01])*np.divide(n_in,[2500,2500,2500]) # user-defined cost vector
        #print(w)
        for level in range(n_LF+1):
            g_test = g_func[level]
            g_outer = g_test(thetas_test,d_test)
            for ii in range(test_size):
                if reuse==2: # would need to update this if using reuse==2
                    thetas_inner = ute.sample_prior(int(n_in[level]), n_param, lb_theta, ub_theta,seed=ii)
                elif reuse==1:
                    thetas_inner = ute.sample_prior(int(n_in[level]), n_param, lb_theta, ub_theta,seed=ii)
                else:
                    thetas_inner = ute.sample_prior(int(n_in[level]), n_param, lb_theta, ub_theta,seed=ii+level*test_size)
                g_inner  = g_test(thetas_inner.flatten(),d_test[ii])
                #PCE VS NON PCE CHOICE:
                #y[level, ii] = ute.eig_eps_pce(eps_test[ii],1,int(n_in[level]),g_inner,np.array(g_outer[ii],ndmin=1))
                y[level, ii] = ute.eig_eps_mult(eps_test[ii],1,int(n_in[level]),g_inner,np.array(g_outer[ii],ndmin=1))
        
        #WEIGHTED VS NON WEIGHTED CHOICE:
        for j in range(intervals):
            covs_all[:,:,j] = np.cov(y[:,j*perInt:(j+1)*perInt])
            avgs[:,j] = np.average(y[:,j*perInt:(j+1)*perInt],axis=1)
        first_term = np.average(covs_all,axis=2)
        second_term = np.cov(avgs)
        cov_test=first_term
        #cov_test = np.cov(y,aweights=np.average(abs(y),axis=0))
        #if d==0:
            #print(np.cov(y))
            #print(cov_test)
        
        # determine optimal estimator using MXMCPy (Bomarito et al)
        variance_results = dict()
        sample_allocation_results = dict()
        
        mxmc_optimizer = Optimizer(w, np.transpose(cov_test))
        
        algorithms = Optimizer.get_algorithm_names()
        for algorithm in algorithms:
        
            opt_result = mxmc_optimizer.optimize(algorithm, target_cost)
            variance_results[algorithm] = opt_result.variance
            sample_allocation_results[algorithm] = opt_result.allocation
        
            #print("{} method variance: {}".format(algorithm, opt_result.variance))
        
        best_method = min(variance_results, key=variance_results.get)
        sample_allocation = sample_allocation_results[best_method]
        
        print("Best method: ", best_method,opt_result.variance)
        estimator = Estimator(sample_allocation, cov_test)
            
            #print(np.corrcoef(y))
           
        for d in d_vals:#np.transpose(n_inVec2):
            print(d)
            for run in range(totalRuns):
                if run%10==0:
                    print(run)
                seed = run # fixing seeds across d
                #dcount*totalRuns + run # linear indexing random seed for unique samples
                # print(run, seed)
                #  allocating samples
                total_samples = sample_allocation.num_total_samples
                epsilons_all = ute.sample_epsilon(total_samples,seed=seed)
                thetas_all = ute.sample_prior(total_samples, n_param, lb_theta, ub_theta,seed=seed)
                thetas_inner_all = ute.sample_prior(np.max(n_in), n_param, lb_theta, ub_theta,seed=seed+totalRuns) # guarantees different from theta_all
                epsilons_allocated = sample_allocation.allocate_samples_to_models(epsilons_all)
                thetas_allocated = sample_allocation.allocate_samples_to_models(thetas_all)
                
                #  evaluating models for above samples
                model_outputs = list()
                level=0
                for theta_sample, epsilon_sample, model in zip(thetas_allocated, epsilons_allocated, g_func):
                    n_samples = len(theta_sample)
                    #if run==0:
                        #print(level, n_samples)
                    if reuse==2:
                        thetas_inner = thetas_inner_all[:n_in[level]]
                    elif reuse==1:
                        if n_samples >= n_in[level]:
                            thetas_inner = np.vstack((thetas_inner_all,ute.sample_prior(n_samples-n_in[level], n_param, lb_theta, ub_theta,seed=np.random.randint(999999))))
                        else:
                            thetas_inner = thetas_inner_all[:n_in[level]]
                    else:
                        thetas_inner = ute.sample_prior(int(n_in[level]), n_param, lb_theta, ub_theta,seed=np.random.randint(999999))
                    
                    #print("model evals g_inner g_outer")
                    g_outer = model(theta_sample,d)
                    g_inner = model(thetas_inner,d)
                    #print("eig computation")
                    #PCE VS NON PCE CHOICE:
                    #eig_outputs = ute.eig_eps_pce_vec(epsilon_sample,n_samples,n_in[level],g_inner,g_outer)
                    eig_outputs = ute.eig_eps_mult_vec(epsilon_sample,n_samples,n_in[level],g_inner,g_outer)
                    model_outputs.append(eig_outputs)
                    level+=1
            
                #  forming and evaluating estimator
                #print("evaluating estimator")
                u_d[run,count2,reuse] = estimator.get_estimate(model_outputs)
                #print("u_d="+str(u_d[run,count2,reuse])[:6])
                
                n_out = int(target_cost/w[0])
               
                #print("DNMC begins")
                thetas_outer_DNMC = ute.sample_prior(n_out, n_param, lb_theta, ub_theta,seed=seed)
                if reuse==0:
                    thetas_inner_DNMC = ute.sample_prior(int(n_in[0]), n_param, lb_theta, ub_theta,seed=np.random.randint(999999))
                else:
                    thetas_inner_DNMC = thetas_inner_all[:n_in[0]]
                eps_outer_DNMC = ute.sample_epsilon(n_out,seed=seed)
                #print("DNMC model evals g_inner g_outer")
                g_outer_DNMC = g_func[0](thetas_outer_DNMC,d)
                g_inner_DNMC = g_func[0](thetas_inner_DNMC,d)
                #PCE VS NON PCE CHOICE:
                #u_d_DNMC[run,count2,reuse] = np.average(ute.eig_eps_pce_vec(epsilons_all[:n_out],n_out,n_in[0],g_inner_DNMC,g_outer_DNMC))     
                u_d_DNMC[run,count2,reuse] = ute.eig_eps_mult(eps_outer_DNMC,n_out,n_in[0],g_inner_DNMC,g_outer_DNMC)       
                #print("u_d_DNMC="+str(u_d_DNMC[run,count2,reuse])[:6])
            count2+=1

    #%%
    plt.rcParams['figure.dpi'] = 1200
    avg_ud = np.average(u_d,axis=0)
    avg_ud_DNMC = np.average(u_d_DNMC,axis=0)
    std_ud = np.std(u_d,axis=0)
    std_ud_DNMC = np.std(u_d_DNMC,axis=0)
    
    fig, ax1 = plt.subplots()
    for kk in range(totalRuns):  
        if kk==totalRuns-1:
            ax1.plot(d_vals,u_d_DNMC[kk,:,1],linewidth=0.25, color='red', label='DNMC estimates') 
            ax1.plot(d_vals,u_d[kk,:,1],linewidth=0.25, color='green',label='MF estimates')
        else:
            ax1.plot(d_vals,u_d_DNMC[kk,:,1],linewidth=0.25, color='red') 
            ax1.plot(d_vals,u_d[kk,:,1],linewidth=0.25, color='green')
            
    ax1.plot(d_vals,avg_ud[:,1],color='green', linewidth=1.5,label='Average MF estimate')
    ax1.plot(d_vals,avg_ud_DNMC[:,1],color='red', linewidth=1.5, label='Average DNMC estimate')
    ax1.set_title('DNMC vs. '+best_method+' method estimator of EIG')
    ax1.set_xlabel(r'$\xi$')
    ax1.set_ylabel(r'$U(d)$')
    ax1.legend()
    
    #%%
    u_d_r0 = u_d[:,:,0]; u_d_r1 = u_d[:,:,1]; u_d_r2 = u_d[:,:,2]
    var_r0 = np.var(u_d_r0,axis=0); var_r1 = np.var(u_d_r1,axis=0); var_r2 = np.var(u_d_r2,axis=0)
    var_ud_DNMC = np.var(u_d_DNMC[:,:,1],axis=0)
    avg_ud_DNMC = np.average(u_d_DNMC[:,:,1])
    fig4, ax4 = plt.subplots()
    ax4.semilogy(d_vals, var_ud_DNMC, color='red', label='NMC')
    #ax4.semilogy(d_vals, var_r0, color='blue', label='no reuse')
    ax4.semilogy(d_vals, var_r1, color='green', label='MF-EIG')
    ax4.set_xlabel(r'$\xi$')
    ax4.set_ylabel(r'$\mathrm{\mathbb{V}}\text{ar}[U]$')
    ax4.legend()
    plt.savefig("var_across_d_mult.pdf")
    #ax4.set_title(r'Estimator variances across reuse methods using opt. $N_{in}$')
    #fig.savefig("vars_optnin_testsize1000.jpg",dpi=1200)
    
    #%%
    u_d_unweighted = np.load('../Data/runs_ud/u_d_allreuses.npy')
    u_d_r0_uw = u_d_unweighted[:,:,0]; u_d_r1_uw = u_d_unweighted[:,:,1]; u_d_r2_uw = u_d_unweighted[:,:,2]
    var_r0_uw = np.var(u_d_r0_uw,axis=0); var_r1_uw = np.var(u_d_r1_uw,axis=0); var_r2_uw = np.var(u_d_r2_uw,axis=0)
    figur,axur = plt.subplots()
    axur.plot(d_vals, var_r0, color='blue', label='no reuse')
    axur.plot(d_vals, var_r0_uw, color='blue', linestyle='dashed')
    axur.plot(d_vals, var_r1, color='green', label='inner-outer reuse')
    axur.plot(d_vals, var_r1_uw, color='green',linestyle='dashed')
    axur.plot(d_vals, var_r2, color='purple', label='across-fid reuse')
    axur.plot(d_vals, var_r2_uw, color='purple', linestyle='dashed')
    axur.set_title("Comparing variances of cov and reuse methods")
    axur.legend()
    #%%
    figur1,axur1 = plt.subplots()
    axur1.plot(d_vals, var_r1, color='green', label='unweighted')
    axur1.plot(d_vals, var_r1_uw, color='yellow', label='weighted')
    axur1.set_title("Comparing variances of cov methods, inner/outer shared")
    axur1.legend()
    #%%
    figur2,axur2 = plt.subplots()
    axur2.plot(d_vals, var_r2, color='purple', label='unweighted')
    axur2.plot(d_vals, var_r2_uw, color='orange', label='weighted')
    axur2.set_title("Comparing variances of cov methods, inner across fids shared")
    axur2.legend()
    #%%
    std_r0 = np.std(u_d_r0,axis=0); std_r1 = np.std(u_d_r1,axis=0); std_r2 = np.std(u_d_r2,axis=0)
    std_ud_DNMC_r0 = np.std(u_d_DNMC[:,:,0],axis=0); std_ud_DNMC_r1 = np.std(u_d_DNMC[:,:,1],axis=0); std_ud_DNMC_r2 = np.std(u_d_DNMC[:,:,2],axis=0)
    avg_r0 = np.average(u_d_r0,axis=0); avg_r1 = np.average(u_d_r1,axis=0); avg_r2 = np.average(u_d_r2,axis=0)
    avg_ud_DNMC_r0 = np.average(u_d_DNMC[:,:,0],axis=0); avg_ud_DNMC_r1 = np.average(u_d_DNMC[:,:,1],axis=0); avg_ud_DNMC_r2 = np.average(u_d_DNMC[:,:,2],axis=0)
    
    #%%
    #u_d0 = np.load('uD_nIn2500_nOut10K.npy')
    fig2, ax2 = plt.subplots()
    #ax2.fill_between(d_vals,avg_r0-std_r0,avg_r0+std_r0, alpha=0.3, color='blue',label='no reuse')
    #ax2.fill_between(d_vals,avg_r1-std_r1,avg_r1+std_r1, alpha=0.3, color='green',label='across-fid inner reuse')
    ax2.fill_between(d_vals,avg_ud_DNMC_r1-2*std_ud_DNMC_r1,avg_ud_DNMC_r1+2*std_ud_DNMC_r1, alpha=0.2, color='red',label=r'NMC $\pm 2$ std')
    ax2.fill_between(d_vals,avg_r1-std_r1,avg_r1+std_r1, alpha=0.3, color='green',label=r'MF-EIG $\pm 2$ std')
    #ax2.plot(d_vals,u_d0,linewidth=.5,color='black',linestyle='dashed',label='DNMC, N_in=2.5K, N_out=10K')
    ax2.plot(d_vals,avg_ud_DNMC_r1,color='red', linewidth=0.5, label='NMC mean')
    ax2.plot(d_vals,avg_r1,color='green', linewidth=0.5, label='MF-EIG mean')
    #ax2.plot(d_vals,avg_r1,color='green', linewidth=0.5)
    #ax2.plot(d_vals,avg_r2,color='purple', linewidth=0.5)
    #ax2.set_title('MF-EIG estimator')
    #ax2.set_title('Single- vs. multi-fidelity estimator of EIG')
    ax2.set_xlabel(r'$\xi$')
    ax2.set_ylabel(r'$U$')
    ax2.legend(loc=4)
    plt.savefig("u_d_mult.pdf")
    
    #%%
    figu32,axur3 = plt.subplots()
    axur3.plot(d_vals, var_r2, color='purple', label='unweighted')
    axur2.plot(d_vals, var_r2_uw, color='orange', label='weighted')
    axur2.set_title("Comparing variances of cov methods, inner across fids shared")
    axur2.legend()