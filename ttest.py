import numpy as np
import scipy.stats as ss

#test along axis 0

def calc_T0(x): #calc integral timescale, used for estimating effective DoF
    xs=x.shape
    xcv=np.reshape(x,(xs[0],xs[1]*xs[2])) # x into column variable form
    r=[]
    for i in range(xcv.shape[1]):
        r.append(np.corrcoef(xcv[1:,i],xcv[:-1,i])[0,1]) #calc lag 1 acf per column
    r=np.array(r)
    r=r.reshape(xs[1],xs[2]) #transform back to original dims
    T0=(1+r)/(1-r)
    return T0

def calc_tstat(X1,X2,X1T0=None,X2T0=None,equalvar=True):
    #base n is number of samples
    X1n=np.tile(X1.shape[0],X1.shape[1:])
    X2n=np.tile(X2.shape[0],X2.shape[1:])
    #if integral timesclaes provided, scale n to get n_eff
    if X1T0 is not None:
        X1n=X1n/X1T0
    if X2T0 is not None:
        X2n=X2n/X2T0

    X1bar=np.mean(X1,axis=0)
    X2bar=np.mean(X2,axis=0)
    X1sig=np.std(X1,ddof=1,axis=0)
    X2sig=np.std(X2,ddof=1,axis=0)
    if equalvar:
    # from wiki: t stat for Equal or unequal sample sizes, similar variances (1/2 < sX1/sX2 < 2)
        #estimator of pooled stddev
        sp=np.sqrt(((X1n-1)*X1sig**2+(X2n-1)*X2sig**2)/(X1n+X2n-2))
        #tstat
        t=(X1bar-X2bar)/(sp*np.sqrt(1/X1n+1/X2n))
    else:
    # from wiki: t stat for Equal or unequal sample sizes, unequal variance variances (Welch's test)
        s_del=np.sqrt(X1sig**2/X1n + X2sig**2/X2n)
        t=(X1bar-X2bar)/s_del
    return t

def calc_nDoF(X1,X2,X1T0=None,X2T0=None,equalvar=True):
    #base n is number of samples
    X1n=np.tile(X1.shape[0],X1.shape[1:])
    X2n=np.tile(X2.shape[0],X2.shape[1:])
    #if integral timesclaes provided, scale n to get n_eff
    if X1T0 is not None:
        X1n=X1n/X1T0
    if X2T0 is not None:
        X2n=X2n/X2T0
    
    if equalvar:
        DoF=X1n+X2n-2
    else:
        X1bar=np.mean(X1,axis=0)
        X2bar=np.mean(X2,axis=0)
        X1sig=np.std(X1,ddof=1,axis=0)
        X2sig=np.std(X2,ddof=1,axis=0)
        DoF=((X1sig**2/X1n + X2sig**2/X2n)**2)/\
            ((X1sig**2/X1n)**2/(X1n-1)+(X2sig**2/X2n)**2/(X2n-1))
    return DoF

def tcdf(x,DoF):
    return ss.t.cdf(x,DoF)
# def tpdf(x,DoF):
#     return ss.t.pdf(x,DoF)

def pval2sided(t,nDoF):
    return(2*(1-tcdf(np.abs(t),nDoF)))

def ttest(X1,X2,equalvar=True,neff=True):
    # print('X1:',X1.shape,'X2:',X2.shape)
    if neff: #calc integral time to correct DoF for auto-correlation
        X1T0=calc_T0(X1)
        X2T0=calc_T0(X2)
        # print('X1T0:',X1T0.shape,'X2T0:',X2T0.shape)
    else:
        X1T0=None
        X2T0=None
    t=calc_tstat(X1,X2,X1T0=X1T0,X2T0=X2T0,equalvar=equalvar)
    # print('t:',t.shape)
    nDoF=calc_nDoF(X1,X2,X1T0=X1T0,X2T0=X2T0,equalvar=equalvar)
    # print('nDoF:',nDoF.shape)
    p=pval2sided(t,nDoF)
    # print('p:',p.shape)
    return p