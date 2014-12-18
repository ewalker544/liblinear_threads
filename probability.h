#ifndef _PROBABILITY_H_
#define _PROBABILITY_H_

static void sigmoid_train(
    int l, const double *dec_values, const double *labels,
    double& A, double& B)
{
    double prior1=0, prior0 = 0;
    int i;

    for (i=0;i<l;i++)
        if (labels[i] > 0) prior1+=1;
        else prior0+=1;

    int max_iter=100;   // Maximal number of iterations
    double min_step=1e-10;  // Minimal step taken in line search
    double sigma=1e-12; // For numerically strict PD of Hessian
    double eps=1e-5;
    double hiTarget=(prior1+1.0)/(prior1+2.0);
    double loTarget=1/(prior0+2.0);
    double *t=Malloc(double,l);
    double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
    double newA,newB,newf,d1,d2;
    int iter;

    // Initial Point and Initial Fun Value
    A=0.0; B=log((prior0+1.0)/(prior1+1.0));
    double fval = 0.0;
    

    for (i=0;i<l;i++)
    {
        if (labels[i]>0) t[i]=hiTarget;
        else t[i]=loTarget;
        fApB = dec_values[i]*A+B;
        if (fApB>=0)
            fval += t[i]*fApB + log(1+exp(-fApB));
        else
            fval += (t[i] - 1)*fApB +log(1+exp(fApB));
    }
    for (iter=0;iter<max_iter;iter++)
    {
        // Update Gradient and Hessian (use H' = H + sigma I)
        h11=sigma; // numerically ensures strict PD
        h22=sigma;
        h21=0.0;g1=0.0;g2=0.0;
        for (i=0;i<l;i++)
        {
            fApB = dec_values[i]*A+B;
            if (fApB >= 0)
            {
                p=exp(-fApB)/(1.0+exp(-fApB));
                q=1.0/(1.0+exp(-fApB));
            }
            else
            {
                p=1.0/(1.0+exp(fApB));
                q=exp(fApB)/(1.0+exp(fApB));
            }
            d2=p*q;
            h11+=dec_values[i]*dec_values[i]*d2;
            h22+=d2;
            h21+=dec_values[i]*d2;
            d1=t[i]-p;
            g1+=dec_values[i]*d1;
            g2+=d1;
        }

        // Stopping Criteria
         if (fabs(g1)<eps && fabs(g2)<eps)
            break;

        // Finding Newton direction: -inv(H') * g
         det=h11*h22-h21*h21;
        dA=-(h22*g1 - h21 * g2) / det;
        dB=-(-h21*g1+ h11 * g2) / det;
        gd=g1*dA+g2*dB;


        stepsize = 1;       // Line Search
        while (stepsize >= min_step)
        {
            newA = A + stepsize * dA;
            newB = B + stepsize * dB;

            // New function value
            newf = 0.0;
            for (i=0;i<l;i++)
            {
                fApB = dec_values[i]*newA+newB;
                if (fApB >= 0)
                    newf += t[i]*fApB + log(1+exp(-fApB));
                else
                    newf += (t[i] - 1)*fApB +log(1+exp(fApB));
            }
            // Check sufficient decrease
            if (newf<fval+0.0001*stepsize*gd)
            {
                A=newA;B=newB;fval=newf;
                break;
            }
            else
                stepsize = stepsize / 2.0;
        }

        if (stepsize < min_step)
        {
            info("Line search fails in two-class probability estimates\n");
            break;
        }
    }

    if (iter>=max_iter)
        info("Reaching maximal iterations in two-class probability estimates\n");
    free(t);
}

static double sigmoid_predict(double decision_value, double A, double B)
{
	double fApB = decision_value*A+B;
	// 1-p used later; avoid catastrophic cancellation
	if (fApB >= 0)
		return exp(-fApB)/(1.0+exp(-fApB));
	else
		return 1.0/(1+exp(fApB)) ;
}

static void multiclass_probability(int k, double **r, double *p)
{
	int t,j;
	int iter = 0, max_iter=max(100,k);
	double **Q=Malloc(double *,k);
	double *Qp=Malloc(double,k);
	double pQp, eps=0.005/k;

	for (t=0;t<k;t++)
	{
		p[t]=1.0/k;  // Valid if k = 1
		Q[t]=Malloc(double,k);
		Q[t][t]=0;
		for (j=0;j<t;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=Q[j][t];
		}
		for (j=t+1;j<k;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=-r[j][t]*r[t][j];
		}
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp=0;
		for (t=0;t<k;t++)
		{
			Qp[t]=0;
			for (j=0;j<k;j++)
				Qp[t]+=Q[t][j]*p[j];
			pQp+=p[t]*Qp[t];
		}
		double max_error=0;
		for (t=0;t<k;t++)
		{
			double error=fabs(Qp[t]-pQp);
			if (error>max_error)
				max_error=error;
		}
		if (max_error<eps) break;

		for (t=0;t<k;t++)
		{
			double diff=(-Qp[t]+pQp)/Q[t][t];
			p[t]+=diff;
			pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
			for (j=0;j<k;j++)
			{
				Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
				p[j]/=(1+diff);
			}
		}
	}
	if (iter>=max_iter)
		info("Exceeds max_iter in multiclass_prob\n");
	for(t=0;t<k;t++) free(Q[t]);
	free(Q);
	free(Qp);
}

#endif
