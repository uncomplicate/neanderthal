package uncomplicate.neanderthal.protocols;

public interface RealVector extends Vector {

    double entry (long i);
    
    double dot (RealVector y);
    
    double nrm2 ();
    
    double asum ();
    
    RealVector rot (RealVector y, 
                      double c, double s);
    
    RealVector rotg (RealVector b, 
                       RealVector c, 
                       RealVector s);
    
    RealVector rotmg (RealVector d2, 
                        RealVector b1, 
                        RealVector b2, 
                        RealVector p);
    
    RealVector rotm (RealVector y,
                       RealVector p);
    
    RealVector scal (double alpha);
    
    RealVector axpy (double alpha,
                       RealVector y);
                       
}
