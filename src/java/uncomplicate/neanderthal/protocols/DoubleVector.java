package uncomplicate.neanderthal.protocols;

public interface DoubleVector extends Vector {

    double entry (long i);
    
    double dot (DoubleVector y);
    
    double nrm2 ();
    
    double asum ();
    
    DoubleVector rot (DoubleVector y, 
                      double c, double s);
    
    DoubleVector rotg (DoubleVector b, 
                       DoubleVector c, 
                       DoubleVector s);
    
    DoubleVector rotmg (DoubleVector d2, 
                        DoubleVector b1, 
                        DoubleVector b2, 
                        DoubleVector p);
    
    DoubleVector rotm (DoubleVector y,
                       DoubleVector p);
    
    DoubleVector scal (double alpha);
    
    DoubleVector axpy (double alpha,
                       DoubleVector y);
                       
}
