package uncomplicate.neanderthal.protocols;

public interface RealMatrix extends Matrix {

    double entry (long i, long j);

    RealVector mv (double alpha, 
                     RealVector x,
                     double beta,
                     RealVector y,
                     long transa);

    RealMatrix mm (double alpha,
                     RealMatrix b,
                     double beta,
                     RealMatrix c, 
                     long transa, long transb);
}
