package uncomplicate.neanderthal.protocols;

public interface DoubleMatrix extends Matrix {

    double entry (long i, long j);

    DoubleVector mv (double alpha, 
                     DoubleVector x,
                     double beta,
                     DoubleVector y,
                     long transa);

    DoubleMatrix mm (double alpha,
                     DoubleMatrix b,
                     double beta,
                     DoubleMatrix c, 
                     long transa, long transb);
}
