package uncomplicate.neanderthal.protocols;

public interface RealMatrix extends Matrix {

    double entry (long i, long j);

    RealVector mv (double alpha,
                     RealVector x,
                     double beta,
                     RealVector y);

    RealMatrix mm (double alpha,
                     RealMatrix b,
                     double beta,
                     RealMatrix c);

    RealMatrix rank (double alpha,
                     RealVector x,
                     RealVector y);
}
