package uncomplicate.neanderthal.protocols;
public interface RealVector extends Vector {

    double entry (long i);

    double dot (RealVector y);

    double nrm2 ();

    double asum ();

    RealVector rot (RealVector y,
                    double c, double s);

    RealVector scal (double alpha);

    RealVector axpy (double alpha,
                     RealVector y);

}
