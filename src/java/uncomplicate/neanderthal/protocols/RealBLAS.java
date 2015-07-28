package uncomplicate.neanderthal.protocols;

public interface RealBLAS extends BLAS {

    double dot (Block x, Block y);

    double nrm2 (Block x);

    double asum (Block x);

    Block rot (Block x, Block y, double c, double s);

    Block scal (double alpha, Block x);

    Block axpy (double alpha, Block x, Block y);

    Block mv (double alpha, Block a, Block x, double beta, Block y);

    Block rank (Block a, double alpha, Block x, Block y);

    Block mm (double alpha, Block a, Block b, double beta, Block c);

}
