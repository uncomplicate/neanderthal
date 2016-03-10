package uncomplicate.neanderthal.protocols;

public interface BLASPlus extends BLAS {

    Object sum (Block x);
    long imax (Block x);
    long imin (Block x);
    void subcopy (Block x, Block y, long kx, long lx, long ky);
}
