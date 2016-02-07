package uncomplicate.neanderthal.protocols;

public interface BLASPlus extends BLAS {

    Object sum (Block x);
    long imax (Block x);
    long imin (Block x);

}
