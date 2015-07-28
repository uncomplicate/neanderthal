package uncomplicate.neanderthal.protocols;

public interface BLAS {

    long iamax (Block x);

    Block rotg (Block x);

    Block rotmg (Block p, Block args);

    Block rotm (Block x, Block y, Block p);

    Block swap (Block x, Block y);

    Block copy (Block x, Block y);

}
//TODO matrix can be processed with vector operations if I add offset to CBLAS C code?
